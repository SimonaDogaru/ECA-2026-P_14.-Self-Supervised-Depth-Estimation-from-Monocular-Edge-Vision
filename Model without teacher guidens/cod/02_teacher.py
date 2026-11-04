"""
FAZA 3 - Teacher: GMStereo pre-antrenat (UniMatch), rulat O SINGURĂ DATĂ.

Pentru fiecare pereche stânga/dreapta:
  1. prezice disparitatea pentru camera stângă ȘI pentru cea dreaptă
     (truc UniMatch: dreapta = teacher pe imaginile oglindite și inversate);
  2. verificare de consistență stânga-dreapta: un pixel e de încredere doar dacă
     cele două predicții sunt de acord -> masca de validitate;
  3. transformă în disparitate normalizată q = d / W (nu depinde de rezoluție);
  4. micșorează q și masca la 96 x 320 și le salvează: PROC_DIR/<secvență>_teacher.npz

Necesită PyTorch și repo-ul UniMatch clonat (vezi notebooks/A_date_si_teacher.ipynb).
Rulează pe GPU în Colab. Pe CPU merge doar pentru câteva cadre (test).

    python 02_teacher.py --splits train val test
"""
import argparse
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C

UNIMATCH_DIR = os.environ.get("UNIMATCH_DIR", "/content/unimatch")
sys.path.insert(0, UNIMATCH_DIR)
from unimatch.unimatch import UniMatch  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_teacher(device):
    # Configurația EXACTĂ a checkpoint-ului gmstereo-scale2-regrefine3
    # (scripts/gmstereo_demo.sh din UniMatch). Nu amesteca setări între modele.
    model = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4,
                     num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
                     reg_refine=True, task="stereo").to(device)
    ckpt_path = os.path.join(C.ROOT, C.TEACHER_ID + ".pth")
    if not os.path.exists(ckpt_path):
        os.makedirs(C.ROOT, exist_ok=True)
        urllib.request.urlretrieve(C.TEACHER_CKPT_URL, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()  # teacher-ul e ÎNGHEȚAT: fără antrenare, fără gradient
    return model


def to_tensor(path, device):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return ((t - IMAGENET_MEAN) / IMAGENET_STD).to(device)


@torch.no_grad()
def predict_bidir(model, left, right):
    """Returnează disparitatea stânga și dreapta (pixeli), la rezoluția TEACHER_SIZE."""
    size = C.TEACHER_SIZE
    left = F.interpolate(left, size=size, mode="bilinear", align_corners=True)
    right = F.interpolate(right, size=size, mode="bilinear", align_corners=True)
    # perechea 2: (dreapta oglindită, stânga oglindită) -> disparitatea camerei drepte
    L = torch.cat([left, torch.flip(right, dims=[3])], 0)
    R = torch.cat([right, torch.flip(left, dims=[3])], 0)
    disp = model(L, R, attn_type="self_swin2d_cross_swin1d",
                 attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
                 prop_radius_list=[-1, 1], num_reg_refine=3,
                 task="stereo")["flow_preds"][-1]          # [2, H, W]
    d_left = disp[0]
    d_right = torch.flip(disp[1], dims=[1])
    return d_left, d_right


def lr_consistency_mask(d_left, d_right):
    """
    Pixelul x din stânga corespunde pixelului x - d_left(x) din dreapta.
    Acolo, disparitatea dreaptă ar trebui să fie aproape aceeași. Dacă nu,
    pixelul e ocluzat sau potrivirea a eșuat -> îl excludem din antrenare.
    """
    H, W = d_left.shape
    xs = torch.arange(W, device=d_left.device, dtype=d_left.dtype).view(1, W).expand(H, W)
    xr = xs - d_left
    # eșantionare liniară a lui d_right în poziția xr (grid_sample cere [-1, 1])
    gx = 2.0 * xr / (W - 1) - 1.0
    gy = torch.linspace(-1, 1, H, device=d_left.device).view(H, 1).expand(H, W)
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    d_r_at = F.grid_sample(d_right[None, None], grid, align_corners=True)[0, 0]
    diff = (d_left - d_r_at).abs()
    ok = (diff < C.LR_CHECK_PX + C.LR_CHECK_REL * d_left)
    ok &= torch.isfinite(d_left) & (d_left > 0.5) & (xr >= 0)
    return ok


def to_student_grid(q, mask):
    """
    q și mască -> 96 x 320. Harta q se păstrează COMPLETĂ (pentru evaluarea
    teacher-ului); masca spune unde ținta e de încredere (doar acolo se
    aplică pierderea de distilare). Un pixel mic e valid dacă >90% din
    pixelii mari pe care îi acoperă au trecut verificarea.
    """
    q, m = q[None, None], mask[None, None].float()
    size = (C.IMG_H, C.IMG_W)
    m_area = F.interpolate(m, size=size, mode="area")
    q_area = F.interpolate(q, size=size, mode="area").clamp(min=C.Q_MIN, max=C.Q_MAX)
    valid = (m_area > 0.9) & (q_area > C.Q_MIN) & (q_area < C.Q_MAX)
    return q_area[0, 0], valid[0, 0]


def process_drive(model, device, drive, max_frames=None):
    name = C.drive_name(drive)
    out_path = os.path.join(C.PROC_DIR, name + C.TEACHER_SUFFIX)
    if os.path.exists(out_path):
        print(f"{name}: există deja, sar peste")
        return
    proc = np.load(os.path.join(C.PROC_DIR, name + ".npz"))
    frames = proc["frames"][:max_frames] if max_frames else proc["frames"]
    folder = os.path.join(C.RAW_DIR, C.DATE, name)
    # Colab șterge discul local la resetare, dar datele procesate rămân pe Drive.
    # Refacem imaginile originale doar dacă lipsesc pentru o secvență neterminată.
    missing = any(
        not os.path.isfile(os.path.join(folder, camera, "data", f"{int(fr):010d}.png"))
        for fr in frames for camera in ("image_02", "image_03")
    )
    if missing:
        import kitti
        print(f"{name}: lipsesc imagini originale; le descarc din nou")
        marker = os.path.join(folder, ".extras_ok")
        if os.path.exists(marker):
            os.remove(marker)
        kitti.download_drive(drive, need_velodyne=False)

    Q, M = [], []
    t0 = time.time()
    for i, fr in enumerate(frames):
        lp = os.path.join(folder, "image_02", "data", f"{fr:010d}.png")
        rp = os.path.join(folder, "image_03", "data", f"{fr:010d}.png")
        d_left, d_right = predict_bidir(model, to_tensor(lp, device), to_tensor(rp, device))
        mask = lr_consistency_mask(d_left, d_right)
        q = d_left / C.TEACHER_SIZE[1]           # disparitate normalizată
        q_s, m_s = to_student_grid(q, mask)
        Q.append(q_s.cpu().numpy().astype(np.float16))
        M.append(m_s.cpu().numpy().astype(np.uint8))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames)}  ({(time.time() - t0) / (i + 1):.2f} s/cadru)")

    Q, M = np.stack(Q), np.stack(M)
    tmp_path = out_path + ".partial.npz"
    np.savez_compressed(tmp_path, q=Q, mask=M, frames=frames,
                        teacher_id=C.TEACHER_ID, checkpoint_url=C.TEACHER_CKPT_URL)
    os.replace(tmp_path, out_path)
    print(f"{name}: {len(frames)} cadre, pixeli valizi {100 * M.mean():.1f}%, "
          f"{(time.time() - t0) / len(frames):.2f} s/cadru -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispozitiv teacher:", device)
    model = load_teacher(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher GMStereo: {n_params / 1e6:.2f} M parametri")
    for split in args.splits:
        for drive in C.SPLITS[split]:
            process_drive(model, device, drive, args.max_frames)


if __name__ == "__main__":
    main()
