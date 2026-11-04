"""
Funcții KITTI: descărcare selectivă, calibrare și adâncime reală din LiDAR.

Proiecția LiDAR -> imagine urmează codul public Monodepth2 (generate_depth_map),
folosit de majoritatea lucrărilor de depth pe KITTI.
"""
import os
import zipfile
import urllib.request

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Descărcare
# ---------------------------------------------------------------------------
def _download(url, dest):
    if os.path.exists(dest):
        return dest
    print("  descarc", url)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def download_calib():
    """Descarcă și extrage fișierele de calibrare pentru ziua C.DATE."""
    out = os.path.join(C.RAW_DIR, C.DATE, "calib_cam_to_cam.txt")
    if os.path.exists(out):
        return
    os.makedirs(C.RAW_DIR, exist_ok=True)
    z = _download(f"{C.KITTI_URL}/{C.DATE}_calib.zip",
                  os.path.join(C.RAW_DIR, f"{C.DATE}_calib.zip"))
    with zipfile.ZipFile(z) as f:
        f.extractall(C.RAW_DIR)
    os.remove(z)


def download_drive(drive, need_velodyne):
    """
    Descarcă o secvență și extrage DOAR ce ne trebuie:
    image_02 (camera color stânga), image_03 (dreapta) și, pentru val/test,
    velodyne_points (LiDAR, pentru adâncimea reală). Restul arhivei se ignoră.
    """
    name = C.drive_name(drive)
    folder = os.path.join(C.RAW_DIR, C.DATE, name)
    marker = os.path.join(folder, ".extras_ok")
    if os.path.exists(marker):
        return folder
    os.makedirs(C.RAW_DIR, exist_ok=True)
    zpath = os.path.join(C.RAW_DIR, name + ".zip")
    _download(f"{C.KITTI_URL}/{C.DATE}_drive_{drive}/{name}.zip", zpath)
    keep = ["image_02/data/", "image_03/data/"]
    if need_velodyne:
        keep.append("velodyne_points/data/")
    with zipfile.ZipFile(zpath) as f:
        for m in f.namelist():
            if any(k in m for k in keep):
                f.extract(m, C.RAW_DIR)
    os.remove(zpath)  # arhiva ocupă mult; o ștergem după extragere
    open(marker, "w").close()
    return folder


# ---------------------------------------------------------------------------
# Calibrare
# ---------------------------------------------------------------------------
def read_calib_file(path):
    data = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(":")
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass  # câmpuri text (ex. data calibrării)
    return data


def camera_params():
    """
    Returnează fx (pixeli), baseline B (metri) și dimensiunea originală
    a imaginilor pentru camera 2 (stânga color) și camera 3 (dreapta color).
    """
    cam = read_calib_file(os.path.join(C.RAW_DIR, C.DATE, "calib_cam_to_cam.txt"))
    P2 = cam["P_rect_02"].reshape(3, 4)
    P3 = cam["P_rect_03"].reshape(3, 4)
    fx = P2[0, 0]
    # P[0,3] = -fx * tx  =>  tx = -P[0,3]/fx ; baseline = diferența dintre camere
    baseline = abs(P2[0, 3] / P2[0, 0] - P3[0, 3] / P3[0, 0])
    w, h = cam["S_rect_02"].astype(int)
    return float(fx), float(baseline), int(w), int(h)


# ---------------------------------------------------------------------------
# Adâncime reală din LiDAR (pentru evaluare)
# ---------------------------------------------------------------------------
def load_velodyne(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts[:, 3] = 1.0
    return pts


def velodyne_to_depth(velo_path, cam=2):
    """Proiectează punctele LiDAR în imaginea camerei `cam` -> hartă de adâncime rară."""
    calib_dir = os.path.join(C.RAW_DIR, C.DATE)
    cam2cam = read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
    velo2cam = read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
    velo2cam = np.hstack((velo2cam["R"].reshape(3, 3), velo2cam["T"][..., None]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    w, h = cam2cam["S_rect_02"].astype(int)
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
    P_rect = cam2cam[f"P_rect_0{cam}"].reshape(3, 4)
    P_velo2im = P_rect @ R_cam2rect @ velo2cam

    velo = load_velodyne(velo_path)
    velo = velo[velo[:, 0] >= 0, :]          # doar punctele din fața mașinii

    pts = (P_velo2im @ velo.T).T
    pts[:, :2] = pts[:, :2] / pts[:, 2][..., None]
    pts[:, 2] = velo[:, 0]                   # adâncimea = distanța pe axa x a LiDAR-ului

    pts[:, 0] = np.round(pts[:, 0]) - 1
    pts[:, 1] = np.round(pts[:, 1]) - 1
    ok = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < w) & (pts[:, 1] < h)
    pts = pts[ok]

    depth = np.zeros((h, w), dtype=np.float32)
    xs, ys = pts[:, 0].astype(int), pts[:, 1].astype(int)
    depth[ys, xs] = pts[:, 2]
    # Dacă două puncte cad în același pixel, păstrăm cel mai apropiat.
    flat = ys * w + xs
    uniq, counts = np.unique(flat, return_counts=True)
    for idx in uniq[counts > 1]:
        sel = np.where(flat == idx)[0]
        depth[ys[sel[0]], xs[sel[0]]] = pts[sel, 2].min()
    depth[depth < 0] = 0
    return depth
