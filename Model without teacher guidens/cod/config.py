"""
Configurare comună pentru tot proiectul (date, teacher, student, export).

AICI este "contractul modelului": dacă schimbi ceva aici, schimbi pentru toți.
Orice script citește valorile de aici, ca să nu existe două versiuni diferite.
"""
import os

# ---------------------------------------------------------------------------
# Căi. În Colab, montează Google Drive și lasă valoarea implicită, ca datele
# să nu se piardă când se închide sesiunea. Poți schimba cu variabila ECA_ROOT.
# ---------------------------------------------------------------------------
ROOT = os.environ.get("ECA_ROOT", "/content/drive/MyDrive/ECA_depth")
# KITTI brut: pe discul local Colab (rapid, temporar). Nu pe Drive: mii de
# fișiere PNG mici se scriu foarte încet pe Google Drive.
RAW_DIR = os.environ.get("ECA_RAW", "/content/kitti_raw")
PROC_DIR = os.path.join(ROOT, "proc")      # date procesate (.npz per secvență)
RUNS_DIR = os.path.join(ROOT, "runs")      # modele antrenate + loguri
EXPORT_DIR = os.path.join(ROOT, "export")  # fișiere .tflite pentru placă

# ---------------------------------------------------------------------------
# Date KITTI (raw, sincronizate și rectificate). Toate secvențele sunt din
# aceeași zi (2011_09_26) => un singur fișier de calibrare.
# Împărțire PE SECVENȚE (nu pe cadre), ca train/val/test să nu se amestece.
# Secvențele de test sunt din lista Eigen de test (folosită în literatură),
# dar folosim TOATE cadrele lor, deci cifrele NU sunt direct comparabile cu
# tabelele din articole (acolo se folosesc doar 697 de cadre fixe).
# ---------------------------------------------------------------------------
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
DATE = "2011_09_26"

SPLITS = {
    "train": ["0001", "0005", "0011", "0014", "0015", "0017", "0018",
              "0035", "0039", "0057", "0060", "0091", "0095", "0104"],
    "val":   ["0079", "0113"],
    "test":  ["0002", "0013", "0020", "0046", "0048", "0052", "0106"],
}
# Mod de test rapid (ECA_DEBUG=1): 2 secvențe mici, doar ca să verifici că
# tot lanțul merge. În acest mod val = test, deci cifrele NU se raportează.
if os.environ.get("ECA_DEBUG") == "1":
    SPLITS = {"train": ["0001"], "val": ["0048"], "test": ["0048"]}

# Cadrele consecutive sunt aproape identice; păstrăm fiecare al N-lea cadru.
FRAME_STEP = {"train": 1, "val": 2, "test": 2}

# ---------------------------------------------------------------------------
# Contractul modelului (identic pentru baseline și pentru varianta ghidată).
#   Intrare : 1 x 96 x 320 x 3, RGB, float în [0, 255] (fără normalizare;
#             MobileNetV3 are rescalarea inclusă). La int8: x_q = round(x/s)+z.
#   Ieșire  : 1 x 48 x 160 x 1, valoare s în [0, 1] (sigmoid).
#   Decodare: q = Q_MIN * (Q_MAX / Q_MIN) ** s  (disparitate normalizată q = d / W)
#             adâncime Z = fx * B / (q * W)
# De ce spațiu logaritmic: la int8 ieșirea are doar 256 de niveluri. Într-o
# scară liniară, obiectele îndepărtate ar avea 1-2 niveluri; în scară log,
# fiecare nivel înseamnă ~2% din adâncime, la orice distanță.
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 96, 320
OUT_H, OUT_W = 48, 160
Q_MIN, Q_MAX = 0.001, 0.3   # q=0.001 ~ 310 m, q=0.3 ~ 1 m (la camera KITTI)

# Evaluare (convenția standard KITTI)
MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0

# ---------------------------------------------------------------------------
# Antrenare
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
SEED = 42
W_SMOOTH = 1e-3     # greutatea pierderii de netezime
W_DISTILL = 1.0     # greutatea pierderii teacher (doar în modul "ghidat")
SSIM_ALPHA = 0.85   # combinația SSIM / L1 din Monodepth

# Teacher oficial GMStereo, deja ajustat pe KITTI. Îl folosim doar la inferență.
TEACHER_ID = "gmstereo_kitti15_04487ebf"
TEACHER_SUFFIX = "_teacher_" + TEACHER_ID + ".npz"
TEACHER_CKPT_URL = ("https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/"
                    "pretrained/gmstereo-scale2-regrefine3-resumeflowthings-"
                    "kitti15-04487ebf.pth")
TEACHER_SIZE = (384, 1248)    # rezoluția la care rulează teacher-ul (multiplu de 32)
LR_CHECK_PX = 1.0             # prag consistență stânga-dreapta (pixeli)
LR_CHECK_REL = 0.05           # + 5% din disparitate


def drive_name(drive):
    return f"{DATE}_drive_{drive}_sync"


def all_drives():
    return [(split, d) for split, ds in SPLITS.items() for d in ds]
