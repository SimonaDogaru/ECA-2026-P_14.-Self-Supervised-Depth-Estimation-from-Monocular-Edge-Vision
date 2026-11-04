# Cod – Depth monocular cu teacher stereo, pe NPU-ul i.MX 8M Plus

Tot lanțul software, de la datele KITTI până la inferența pe NPU. Testat cap-coadă pe date KITTI reale (în modul `ECA_DEBUG=1`), cu TensorFlow 2.19.0, aceeași versiune ca TFLite 2.19.0 de pe placă.

## Ordinea pașilor

| Pas | Unde | Ce rulezi | Rezultat |
|---|---|---|---|
| 0 | Colab | urci folderul `cod/` în `MyDrive/ECA_depth/cod` | – |
| 1 | Colab GPU | `notebooks/A_date_si_teacher.ipynb` | `proc/*.npz` (imagini 96×320, LiDAR), `proc/*_teacher_gmstereo_kitti15_04487ebf.npz` |
| 2 | Colab GPU | `notebooks/B_student_antrenare_export.ipynb` | `runs/baseline`, `runs/ghidat`, `export/*.tflite`, `rezultate_test.md`, `pachet_placa.zip` |
| 3 | Placă | `ruleaza_depth.py --verifica ...` | confirmă că placa dă aceeași ieșire ca PC-ul |
| 4 | Placă | `sh benchmark.sh model.tflite` | latență CPU vs NPU, câte operații rulează pe NPU |
| 5 | Placă | `ruleaza_depth.py --npu --intrare demo_imagini` | demo: hărți de adâncime colorate + FPS + memorie |

## Fișiere

| Fișier | Faza | Rol |
|---|---|---|
| `config.py` | toate | contractul modelului, împărțirea datelor, hiperparametri |
| `kitti.py` | 2 | descărcare selectivă, calibrare, adâncime LiDAR |
| `01_pregatire_date.py` | 2 | KITTI → `proc/<secvență>.npz` |
| `02_teacher.py` | 3 | GMStereo (PyTorch) → disparitate + mască de încredere |
| `student_model.py` | 4 | MobileNetV3Small (minimalistic) + decoder U-Net |
| `losses.py` | 5–6 | pierdere fotometrică stereo, netezime, distilare |
| `dataset.py` | 5–6 | încărcare date, augmentare de culoare |
| `03_antrenare.py` | 5–6 | `--mod baseline` sau `--mod ghidat` |
| `metrics.py`, `04_evaluare.py` | 7 | metrici KITTI, tabel, figuri |
| `05_export.py`, `tflite_runner.py` | 8 | TFLite float / int8 / int8 per-tensor + referință |
| `06_imagini_demo.py` | 10 | imagini de test pentru demo |
| `placa/ruleaza_depth.py` | 9–10 | inferență pe NPU, verificare, demo, măsurători |
| `placa/benchmark.sh` | 9 | `benchmark_model` CPU vs NPU |
| `teste/test_pierderi.py` | – | teste rapide (geometrie stereo, gradienți) |

## Decizii importante

1. **Contractul modelului**: intrare 1×96×320×3 (RGB, valori 0–255), ieșire 1×48×160×1 cu valoarea `s` în [0, 1]. Disparitatea se calculează ca `q = 0,001 · 300^s`, iar adâncimea ca `Z = fx · B / (q · W)`. Scara logaritmică face ca int8 (256 de niveluri) să aibă aceeași precizie relativă (~2%) la orice distanță.
2. **MobileNetV3Small „minimalistic”** (fără squeeze-excite și hard-swish). Varianta completă, cu greutăți ImageNet, dă eroare XNNPACK în TFLite 2.19 după cuantizarea int8. Graful int8 conține doar CONV_2D, DEPTHWISE_CONV_2D, ADD, CONCATENATION, RESIZE_NEAREST_NEIGHBOR, LOGISTIC și un MUL la intrare.
3. **Fără `UpSampling2D`**: în Keras 3 devine RESHAPE+TILE. Folosim `Resizing(nearest)`, care devine o singură operație RESIZE_NEAREST_NEIGHBOR.
4. **Comparație corectă** baseline vs. ghidat: aceeași arhitectură, aceleași cadre, aceeași inițializare (seed 42), aceleași setări. Diferă doar pierderea de distilare.
5. **Teacher**: GMStereo `scale2-regrefine3` deja ajustat pe KITTI, folosit doar la inferență. Verificarea stânga–dreapta exclude pixelii nesiguri din distilare.
6. **Selecția modelului** se face pe validare (AbsRel față de LiDAR). Testul se folosește o singură dată, în `04_evaluare.py`.

## Schimbarea teacher-ului la checkpoint-ul KITTI

Teacher-ul este deja antrenat: scriptul `02_teacher.py` face numai inferență,
fără optimizer, backward sau actualizarea greutăților. Checkpoint-ul oficial
este `gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth`,
listat la Stereo Matching în [UniMatch Model Zoo](https://github.com/autonomousvision/unimatch/blob/master/MODEL_ZOO.md).

Înlocuiește fișierele din `cod/` în Drive cu versiunea nouă, fără să ștergi
`proc/`, `runs/` sau imaginile originale. Repornește sesiunea Colab, montează
Drive și rulează din nou `!python 02_teacher.py`. Dacă imaginile originale
au dispărut de pe discul temporar, scriptul le descarcă automat.
Predicțiile vechi SceneFlow rămân intacte, dar nu sunt folosite de noul cod:
cele noi au sufixul `_teacher_gmstereo_kitti15_04487ebf.npz`.
Notebook-ul B citește exclusiv noile predicții.

Trebuie verificate suprapunerile dintre datele de ajustare KITTI ale teacher-ului
și secvențele de validare/test înainte de a raporta generalizare independentă.
Faptul că teacher-ul este deja antrenat pe KITTI nu garantează că nu a văzut
scenele de evaluare. Dacă ajustarea lui a folosit adnotări de adâncime/disparitate,
varianta ghidată trebuie descrisă drept distilare de la un teacher supervizat,
nu drept sistem complet fără supervizare.

## Pe placă

```bash
# de pe PC (sau Colab -> pachet_placa.zip):
scp pachet_placa.zip root@<IP_PLACA>:~/proiect/
# pe placă:
cd ~/proiect && unzip -o pachet_placa.zip
python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --verifica ref_input.npy ref_output_ghidat_int8.npy
sh benchmark.sh student_ghidat_int8.tflite
sh benchmark.sh student_ghidat_int8_pt.tflite       # per-tensor: de obicei mai rapid pe NPU
python3 ruleaza_depth.py --model student_ghidat_int8.tflite --npu --intrare demo_imagini
python3 ruleaza_depth.py --model student_ghidat_int8.tflite --intrare demo_imagini   # CPU, pentru comparație
```

Pentru raport notați: `Inference (avg)` pe CPU și pe NPU, linia `Replacing X out of Y node(s)` (câte operații rulează pe NPU), dimensiunea modelului, memoria (VmHWM) și FPS-ul din `ruleaza_depth.py`. Primul apel pe NPU durează mult, pentru că atunci se compilează graful; de aceea se exclude din medie.

## Test rapid (fără GPU, ~15 minute pe CPU)

```bash
export ECA_DEBUG=1 ECA_ROOT=/tmp/eca ECA_RAW=/tmp/kitti_raw
python 01_pregatire_date.py && python 02_teacher.py --max_frames 4
python 03_antrenare.py --mod baseline --epochs 2 --batch 4 --steps_per_epoch 4
python 05_export.py --nume baseline
```
