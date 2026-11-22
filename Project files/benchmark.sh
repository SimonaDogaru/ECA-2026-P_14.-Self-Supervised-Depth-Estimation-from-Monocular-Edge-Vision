#!/bin/sh
# FAZA 9 - Benchmark CPU vs NPU cu unealta oficială benchmark_model (TFLite 2.19.0).
#
# Folosire (pe placă):  sh benchmark.sh student_ghidat_int8.tflite [nr_rulari] [fps_tinta]
# Logurile complete se salvează în bench_<model>_cpu.log și bench_<model>_npu.log.
#
# Ce dovedește că modelul rulează pe NPU:
#   linia "Replacing X out of Y node(s) with delegate (...)" și, în profil,
#   "Summary by node type" cu un singur tip de nod = delegate-ul VX.
#   Orice alt tip de nod din acel tabel (CONV_2D, RESIZE_..., etc.) rulează pe CPU.

MODEL="$1"
RUNS="${2:-100}"
FPS_TINTA="${3:-30}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Folosire: sh benchmark.sh <model.tflite> [nr_rulari=100] [fps_tinta=30]"; exit 1
fi
BM=$(ls /usr/bin/tensorflow-lite-*/examples/benchmark_model 2>/dev/null | head -1)
if [ -z "$BM" ]; then echo "Nu găsesc benchmark_model"; exit 1; fi
DELEGATE=/usr/lib/libvx_delegate.so
NAME=$(basename "$MODEL" .tflite)
CPU_LOG="bench_${NAME}_cpu.log"
NPU_LOG="bench_${NAME}_npu.log"

# Ultimul tabel "Summary by node type" (cel de la rulările măsurate), fără antet.
node_summary() {
    awk '/Summary by node type/ {buf=""; f=1; next}
         f && (/^[[:space:]]*$/ || /Timings/ || /====/) {if (buf != "") last=buf; f=0}
         f {buf = buf $0 "\n"}
         END {if (f && buf != "") last=buf; printf "%s", last}' "$1"
}
# "Inference (avg)" în microsecunde
avg_us() { sed -n 's/.*Inference (avg): \([0-9.e+]*\).*/\1/p' "$1" | tail -1; }

echo "== benchmark_model: $BM"
echo "== Model: $MODEL ($(du -k "$MODEL" | cut -f1) kB), $RUNS rulări"

echo; echo "== 1/2 CPU (4 fire) =="
"$BM" --graph="$MODEL" --num_threads=4 --warmup_runs=10 --num_runs="$RUNS" \
      --enable_op_profiling=true > "$CPU_LOG" 2>&1
grep -iE "Inference timings|Memory footprint|peak memory" "$CPU_LOG"

echo; echo "== 2/2 NPU (VX delegate) =="
TEMP0=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
USE_GPU_INFERENCE=0 "$BM" --graph="$MODEL" --external_delegate_path=$DELEGATE \
      --warmup_runs=10 --num_runs="$RUNS" --enable_op_profiling=true \
      > "$NPU_LOG" 2>&1
TEMP1=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
grep -iE "replacing|delegate.*(applied|node)|Inference timings|Memory footprint|peak memory" "$NPU_LOG"

echo; echo "== Ce operații au rulat unde (NPU, rulările măsurate) =="
SUMMARY=$(node_summary "$NPU_LOG")
if [ -n "$SUMMARY" ]; then
    printf "%s" "$SUMMARY"
    CPU_OPS=$(printf "%s" "$SUMMARY" | awk 'NR>1 && $0 !~ /[Dd]elegate/ && NF {print $1}')
    if [ -z "$CPU_OPS" ]; then
        echo "=> TOATE operațiile rulează pe NPU (un singur nod: delegate-ul)."
    else
        echo "=> ATENȚIE: operații rămase pe CPU: $(echo $CPU_OPS)"
    fi
else
    echo "(nu am găsit 'Summary by node type' în $NPU_LOG)"
fi

echo; echo "== Rezumat =="
CPU_US=$(avg_us "$CPU_LOG"); NPU_US=$(avg_us "$NPU_LOG")
if [ -n "$CPU_US" ] && [ -n "$NPU_US" ]; then
    awk -v c="$CPU_US" -v n="$NPU_US" -v f="$FPS_TINTA" 'BEGIN {
        b = 1000 / f
        printf "  CPU: %.1f ms (%.1f FPS max)  |  NPU: %.1f ms (%.1f FPS max)  |  accelerare NPU: %.1fx\n",
               c/1000, 1e6/c, n/1000, 1e6/n, c/n
        printf "  la %g FPS (%.1f ms/cadru) inferența pe NPU ocupă %.0f%% din buget\n", f, b, n/10/b
        printf "  (restul bugetului rămâne pentru preprocesare/postprocesare: vezi ruleaza_depth.py)\n"
    }'
fi
if [ -n "$TEMP0" ] && [ -n "$TEMP1" ]; then
    awk -v a="$TEMP0" -v b="$TEMP1" 'BEGIN {printf "  temperatură SoC: %.1f -> %.1f °C\n", a/1000, b/1000}'
fi
DISK_FREE=$(df -k . | awk 'NR==2 {print $4}')
awk -v m="$(du -k "$MODEL" | cut -f1)" -v d="$DISK_FREE" \
    'BEGIN {printf "  stocare: model %d kB = %.4f%% din spațiul liber (%.1f GB)\n", m, m/d*100, d/1048576}'
FREE_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
[ -n "$FREE_MB" ] && echo "  RAM total placă: ${FREE_MB} MB (compară cu 'peak memory' de mai sus)"

echo
echo "Explicație: 'Inference timings in us: Init, First inference, Warmup, Inference (avg)'."
echo "  First inference pe NPU include compilarea grafului (durează mult o singură dată)."
echo "  Pentru raport folosește 'Inference (avg)' de pe NPU vs CPU."
echo "Logurile complete: $CPU_LOG, $NPU_LOG"
