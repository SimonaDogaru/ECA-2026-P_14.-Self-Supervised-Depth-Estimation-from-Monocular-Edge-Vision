#!/bin/sh
# FAZA 9 - Benchmark CPU vs NPU cu unealta oficială benchmark_model (TFLite 2.19.0).
#
# Folosire (pe placă):  sh benchmark.sh student_ghidat_int8.tflite
# Logurile complete se salvează în bench_<model>_cpu.log și bench_<model>_npu.log.
#
# Ce dovedește că modelul rulează pe NPU:
#   linia "Replacing X out of Y node(s) with delegate (Vx Delegate)".
#   Dacă X < Y, restul operațiilor rulează pe CPU -> vezi profilul pe operații
#   (--enable_op_profiling) ca să afli care sunt.

MODEL="$1"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Folosire: sh benchmark.sh <model.tflite>"; exit 1
fi
BM=$(ls /usr/bin/tensorflow-lite-*/examples/benchmark_model 2>/dev/null | head -1)
if [ -z "$BM" ]; then echo "Nu găsesc benchmark_model"; exit 1; fi
DELEGATE=/usr/lib/libvx_delegate.so
NAME=$(basename "$MODEL" .tflite)
RUNS=100

echo "== benchmark_model: $BM"
echo "== Model: $MODEL ($(du -k "$MODEL" | cut -f1) kB)"

echo; echo "== 1/2 CPU (4 fire) =="
"$BM" --graph="$MODEL" --num_threads=4 --warmup_runs=10 --num_runs=$RUNS \
      --enable_op_profiling=true > "bench_${NAME}_cpu.log" 2>&1
grep -E "Inference timings|Memory footprint|Peak memory" "bench_${NAME}_cpu.log"

echo; echo "== 2/2 NPU (VX delegate) =="
USE_GPU_INFERENCE=0 "$BM" --graph="$MODEL" --external_delegate_path=$DELEGATE \
      --warmup_runs=10 --num_runs=$RUNS --enable_op_profiling=true \
      > "bench_${NAME}_npu.log" 2>&1
grep -E "Replacing|Inference timings|Memory footprint|Peak memory|Warmup" "bench_${NAME}_npu.log"

echo
echo "Explicație: 'Inference timings in us: Init, First inference, Warmup, Inference (avg)'."
echo "  First inference pe NPU include compilarea grafului (durează mult o singură dată)."
echo "  Pentru raport folosește 'Inference (avg)' de pe NPU vs CPU."
echo "Operațiile care NU au mers pe NPU apar în bench_${NAME}_npu.log, în profil."
