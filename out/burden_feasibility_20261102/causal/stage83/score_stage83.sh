#!/bin/bash
# Stage 83's scoring on the PC (submit_stage83.sh's header has the rules), once all ten runs log DONE on the cluster:
# FETCH=1 first copies their runs and logs back into runs/ and logs/ (names new to the PC). Then bundles,
# pool_check.py detect (holds and rho1), style_audit.py, and the rules.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
D=out/burden_feasibility_20261102
S=$D/causal/stage83
PY=${PY:-$HOME/miniconda3/envs/monai_env/bin/python}
export PYTHONPATH=$D:out/regional_pilot_20261028/shim:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P=ssl_uni_jitter_nopatch_restail_t104_causal_nofield_fixedpos_abl_codebatch_predict1_proj64out16_c8_20k_a100
R=/nfs/home/nglazman/crl-2/multiview-crl/$D
PAIRS=() RUNS=()
for seed in $(seq 26 35); do
    n=${P}_s$seed
    if [ "${FETCH:-0}" = 1 ] && [ ! -f "$D/runs/$n/model.pt" ]; then
        ssh -o BatchMode=yes aicheadnode "grep -qx DONE $R/logs/$n.log" || { echo "$n has not finished"; exit 1; }
        scp -rq -o BatchMode=yes "aicheadnode:$R/runs/$n" "$D/runs/" && scp -q -o BatchMode=yes "aicheadnode:$R/logs/$n.log" "$D/logs/"
    fi
    grep -qx DONE "$D/logs/$n.log" || { echo "$n is not finished or not copied back"; exit 1; }
    if [ ! "$D/causal/$n.npz" -nt "$D/runs/$n/model.pt" ]; then
        "$PY" "$D/causal_bundle.py" "$D/runs/$n" "$D/causal/$n.npz" > "$S/bundle_s$seed.log" 2>&1
    fi
    PAIRS+=("c8_wproj_s$seed=$n")
    RUNS+=("$D/runs/$n")
done
echo "$(date +%H:%M) pool_check detect into $S/detect.txt"
"$PY" "$D/pool_check.py" detect "$S/detect.json" "${PAIRS[@]}" --subsets 3 --rule-k 3 > "$S/detect.txt" 2>&1
echo "$(date +%H:%M) style_audit into $S/style_audit.txt"
"$PY" "$D/style_audit.py" "$S/style_audit.json" "${RUNS[@]}" > "$S/style_audit.txt" 2>&1
"$PY" - "$S/detect.json" "$S/style_audit.json" <<'PYEOF' | tee "$S/summary.txt"
import json, sys
rows = json.load(open(sys.argv[1]))["rows"]
style = json.load(open(sys.argv[2]))
held = sum(r["holds"] for r in rows.values())
worst_style = max(max(v for stage in ("code", "code_boosted") for v in read[view][stage]["own"])
                  for s in style.values() for read in [s["read"]] for view in read)
for label, r in sorted(rows.items(), key=lambda kv: int(kv[0].rsplit("_s", 1)[1])):
    print(f"{label:16s} {'holds' if r['holds'] else 'fails'}  last {r['last'][0]:+.2f}/{r['last'][1]:+.2f}  "
          f"worst {r['worst'][0]:+.2f}/{r['worst'][1]:+.2f}  rho1 {r['rho1']:.2f}")
print(f"holds {held} of {len(rows)}; largest own-view style R2 from the code {worst_style:.3f}")
verdict = "works" if held >= 9 and worst_style <= 0.10 else "fails" if held <= 6 else "unresolved"
print(f"STAGE 83: {verdict}" + ("" if len(rows) == 10 else " (incomplete)"))
PYEOF
