#!/usr/bin/env bash
# run_pc.sh stages (default: 68) inside the Run:ai training image, as runai_submit.sh submits them. The image has
# neither git nor SimpleITK, so before handing over to run_pc.sh this
#   - gives it a stand-in git that answers only `rev-parse --show-toplevel` (run_pc.sh's one git call);
#   - for stage 68, installs SimpleITK once into PYTHONUSERBASE on /nfs (default /nfs/home/nglazman/.pyuser/py312,
#     which python reads whatever PYTHONPATH run_pc.sh sets), from a wheel in WHEELS when one is there, else PyPI;
#   - fills stage 68's ADNI paths with experiments/cluster/runai.yaml's unless they are set, and checks the split
#     against the October conv_mlp run's split.json when that file exists.
# Everything is also appended to out/burden_feasibility_20261102/adni/runai.log, readable from the submit host.
# DRY=1 prints what would run and installs nothing. RUN_PC replaces run_pc.sh (for tests).
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd -P)
D=out/burden_feasibility_20261102
LOG=$D/adni/runai.log
STAGES=("${@:-68}")
export PY=${PY:-python}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

main() {
    echo "$(date '+%F %T') Run:ai job on $(hostname): run_pc.sh ${STAGES[*]} in $REPO"
    if [ -f .git/HEAD ]; then
        local head
        head=$(cat .git/HEAD)
        echo "checkout: $head $(cat ".git/${head#ref: }" 2> /dev/null || true)"
    fi
    if ! command -v git > /dev/null 2>&1; then
        local shim
        shim=$(mktemp -d)
        printf '#!/bin/sh\n[ "$1 $2" = "rev-parse --show-toplevel" ] && { echo %s; exit 0; }\nexit 1\n' "$REPO" \
            > "$shim/git"
        chmod +x "$shim/git"
        export PATH=$shim:$PATH
        echo "no git here: run_pc.sh gets a stand-in that answers only rev-parse --show-toplevel"
    fi
    case " ${STAGES[*]} " in
        *" 68 "*)
            export PYTHONUSERBASE=${PYTHONUSERBASE:-/nfs/home/nglazman/.pyuser/py312}
            export ADNI_DATA=${ADNI_DATA:-/nfs/home/nglazman/data/ADNI_stripped_masks}
            export ADNI_LABELS=${ADNI_LABELS:-/nfs/home/nglazman/nmpevqvae/labels_cleaned_3class_demog.csv}
            local split=$REPO/results/encoder_adni/runs/encoder_adni_conv_mlp_s42_demog/split.json
            if [ -z "${ADNI_SPLIT:-}" ] && [ -f "$split" ]; then
                export ADNI_SPLIT=$split
            fi
            echo "ADNI_DATA $ADNI_DATA"
            echo "ADNI_LABELS $ADNI_LABELS"
            echo "ADNI_SPLIT ${ADNI_SPLIT:-none: the split is computed but not checked}"
            if "$PY" -c 'import SimpleITK' > /dev/null 2>&1; then
                echo "SimpleITK found"
            elif [ "${DRY:-0}" = 1 ]; then
                echo "SimpleITK missing; a real run would install it into $PYTHONUSERBASE"
            else
                local wheels=${WHEELS:-/nfs/home/nglazman/wheels}
                echo "installing SimpleITK into $PYTHONUSERBASE"
                if compgen -G "$wheels/[Ss]imple[Ii][Tt][Kk]-*.whl" > /dev/null; then
                    "$PY" -m pip install --user --no-cache-dir --no-index --find-links "$wheels" SimpleITK || true
                else
                    "$PY" -m pip install --user --no-cache-dir SimpleITK || true
                fi
                if ! "$PY" -c 'import SimpleITK'; then
                    echo "SimpleITK could not be installed. Where pip can reach PyPI, download the wheel to $wheels:"
                    echo "  python3 -m pip download SimpleITK --only-binary=:all: --python-version 3.12" \
                        "--platform manylinux2014_x86_64 --no-deps -d $wheels"
                    echo "then submit again."
                    return 1
                fi
            fi
            ;;
    esac
    bash "${RUN_PC:-$D/run_pc.sh}" "${STAGES[@]}"
}

mkdir -p "$(dirname "$LOG")"
set +e
main 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "$(date '+%F %T') run_pc.sh ${STAGES[*]} exited with status $status" | tee -a "$LOG"
exit "$status"
