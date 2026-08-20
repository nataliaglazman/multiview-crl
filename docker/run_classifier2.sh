runai submit --name classifier \
 -i aicregistry:5000/nglazman:multiview-crl-vqvae-latest-2 \
 -e PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
 --node-type A100 \
 --run-as-user \
 --gpu 1 \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs --large-shm --command -- python /nfs/home/nglazman/crl-2/multiview-crl/eval/disease_classifier.py --run-dir /nfs/home/nglazman/results/ADNI_stripped_masks/ablation-baseline-levels-shared-mask-style-dropout-moco-06-06-06 --features content style all --feature-levels 0 --classifier-epochs 30 --classifier-batch-size 8
