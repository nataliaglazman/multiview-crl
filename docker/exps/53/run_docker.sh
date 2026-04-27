runai submit --name multiview-ablation-levels \
 -i aicregistry:5000/nglazman:multiview-crl-vqvae-latest \
 --node-type A100 \
 --run-as-user \
 --gpu 1 \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs --large-shm --command -- bash /nfs/home/nglazman/crl-2/multiview-crl/scripts/ablation_baseline_levels.sh
