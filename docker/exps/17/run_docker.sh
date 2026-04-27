runai submit --name multiview-05-patches-no-proj-scaling-002-4-05-05 \
 -i aicregistry:5000/nglazman:multiview-crl-vqvae-latest \
 --node-type A100 \
 --run-as-user \
 --gpu 1 \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs --large-shm --command -- bash /nfs/home/nglazman/crl-2/multiview-crl/docker/exps/17/run_training_c.sh
