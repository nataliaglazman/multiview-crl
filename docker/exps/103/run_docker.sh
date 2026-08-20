runai submit --name multiview-synthetic-high-res-cont-n-latent-no-nan-2 \
 -i aicregistry:5000/nglazman:multiview-crl-vqvae-latest-2 \
 --node-type A100 \
 --run-as-user \
 --gpu 1 \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs --large-shm --command -- bash /nfs/home/nglazman/crl-2/multiview-crl/docker/exps/103/run_synthetic.sh
