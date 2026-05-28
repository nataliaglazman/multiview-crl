runai submit --name classifier-binary-8 \
 -i aicregistry:5000/nglazman:multiview-crl-vqvae-latest-2 \
 -e PYTHONPATH=/nfs/home/nglazman/crl-2/multiview-crl \
 --run-as-user \
 --gpu 1 \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs \
 --large-shm \
 --command -- python /nfs/home/nglazman/crl-2/multiview-crl/eval/run_dci_synthetic.py \
 --run-dir /nfs/home/nglazman/results/synthetic/synthetic-create-causal-random \
 --checkpoint /nfs/home/nglazman/results/synthetic/synthetic-create-causal-random/vqvae_best.pt --pooling
4,4,4 --levels 0 --num-samples 400
