runai submit --name multiview-crl-cl-10 --interactive --attach\
 -i aicregistry:5000/nglazman:multiview-crl-vqvae-latest \
 --run-as-user \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs --large-shm 