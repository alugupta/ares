#Run the docker container with appropriate mount points
#Edit the imagenet location to the correct mount
nvidia-docker run --rm -it --init \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  -v "$(pwd)":/ares \
  -v /data:/data \
  -v /data/ilsvrc2012:/data/imagenet \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  arestest:gpu ./run.sh
