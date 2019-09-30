#docker pull anibali/pytorch:cuda-8.0
nvidia-docker build . --rm -f Dockerfile.gpu -t arestest:gpu
