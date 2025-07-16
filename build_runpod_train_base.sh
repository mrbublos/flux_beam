export VERSION=latest
#docker build -t skrendelauth/file-saver:$VERSION -t skrendelauth/file-saver:latest .
docker buildx build --platform linux/amd64 -f src/runpod/inference/Dockerfile_base -t skrendelauth/train-base:$VERSION .
docker push skrendelauth/train-base:$VERSION