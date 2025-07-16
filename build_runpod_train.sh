export VERSION=latest
#docker build -t skrendelauth/file-saver:$VERSION -t skrendelauth/file-saver:latest .
docker buildx build --platform linux/amd64 -f src/runpod/inference/Dockerfile -t skrendelauth/train:$VERSION .
docker push skrendelauth/train:$VERSION