export VERSION=latest
docker buildx build --platform linux/amd64 -f src/runpod/style_check/Dockerfile_base -t skrendelauth/style-check-base:$VERSION -t skrendelauth/style-check-base:latest .
docker push skrendelauth/style-check-base:$VERSION