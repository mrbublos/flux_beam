export VERSION=latest
#docker build -t skrendelauth/file-saver:$VERSION -t skrendelauth/file-saver:latest .
docker buildx build --platform linux/amd64 -t skrendelauth/style-check:$VERSION -t skrendelauth/style-check:latest .
#docker push skrendelauth/style-check:$VERSION