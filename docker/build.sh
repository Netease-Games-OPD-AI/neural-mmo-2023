DOCKER_BASE_IMAGE="${DOCKER_BASE_IMAGE:-rayproject/ray:1.6.0-py39-gpu}"

docker build --no-cache \
--build-arg DOCKER_BASE_IMAGE=${DOCKER_BASE_IMAGE} \
-t ${DOCKER_BASE_IMAGE}-nmmo2023 .
