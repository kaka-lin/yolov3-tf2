#!/bin/bash

xhost +local:docker

DOCKER_REPO="kakalin/"
BRAND=kimage
VERSION=cuda11.6-tf2.9.1-devel

IMAGE_NAME=${DOCKER_REPO}${BRAND}:${VERSION}

docker_run_params=$(cat <<-END
    --rm \
    -it \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --network=host \
    -p 8888:8888 \
    -v $PWD:/root/yolov3-tf2 \
    -w /root/yolov3-tf2 \
    $IMAGE_NAME
END
)

if [[ -z "$1" ]]; then
    docker run \
        --gpus all \
        $docker_run_params
else
    docker run \
        --gpus '"device='$1'"' \
        $docker_run_params
fi
