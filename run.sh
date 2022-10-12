#!/bin/bash

xhost +local:docker

docker run -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    --volume="$PWD:/root/yolov3-tf2" \
    --network=host \
    kakalin/kimage:cuda11.6-tf2.9.1-devel
