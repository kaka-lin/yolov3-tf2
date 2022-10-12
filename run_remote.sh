#!/bin/bash

xhost +local:docker

XSOCK=/tmp/.X11-unix
XAUTH=/root/.Xauthority

if [[ $1 == "gpu" ]]; then
    docker run -it \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY=$XAUTH \
        -v $XSOCK:$XSOCK \
        -v $HOME/.Xauthority:$XAUTH \
        --shm-size="20g" \
        --volume="$PWD:/root/yolov3-tf2" \
        --network=host \
        kakalin/kimage:cuda11.6-tf2.9.1-devel

else
    docker run -it \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY=$XAUTH \
        -v $XSOCK:$XSOCK \
        -v $HOME/.Xauthority:$XAUTH \
        --shm-size="20g" \
        --volume="$PWD:/root/yolov3-tf2" \
        --network=host \
        kakalin/kimage:cuda11.6-tf2.9.1-devel
fi
