#!/bin/sh
set -ex

# install docker image if not present
out="$(sudo docker pull voyz/ibeam)"

case "$out" in
    *"up to date"*) ;;
    *)
    sudo docker stop voyz/ibeam
    echo 'deleting outdated image..'
    sudo docker rm -f voyz/ibeam
    sudo docker image prune -f
    exit 0
    ;;
esac

sudo docker run -v "${PWD}/container_inputs":/srv/inputs --env-file env.list \
    --name "ibkr-algotrading" -p 5000:5000 voyz/ibeam
