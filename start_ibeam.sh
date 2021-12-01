#!/bin/sh
set -ex

# install docker image if not present
out="$(docker pull voyz/ibeam)"

case "$out" in
    *"up to date"*) ;;
    *)
    docker stop voyz/ibeam
    echo 'deleting outdated image..'
    docker rm -f voyz/ibeam
    docker image prune -f
    exit 0
    ;;
esac

docker run -v "${PWD}/container_inputs":/srv/inputs --env-file env.list \
    -p 5000:5000 voyz/ibeam
