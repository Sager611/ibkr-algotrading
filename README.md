![flake8 and pytests](https://github.com/Sager611/ibkr-algotrading/actions/workflows/python-app.yml/badge.svg)

## Install

Run `make`.

Create a file `env.list` with the following contents:

```conf
## this file contains sensitive information!
# you can sign in to paper accounts as well
IBEAM_ACCOUNT=<your-IBKR-username>
IBEAM_PASSWORD=<your-IBKR-password>
MAX_FAILED_AUTH=1
MAX_IMMEDIATE_ATTEMPTS=1
IBEAM_INPUTS_DIR=/srv/inputs
```

Change `<your-IBKR-username>` and `<your-IBKR-password>`.

Download ibeam's docker image:

```shell
sudo docker pull voyz/ibeam
```

More info in the [ibeam github page](https://github.com/Voyz/ibeam).

## Run

[Generate the certificates first](#generating-certificates).

In one terminal start the ibeam server,

```shell
./start_ibeam.sh
```

In another terminal, execute:

```shell
. env/bin/activate
python3 main.py
```

You could instead try a small interactive program by running:

```shell
. env/bin/activate
python3 interactive.py
```

<a href='#generating-certificates'></a>

## Generating certificates

Save them under `container_inputs`.

Tutorial: [link](https://github.com/Voyz/ibeam/wiki/TLS-Certificates-and-HTTPS#generating-certificates)

Or run `generate-certificates.sh` (better option).
