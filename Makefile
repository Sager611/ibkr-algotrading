VENV_NAME=env

install:
	python3 -m venv ${VENV_NAME}
	{ . env/bin/activate && python3 -m pip install --upgrade --upgrade-strategy eager -r requirements.txt; }
