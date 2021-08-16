VENV_NAME=env

install:
	python3 -m venv ${VENV_NAME}
	{ source env/bin/activate && python3 -m pip install -r requirements.txt; }
