PYTHON=python3
PIP=pip3

PYTHONENV=./venv
PYTHONPATH=$(PYTHONENV)/bin/$(PYTHON)
PIPPATH=$(PYTHONENV)/bin/$(PIP)

setup:
	sudo apt install $(PYTHON)  $(PYTHON)-venv $(PYTHON)-pip

setup_pyenv: setup
	$(PYTHON) -m venv $(PYTHONENV)

setup_py:
	$(PIPPATH) install -r requirements.txt


test: setup_py
	$(PYTHONPATH) test.py