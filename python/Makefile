PYTHON=python3
PIP=pip3

PYTHONENV=./venv
PYTHONPATH=$(PYTHONENV)/bin/$(PYTHON)
PIPPATH=$(PYTHONENV)/bin/$(PIP)
KERNPROFPATH=$(PYTHONENV)/bin/$(KERNPROF)

setup:
	sudo apt install $(PYTHON)  $(PYTHON)-venv $(PYTHON)-pip

setup_pyenv: setup
	$(PYTHON) -m venv $(PYTHONENV)

setup_py:
	$(PIPPATH) install -r requirements.txt

build_lib:
	$(PYTHONPATH) setup.py build_ext --inplace

test: setup_py
	$(PYTHONPATH) test.py 100

# Used in 29th Oct for profiling the first naive numpy port
profile_hydra:
	$(PYTHONPATH)  profile_hydra.py