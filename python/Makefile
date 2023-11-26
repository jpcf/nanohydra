PYTHON=python3
PIP=pip3

PYTHONENV=./venv
PYTHONPATH=$(PYTHONENV)/bin/$(PYTHON)
PIPPATH=$(PYTHONENV)/bin/$(PIP)
KERNPROFPATH=$(PYTHONENV)/bin/$(KERNPROF)

.PHONY: test

setup:
	sudo apt install $(PYTHON)  $(PYTHON)-venv $(PYTHON)-pip

setup_pyenv: setup
	$(PYTHON) -m venv $(PYTHONENV)

setup_py:
	$(PIPPATH) install -r requirements.txt

build_lib:
	$(PYTHONPATH) ./nanohydra/setup.py build_ext --build-lib=./nanohydra/optimized_fns

clean_lib:
	rm nanohydra/optimized_fns/*.c nanohydra/optimized_fns/*.so

test: 
	$(PYTHONPATH) test.py 10000

run_ucr_benchmark: 
	$(PYTHONPATH) run_ucr_benchmark.py all

analyze_ucr_benchmark: 
	$(PYTHONPATH) scripts/ucr_compare.py

# Used in 29th Oct for profiling the first naive numpy port
profile_hydra:
	$(PYTHONPATH)  profile_hydra.py

download_speech_commands:
	$(PYTHONPATH)  download_sc.py

test_speech_commands: 
	$(PYTHONPATH) test_sc.py 90000

# Testing
test_conv1d:
	$(PYTHONPATH) test_conv_fn_equivalence.py