CC=gcc
PYTHON=python3
PIP=pip3

PYTHONFOLDER=../python
PYTHONENV=$(PYTHONFOLDER)/venv
PYTHONPATH=$(PYTHONENV)/bin/$(PYTHON)
PIPPATH=$(PYTHONENV)/bin/$(PIP)

SRCPATH=./src
INCPATH=./include
DISTPATH=./dist
BUILDPATH=./build
OUTPATH=./out
TESTPATH=./test
DIRS=$(DISTPATH) $(BUILDPATH) $(OUTPATH)

CFLAGS=-Wall

ifeq ($(DEBUG),1)
CFLAGS+=-g
else
CFLAGS+=-O3
endif

ifdef BENCHMARK_AVG_RUNS
CFLAGS+=-DBENCHMARK_AVG_RUNS=$(BENCHMARK_AVG_RUNS)
endif

ifdef SCHEDULING
CFLAGS+=-DSCHEDULING=$(SCHEDULING)
endif

ifndef OUTFOLDER
OUTFOLDER=$(OUTPATH)
endif 

ifdef LOOP_COLLAPSE
CFLAGS+=-DLOOP_COLLAPSE=$(LOOP_COLLAPSE)
endif 


run_python_c_equiv_test:
	$(PYTHONPATH) test/model_equivalence_check.py

$(BUILDPATH)/hydra_convolve.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_convolve.c -o $(BUILDPATH)/hydra_convolve.o

$(BUILDPATH)/hydra_forward.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_forward.c -o $(BUILDPATH)/hydra_forward.o

$(BUILDPATH)/hydra_reset.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_reset.c -o $(BUILDPATH)/hydra_reset.o

$(BUILDPATH)/hydra_utils.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_utils.c -o $(BUILDPATH)/hydra_utils.o 

forward_equivalence_check: $(BUILDPATH)/hydra_convolve.o  $(BUILDPATH)/hydra_forward.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_utils.o
	$(CC) $(CFLAGS) -fopenmp $(TESTPATH)/forward_equivalence_check.c -o $(DISTPATH)/forward_equivalence_check $(BUILDPATH)/hydra_forward.o $(BUILDPATH)/hydra_convolve.o $(BUILDPATH)/hydra_utils.o -lm

clean:
	rm -rf $(BUILDPATH)
	rm -rf $(DISTPATH)

$(shell mkdir -p $(DIRS))
