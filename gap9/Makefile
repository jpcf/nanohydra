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

ifndef OUTFOLDER
OUTFOLDER=$(OUTPATH)
endif 


# Code performance enhancing options
ifdef VECTORIZE
CFLAGS+=-DVECTORIZE=$(VECTORIZE)
endif

ifdef PARALLELIZE
CFLAGS+=-DPARALLELIZE=$(PARALLELIZE)
endif

ifdef QUANT_8BIT
CFLAGS+=-DQUANT_8BIT=$(QUANT_8BIT)
endif

run_forward_equiv_test: forward_equivalence_check
	$(PYTHONPATH) test/forward_equivalence_check.py

run_classf_equiv_test: classf_equivalence_check
	$(PYTHONPATH) test/classf_equivalence_check.py

run_scaler_equiv_test: scaler_equivalence_check
	$(PYTHONPATH) test/scaler_equivalence_check.py

run_model_equiv_test: 
	$(PYTHONPATH) test/model_equivalence_check.py

$(BUILDPATH)/hydra_classifier.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_classifier.c -o $(BUILDPATH)/hydra_classifier.o

$(BUILDPATH)/hydra_convolve.o: 
	$(CC) $(CFLAGS) -fopenmp -c $(SRCPATH)/hydra_convolve.c -o $(BUILDPATH)/hydra_convolve.o

$(BUILDPATH)/hydra_forward.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_forward.c -o $(BUILDPATH)/hydra_forward.o

$(BUILDPATH)/hydra_sparse_scale.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_sparse_scale.c -o $(BUILDPATH)/hydra_sparse_scale.o

$(BUILDPATH)/hydra_init.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_init.c -o $(BUILDPATH)/hydra_init.o

$(BUILDPATH)/hydra_reset.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_reset.c -o $(BUILDPATH)/hydra_reset.o

$(BUILDPATH)/hydra_utils.o: 
	$(CC) $(CFLAGS) -c $(SRCPATH)/hydra_utils.c -o $(BUILDPATH)/hydra_utils.o 

forward_equivalence_check: $(BUILDPATH)/hydra_convolve.o  $(BUILDPATH)/hydra_forward.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_utils.o $(BUILDPATH)/hydra_init.o
	$(CC) $(CFLAGS) -fopenmp $(TESTPATH)/forward_equivalence_check.c -o $(DISTPATH)/forward_equivalence_check $(BUILDPATH)/hydra_forward.o $(BUILDPATH)/hydra_convolve.o $(BUILDPATH)/hydra_utils.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_init.o -lm

classf_equivalence_check: $(BUILDPATH)/hydra_classifier.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_init.o $(BUILDPATH)/hydra_utils.o
	$(CC) $(CFLAGS) -fopenmp $(TESTPATH)/classf_equivalence_check.c -o $(DISTPATH)/classf_equivalence_check $(BUILDPATH)/hydra_classifier.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_init.o $(BUILDPATH)/hydra_utils.o -lm

scaler_equivalence_check: $(BUILDPATH)/hydra_sparse_scale.o $(BUILDPATH)/hydra_init.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_utils.o
	$(CC) $(CFLAGS) -fopenmp $(TESTPATH)/scaler_equivalence_check.c -o $(DISTPATH)/scaler_equivalence_check $(BUILDPATH)/hydra_init.o $(BUILDPATH)/hydra_sparse_scale.o $(BUILDPATH)/hydra_utils.o  $(BUILDPATH)/hydra_reset.o -lm

model_equivalence_check: $(BUILDPATH)/hydra_convolve.o  $(BUILDPATH)/hydra_forward.o $(BUILDPATH)/hydra_reset.o $(BUILDPATH)/hydra_utils.o $(BUILDPATH)/hydra_init.o $(BUILDPATH)/hydra_sparse_scale.o $(BUILDPATH)/hydra_classifier.o
	$(CC) $(CFLAGS) -fopenmp $(TESTPATH)/model_equivalence_check.c -o $(DISTPATH)/model_equivalence_check $(BUILDPATH)/hydra_*.o -lm -lc
clean:
	rm -rf $(BUILDPATH)
	rm -rf $(DISTPATH)

$(shell mkdir -p $(DIRS))
