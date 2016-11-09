# This file in metis was modified by Kavitha Chandrasekar at UIUC
# to skip ./configure step and directly make the metis library

CDIR=../../../../..
LIBDIR=$(CDIR)/lib
CC=$(CDIR)/bin/charmc -I. -I../GKlib

OBJS = b64.o blas.o csr.o error.o \
	evaluate.o fkvkselect.o fs.o getopt.o \
	gkregex.o graph.o htable.o io.o itemsets.o \
	mcore.o memory.o omp.o pdb.o pqueue.o \
	random.o rw.o seq.o sort.o string.o \
	timers.o tokenizer.o util.o

.o:
	$(CC) -c $*.c

all:  $(OBJS)

clean:
	rm -f *.o
