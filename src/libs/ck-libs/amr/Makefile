# Makefile for AMR (Adaptive Mesh Refinement) library

CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

OBJS = 	amr.o fifo.o statcoll.o
HEADERS=amr.h bitvec.h fifo.h amr.decl.h amr.def.h statcoll.h

all: libamr.a headers

libamr.a: $(HEADERS) $(OBJS)
	$(CHARMC) -cp $(CDIR)/lib -o libamr.a $(OBJS)

headers: $(HEADERS)
	/bin/cp $(HEADERS) $(CDIR)/include

amr.decl.h amr.def.h : amr.ci
	$(CHARMC) amr.ci

amr.o: amr.C $(HEADERS)
	$(CHARMC) -c amr.C

fifo.o: fifo.c fifo.h
	$(CHARMC) -c fifo.c

statcoll.o: statcoll.C statcoll.h
	$(CHARMC) -c statcoll.C

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache




