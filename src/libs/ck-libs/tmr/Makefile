OPTS=-g -Wall
CHARMDIR=../../../..
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS) 

OBJS=libmodulerefine.a cst_nl.o pgm.o 

LOBJS=refine.o tri.o

all: pgm

pgm: $(OBJS)
	$(CHARMC) -L. -o pgm $(OBJS) -language fem -module netfem -module refine

pgm.o: pgm.C pgm.h
	$(CHARMC) -c pgm.C

cst_nl.o: cst_nl.C pgm.h
	$(CHARMC) -c cst_nl.C

libmodulerefine.a: $(LOBJS)
	$(AR) cr $@ $(LOBJS)

refine.o: refine.C refine.h tri.h
	$(CHARMC) -c refine.C

tri.o: tri.C tri.h 
	$(CHARMC) -c tri.C

tri.h: refine.decl.h

refine.decl.h: tri.ci
	$(CHARMC) tri.ci	

clean:
	rm -f pgm fpgm *~ *.o *.decl.h *.def.h conv-host charmrun $(OBJS)
