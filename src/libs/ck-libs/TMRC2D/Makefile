CHARMDIR=../../../../
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)

OBJS=cst_nl.o pgm.o 
OBJS1=cst_nl.o pgm-coarsen.o
OBJS2=cst_nl.o pgm-remesh.o
OBJS3=cst_nl.o pgm-test.o

LIB=libmodulerefine
DEST=$(CHARMDIR)/lib/$(LIB).a
LOBJS=ref.o edge.o element.o tri.o refine.o femrefine.o
HEADERS=refine.h refinef.h femrefine.h femrefinef.h

all: $(DEST) pgm coarse remesh test

pgm: $(DEST) $(OBJS)
	$(CHARMC) -L. -o pgm $(OBJS) -language fem -module netfem -module refine -memory paranoid

coarse: $(DEST) $(OBJS1)
	$(CHARMC) -L. -o coarse $(OBJS1) -language fem -module netfem -module refine -memory paranoid

remesh: $(DEST) $(OBJS2)
	$(CHARMC) -L. -o remesh $(OBJS2) -language fem -module netfem -module refine -memory paranoid

test: $(DEST) $(OBJS3)
	$(CHARMC) -L. -o test $(OBJS3) -language fem -module netfem -module refine -memory paranoid

pgm.o: pgm.C pgm.h
	$(CHARMC) -c pgm.C

pgm-coarsen.o: pgm-coarsen.C pgm.h
	$(CHARMC) -c pgm-coarsen.C

pgm-remesh.o: pgm-remesh.C pgm.h
	$(CHARMC) -c pgm-remesh.C

pgm-test.o: pgm-test.C pgm.h
	$(CHARMC) -c pgm-test.C

cst_nl.o: cst_nl.C pgm.h
	$(CHARMC) -c cst_nl.C

$(DEST): $(LOBJS) $(HEADERS)
	$(CHARMC) $(LOBJS) -o $@
	cp $(HEADERS) $(CHARMDIR)/include/

refine.o: refine.C refine.h tri.h
	$(CHARMC) -c refine.C

femrefine.o: femrefine.C femrefine.h refine.h
	$(CHARMC) -c femrefine.C

tri.o:	tri.C tri.h refine.def.h refine.decl.h
	$(CHARMC) -c tri.C

tri.h: refine.decl.h

refine.def.h refine.decl.h:	tri.ci
	$(CHARMC) tri.ci

element.o:	element.C element.h
	$(CHARMC) -c element.C

edge.o:	edge.C edge.h
	$(CHARMC) -c edge.C

ref.o:	ref.C ref.h refine.def.h refine.decl.h
	$(CHARMC) -c ref.C

clean:	
	rm -f *.*de*.h *.o pgm 
