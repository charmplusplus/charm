OPTS=-g
CHARMDIR=../../../..
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS) 

OBJS=cst_nl.o pgm.o 

DEST=$(CHARMDIR)/lib/libmodulerefine.a
LOBJS=refine.o tri.o femrefine.o
HEADERS=refine.h refinef.h femrefine.h femrefinef.h

all: $(DEST) pgm

pgm: $(DEST) $(OBJS)
	$(CHARMC) -L. -o pgm $(OBJS) -language fem -module netfem -module refine -memory paranoid

pgm.o: pgm.C pgm.h
	$(CHARMC) -c pgm.C

cst_nl.o: cst_nl.C pgm.h
	$(CHARMC) -c cst_nl.C

$(DEST): $(LOBJS) $(HEADERS)
	$(AR) cr $@ $(LOBJS)
	cp $(HEADERS) $(CHARMDIR)/include/

refine.o: refine.C refine.h tri.h
	$(CHARMC) -c refine.C

femrefine.o: femrefine.C femrefine.h refine.h
	$(CHARMC) -c femrefine.C

tri.o: tri.C tri.h 
	$(CHARMC) -c tri.C

tri.h: refine.decl.h

refine.decl.h: tri.ci
	$(CHARMC) tri.ci	

clean:
	rm -f pgm fpgm *~ *.o *.decl.h *.def.h conv-host charmrun $(OBJS)
