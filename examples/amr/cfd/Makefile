CHARMDIR=../../..
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)
AMRLIB=$(CHARMDIR)/lib/libamr.a

DEST=cfdAMR
OBJS = 	cfdAMR.o 
MODS = -module RefineLB

all: $(DEST)

$(DEST): $(OBJS)
	$(CHARMC) -lm -language charm++ -o $(DEST) $(OBJS) $(AMRLIB) $(MODS)
	chmod 755 $(DEST)

cfdAMR.o: cfdAMR.C cfdAMR.decl.h cfdAMR.def.h cfdAMR.h cfd.h
	$(CHARMC) -c cfdAMR.C

cfdAMR.decl.h cfdAMR.def.h: cfdAMR.ci
	$(CHARMC) cfdAMR.ci

clean:
	rm -f *.o *~ *.de*.h $(DEST) conv-host charmrun
