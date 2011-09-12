CHARMC=../../../../bin/charmc $(OPTS)

OBJS = PMEMimic.o

all: pgm pgm.prj

pgm: $(OBJS)
	$(CHARMC) -language charm++ -o pgm $(OBJS) -module CkMulticast

pgm.prj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o pgm.prj $(OBJS) -module CkMulticast

PMEMimic.decl.h: PMEMimic.ci
	$(CHARMC)  PMEMimic.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o PMEMimic charmrun

PMEMimic.o: PMEMimic.C PMEMimic.decl.h
	$(CHARMC) -c PMEMimic.C

test: all
	./charmrun +p12 pgm 3 2 2 $(TESTOPTS)

bgtest: all
	./charmrun pgm +p4 10 +x2 +y2 +z1
