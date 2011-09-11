CHARMC=../../../bin/charmc $(OPTS)

OBJS = PMEMimic.o

all: PMEMimic

PMEMimic: $(OBJS)
	$(CHARMC) -language charm++ -o PMEMimic $(OBJS)

PMEMimic.decl.h: PMEMimic.ci
	$(CHARMC)  PMEMimic.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o PMEMimic charmrun

PMEMimic.o: PMEMimic.C PMEMimic.decl.h
	$(CHARMC) -c PMEMimic.C

test: all
	./charmrun PMEMimic +p4 10 $(TESTOPTS)

bgtest: all
	./charmrun PMEMimic +p4 10 +x2 +y2 +z1
