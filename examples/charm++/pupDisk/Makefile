CHARMC=../../../bin/charmc $(OPTS)

OBJS = pupDisk.o

all: pupDisk

pupDisk: $(OBJS)
	$(CHARMC) -language charm++ -o pupDisk $(OBJS)

pupDisk.decl.h: pupDisk.ci
	$(CHARMC)  pupDisk.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o pupDisk charmrun

pupDisk.o: pupDisk.C pupDisk.h someData.h pupDisk.decl.h
	$(CHARMC) -c pupDisk.C

test: all
	./charmrun pupDisk +p4 10 20 4 $(TESTOPTS)
	./charmrun pupDisk +p4 10 20 4 r $(TESTOPTS)

bgtest: all
	./charmrun pupDisk +p4 10 +x2 +y2 +z1
