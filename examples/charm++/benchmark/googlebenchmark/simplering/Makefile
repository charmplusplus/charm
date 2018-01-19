# modify the path of CHARMC
CHARMC=../../../../../bin/charmc $(OPTS)

# modify the path of Google Benchmark Library
LIBS=-L/usr/local/lib/

all: simplering

simplering: simplering.o
	$(CHARMC) simplering.o  -o simplering -language charm++ ${LIBS} -lpthread -lbenchmark

simplering.o : simplering.C simplering.def.h
	$(CHARMC) -c simplering.C

simplering.decl.h simplering.def.h: simplering.ci
	$(CHARMC)  simplering.ci

test:
	make; ./charmrun ++local +p4 ./simplering

clean:
	rm -f simplering simplering_prof *.o conv-host *.decl.h *.def.h *~ charmrun *.log *.sts core charmrun.exe simplering.exe simplering.pdb simplering.ilk
