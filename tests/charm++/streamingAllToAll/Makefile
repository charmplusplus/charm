-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = ataTest.o

all: ataTest

ataTest: $(OBJS)
	$(CHARMC) -language charm++  -o ataTest -module NDMeshStreamer $(OBJS)

ataTest.decl.h: ataTest.ci
	$(CHARMC)  ataTest.ci

test: all
	$(call run, +p4 ./ataTest )

clean:
	rm -f *.decl.h *.def.h conv-host *.o ataTest charmrun

ataTest.o: ataTest.C ataTest.decl.h
	$(CHARMC) -c -O3 -optimize -production ataTest.C
