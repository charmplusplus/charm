-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

OBJS = ataTest.o

TARGET = ataTest

all: $(TARGET)
test: $(foreach i,$(TARGET),test-$i)

ataTest: $(OBJS)
	$(CHARMC) -language charm++  -o ataTest -module NDMeshStreamer $(OBJS)

ataTest.decl.h: ataTest.ci
	$(CHARMC)  ataTest.ci

test-ataTest: ataTest
	$(call run, +p4 ./ataTest )

testp: all
	$(call run, +p$(P) ./ataTest )

clean:
	rm -f *.decl.h *.def.h conv-host *.o ataTest charmrun

ataTest.o: ataTest.C ataTest.decl.h
	$(CHARMC) -c ataTest.C
