-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello $(OBJS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun *.log *.sum *.sts hello.exe hello.pdb hello.ilk

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
ifeq ($(CMK_NO_PARTITIONS),1)
	echo "Skipping test since build does not support partitions"
else ifeq ($(CMK_BLUEGENEQ),1)
	echo "Skipping partitions test on BGQ"
else
	$(call run, ./hello +p4 10 2 +partitions 2)
endif

smptest: all
ifeq ($(CMK_NO_PARTITIONS),1)
	echo "Skipping test since build does not support partitions"
else ifeq ($(CMK_BLUEGENEQ),1)
	echo "Skipping partitions test on BGQ"
else
	$(call run, ./hello +p4 10 2 +partitions 2 ++ppn 2)
endif

bgtest: all
	$(call run, ./hello +p4 10 2 +x2 +y2 +z2 +cth1 +wth1 +partitions 2)
