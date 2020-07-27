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

testp: all
ifeq ($(CMK_NO_PARTITIONS),1)
	echo "Skipping test since build does not support partitions"
else ifeq ($(CMK_BLUEGENEQ),1)
	echo "Skipping partitions test on BGQ"
else
	@if [ $$(( $(P) % 2 )) = 0 ]; then \
	echo "P is even, running partitions test with $(P)/2 partitions" && \
	echo $(call run, ./hello +p$(P) $$(( $(P) * 4 )) 2 +partitions $$(( $(P) / 2 )) ) && \
	$(call run, ./hello +p$(P) $$(( $(P) * 4 )) 2 +partitions $$(( $(P) / 2 )) ); \
	else echo "P is odd, running partitions test with 1 partition" && \
	echo $(call run, ./hello +p$(P) $$(( $(P) * 4 )) $(P) +partitions 1) && \
	$(call run, ./hello +p$(P) $$(( $(P) * 4 )) $(P) +partitions 1); fi
endif

smptest: all
ifeq ($(CMK_NO_PARTITIONS),1)
	echo "Skipping test since build does not support partitions"
else ifeq ($(CMK_BLUEGENEQ),1)
	echo "Skipping partitions test on BGQ"
else
	$(call run, ./hello +p4 10 2 +partitions 2 ++ppn 2)
endif
