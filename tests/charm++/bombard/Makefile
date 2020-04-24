-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

PROC=4
ARRSIZE=1000
COUNT=5

all: bombard

OBJS = bombard.o

bombard: $(OBJS)
	$(CHARMC) -language charm++ -o bombard $(OBJS)

cifiles: bombard.ci
	$(CHARMC) bombard.ci
	touch cifiles

bombard.o: bombard.C cifiles
	$(CHARMC) -c bombard.C

clean:
	rm -f *.decl.h *.def.h *.o bombard charmrun cifiles

test: all
	$(call run, +p$(PROC) ./bombard 0 $(ARRSIZE) $(COUNT))
	$(call run, +p$(PROC) ./bombard 1 $(ARRSIZE) $(COUNT))
	$(call run, +p$(PROC) ./bombard 2 $(ARRSIZE) $(COUNT))

smptest: all
	$(call run, +p$(PROC) ./bombard 0 $(ARRSIZE) $(COUNT) ++ppn 1)
	$(call run, +p$(PROC) ./bombard 1 $(ARRSIZE) $(COUNT) ++ppn 1)
	$(call run, +p$(PROC) ./bombard 2 $(ARRSIZE) $(COUNT) ++ppn 1)
	$(call run, +p$(PROC) ./bombard 0 $(ARRSIZE) $(COUNT) ++ppn 2)
	$(call run, +p$(PROC) ./bombard 1 $(ARRSIZE) $(COUNT) ++ppn 2)
	$(call run, +p$(PROC) ./bombard 2 $(ARRSIZE) $(COUNT) ++ppn 2)
