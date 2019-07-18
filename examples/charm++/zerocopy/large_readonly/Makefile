-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

all: readonly

OBJS = readonly.o

readonly: $(OBJS)
	$(CHARMC) -language charm++ -o readonly $(OBJS)

cifiles: readonly.ci
	$(CHARMC) readonly.ci
	touch cifiles

readonly.o: readonly.C cifiles
	$(CHARMC) -c readonly.C

test: all
	$(call run, +p1 ./readonly)
	$(call run, +p4 ./readonly)
	$(call run, +p6 ./readonly)
	$(call run, +p4 ./readonly +noCMAForZC)
	$(call run, +p6 ./readonly +noCMAForZC)

clean:
	rm -f *.decl.h *.def.h conv-host *.o readonly charmrun cifiles

