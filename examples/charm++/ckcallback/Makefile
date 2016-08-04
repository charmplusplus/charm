-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = reftest.o

all: reftest

reftest: $(OBJS)
	$(CHARMC) -language charm++ -o reftest $(OBJS)

reftest.decl.h: reftest.ci
	$(CHARMC) reftest.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o reftest charmrun

reftest.o: reftest.C reftest.decl.h
	$(CHARMC) -c reftest.C

test: all
	$(call run, +p4 ./reftest 16 4 )