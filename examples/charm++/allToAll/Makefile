-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = allToAll.o

all: allToAll

allToAll: $(OBJS)
	$(CHARMC) -language charm++ -o allToAll $(OBJS)

allToAll.decl.h: allToAll.ci
	$(CHARMC)  allToAll.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o allToAll charmrun

allToAll.o: allToAll.C allToAll.decl.h
	$(CHARMC) -c allToAll.C

test: all
	$(call run, ./allToAll +p4 1024 )

