-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = immediateEM.o

all: immediateEM

immediateEM: $(OBJS)
	$(CHARMC) -language charm++ -o immediateEM $(OBJS)

immediateEM.o: immediateEM.C immediateEM.decl.h
	$(CHARMC) -c immediateEM.C

immediateEM.decl.h: immediateEM.ci
	$(CHARMC) immediateEM.ci

clean:
	rm -rf *.decl.h *.def.h conv-host *.o immediateEM charmrun *~ *log *projrc *sts

test: all
	$(call run, +p4 ./immediateEM)
