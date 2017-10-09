-include ../../../../common.mk
CHARMC=../../../../../bin/charmc $(OPTS)

all: simple_rget

OBJS = simple_rget.o

simple_rget: $(OBJS)
	$(CHARMC) -language charm++ -o simple_rget $(OBJS)

cifiles: simple_rget.ci
	$(CHARMC)  simple_rget.ci
	touch cifiles

simple_rget.o: simple_rget.C cifiles
	$(CHARMC) -c simple_rget.C

test: all
	$(call run, +p1 ./simple_rget 20 )
	$(call run, +p2 ./simple_rget 20 )

clean:
	rm -f *.decl.h *.def.h conv-host *.o simple_rget charmrun cifiles
