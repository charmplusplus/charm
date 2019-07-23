-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all: simple_get

OBJS = simple_get.o

simple_get: $(OBJS)
	$(CHARMC) -language charm++ -o simple_get $(OBJS)

cifiles: simple_get.ci
	$(CHARMC)  simple_get.ci
	touch cifiles

simple_get.o: simple_get.C cifiles
	$(CHARMC) -c simple_get.C

test: all
	$(call run, +p1 ./simple_get 20 )
	$(call run, +p2 ./simple_get 20 )
	$(call run, +p2 ./simple_get 20 +noCMAForZC )

clean:
	rm -f *.decl.h *.def.h conv-host *.o simple_get charmrun cifiles
