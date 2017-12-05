-include ../../../../../common.mk
CHARMC=../../../../../../bin/charmc $(OPTS)

all: simple_rput

OBJS = simple_rput.o

simple_rput: $(OBJS)
	$(CHARMC) -language charm++ -o simple_rput $(OBJS)

cifiles: simple_rput.ci
	$(CHARMC)  simple_rput.ci
	touch cifiles

simple_rput.o: simple_rput.C cifiles
	$(CHARMC) -c simple_rput.C

test: all
	$(call run, +p1 ./simple_rput 20 )
	$(call run, +p2 ./simple_rput 20 )

clean:
	rm -f *.decl.h *.def.h conv-host *.o simple_rput charmrun cifiles
