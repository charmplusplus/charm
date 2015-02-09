-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = manyToMany.o

all: manyToMany

manyToMany: $(OBJS)
	$(CHARMC) -language charm++ -o manyToMany $(OBJS)

manyToMany.decl.h: manyToMany.ci
	$(CHARMC)  manyToMany.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o manyToMany charmrun

manyToMany.o: manyToMany.C manyToMany.decl.h
	$(CHARMC) -c manyToMany.C

test: all
	$(call run, ./manyToMany +p4 1024 )

