-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = hello.o hellof.o

all:pgm

pgm: $(OBJS) 
	$(CHARMC) -o pgm $(OBJS) -language f90charm

hello.decl.h hello.def.h: hello.ci
	$(CHARMC) hello.ci -language f90charm

hello.o: hello.C hello.decl.h  hello.def.h
	$(CHARMC) -c hello.C

hellof.o: hellof.f90
	$(CHARMC) -c hellof.f90

test: pgm
	$(call run, +p2 ./pgm)

clean:
	rm -f charmrun conv-host pgm *.o *.mod *.decl.h *.def.h

