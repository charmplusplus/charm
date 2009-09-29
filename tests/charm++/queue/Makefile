CHARMC := ../../../bin/charmc
CXX := $(CHARMC)

all: pgm
test: pgm
	./charmrun ./pgm

pgm.C: main.decl.h

main.decl.h: test.ci
	$(CHARMC) $<

clean:
	rm -f pgm *.o *.decl.h *.def.h
