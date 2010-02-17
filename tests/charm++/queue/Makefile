CHARMC := ../../../bin/charmc
CXX := $(CHARMC) $(OPTS)

all: pgm
test: pgm
	./charmrun ./pgm $(TESTOPTS)

pgm.C: main.decl.h

main.decl.h: test.ci
	$(CHARMC) $<

clean:
	rm -f pgm *.o *.decl.h *.def.h charmrun pgm.exe pgm.pdb pgm.ilk
