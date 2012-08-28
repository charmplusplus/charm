CHARMC := ../../../bin/charmc
CXX := $(CHARMC) $(OPTS)

TARGETS = pgm msgqtest
all: $(TARGETS)
test: $(TARGETS)
	./pgm $(TESTOPTS)
	./msgqtest $(TESTOPTS)

pgm.C: main.decl.h

msgqtest.C: main.decl.h

main.decl.h: test.ci
	$(CHARMC) $<

clean:
	rm -f $(TARGETS) *.o *.decl.h *.def.h charmrun pgm.exe pgm.pdb pgm.ilk
