-include ../../common.mk
CHARMC := ../../../bin/charmc
CXX := $(CHARMC) $(OPTS)

TARGETS = pgm msgqtest
all: $(TARGETS)
test: $(TARGETS)
	$(call run, ./pgm  +p1)
	$(call run, ./msgqtest  +p1)

pgm.C: main.decl.h

msgqtest.C: main.decl.h

main.decl.h: test.ci.stamp

test.ci.stamp: test.ci
	$(CHARMC) $<
	touch $@

clean:
	rm -f $(TARGETS) *.o *.decl.h *.def.h test.ci.stamp charmrun pgm.exe pgm.pdb pgm.ilk
