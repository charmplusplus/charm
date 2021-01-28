-include ../../common.mk
CHARMC := ../../../bin/charmc
CXX := $(CHARMC) $(OPTS)

TARGETS = queueperf msgqtest
all: $(TARGETS)
test: $(TARGETS)
	$(call run, ./queueperf  +p1)
	$(call run, ./msgqtest  +p1)

queueperf: pgm.C main.decl.h
	$(CHARMC) $(OPTS) -o $@ pgm.C

msgqtest.C: main.decl.h

main.decl.h: test.ci.stamp

test.ci.stamp: test.ci
	$(CHARMC) $<
	touch $@

clean:
	rm -f $(TARGETS) *.o *.decl.h *.def.h test.ci.stamp charmrun queueperf.exe queueperf.pdb queueperf.ilk
