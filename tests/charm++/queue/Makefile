-include ../../common.mk
CHARMC := ../../../bin/charmc
CXX := $(CHARMC) $(OPTS)

TARGETS = queue msgqtest
all: $(TARGETS)
test: $(TARGETS)
	$(call run, ./queue  +p1)
	$(call run, ./msgqtest  +p1)

smptest: $(TARGETS)

queue: pgm.C main.decl.h
	$(CHARMC) $(OPTS) -c pgm.C
	$(CHARMC) $(OPTS) -o $@ pgm.o

msgqtest.C: main.decl.h

main.decl.h: test.ci.stamp

test.ci.stamp: test.ci
	$(CHARMC) $<
	touch $@

clean:
	rm -f $(TARGETS) *.o *.decl.h *.def.h test.ci.stamp charmrun queue.exe queue.pdb queue.ilk
