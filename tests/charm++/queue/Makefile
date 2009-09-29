CHARMC := ../../../bin/charmc
CXX := $(CHARMC)

all: pgm

pgm.C: test.decl.h

test.decl.h: test.ci
	$(CHARMC) $<
