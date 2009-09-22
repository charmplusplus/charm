CHARMC := ../../../bin/charmc
CXX := $(CHARMC)

all: test

test.C: test.decl.h

test.decl.h: test.ci
	$(CHARMC) $<