-include ../../../common.mk
CHARMC = ../../../../bin/charmc
CXX = $(CHARMC)
CXXFLAGS = $(OPTS)

BIN = pgm

ifndef $(PROCS)
	PROCS=2
endif

all: $(BIN)

clean:
	-rm -rf *.o *~ core *.decl.h *.def.h *.txt charmrun $(BIN) $(BIN).exe $(BIN).pdb $(BIN).ilk

pgm.decl.h: pgm.ci
	$(CHARMC) $(OPTS) $<
pgm.o: pgm.decl.h

pgm: pgm.o
	$(CHARMC) $(OPTS) -o $@ $^

test: $(BIN)
	$(call run, +p$(PROCS) ./$< )

.PHONY: all clean test
