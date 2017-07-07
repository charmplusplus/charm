CHARMRUN=./charmrun
-include ../../common.mk
CHARMC=../../../bin/charmc

CXX=$(CHARMC)
CXXFLAGS=$(OPTS)

BINARY=matmul

all: $(BINARY)

$(BINARY): matmul.o
	$(CHARMC) -o $@ $< $(OPTS)

matmul.o: matmul.C matmul.decl.h
matmul.decl.h: matmul.ci
	$(CHARMC) $<

test-local: $(BINARY)
	$(call run, ./matmul +p4 50 5 )

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun
