CHARMC=../../../bin/charmc

CXX=$(CHARMC)
CXXFLAGS=$(OPTS)

BINARY=matmul

all: $(BINARY)

$(BINARY): matmul.o
	$(CHARMC) -o $@ $< $(OPTS) -lblas

matmul.o: matmul.C matmul.decl.h
matmul.decl.h: matmul.ci
	$(CHARMC) $<

test: $(BINARY)
	./charmrun +p4 ./matmul 50 5

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun
