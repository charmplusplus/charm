CHARMC=../../../bin/charmc

CXX=$(CHARMC)
CXXFLAGS=$(OPTS)

BINARY=matmul

$(BINARY): matmul.o
	$(CHARMC) -o $@ $< $(OPTS) -lblas

matmul.o: matmul.C matmul.decl.h
matmul.decl.h: matmul.ci
	$(CHARMC) $<

clean:
	rm *.o *.decl.h *.def.h $(BINARY)
