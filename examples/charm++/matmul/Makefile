-include ../../common.mk
CHARMC=../../../bin/charmc

CXX=$(CHARMC)
CXXFLAGS=$(OPTS)
OPTS  = -O3 -I${TACC_PAPI_INC} -xCORE-AVX512

BINARY=matmul

all: $(BINARY)

$(BINARY): matmul.o
	$(CHARMC) -L${TACC_PAPI_LIB} -lpapi -module CommonLBs -o $@ $^ $(OPTS)

matmul.o: matmul.C matmul.decl.h
matmul.decl.h: matmul.ci
	$(CHARMC) $<

test: $(BINARY)
	$(call run, +p4 ./matmul 50 5 )

testp: $(BINARY)
	$(call run, +p$(P) ./matmul $$(( $(P) * 10 ))  5 )

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun
