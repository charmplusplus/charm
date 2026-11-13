-include ../../common.mk
CHARMC=../../../bin/charmc

CXX=$(CHARMC)
CXXFLAGS=$(OPTS) -std=c++20

BINARY=matmul

all: $(BINARY)

$(BINARY): matmul.o
	$(CHARMC) -o $@ $^ $(OPTS)

matmul.o: matmul.C matmul.decl.h ../../../include/ckcoro.h
matmul.decl.h: matmul.ci
	$(CHARMC) $<

# The input depends only on the element index (the same code as in
# ../matmul), so every run prints the same checksum, which ../matmul prints
# too; +ck_fakemigrate rebuilds the coroutine state at every wait.
test: $(BINARY)
	$(call run, ./matmul +p4 50 5 ) 2>&1 | tee p4.log
	grep "matmul: checksum 3915152.233307" p4.log
	$(call run, ./matmul +p1 50 5 ) 2>&1 | tee p1.log
	grep "matmul: checksum 3915152.233307" p1.log
	$(call run, ./matmul +p4 50 5 +ck_fakemigrate ) 2>&1 | tee fake.log
	grep "matmul: checksum 3915152.233307" fake.log

testp: $(BINARY)
	$(call run, +p$(P) ./matmul $$(( $(P) * 10 ))  5 )

clean:
	rm -f *.o *.decl.h *.def.h $(BINARY) charmrun p4.log p1.log fake.log
