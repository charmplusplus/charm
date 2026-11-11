-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = sparse.o

all: sparse

sparse: $(OBJS)
	$(CHARMC) -language charm++ -o sparse $(OBJS)

sparse.decl.h: sparse.ci
	$(CHARMC) sparse.ci

clean:
	rm -f *.decl.h *.def.h *.o charmrun sparse

sparse.o: sparse.C sparse.h sparse.decl.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 -c sparse.C

# The coroutine version of ../sparse: run_tests is a [coroutine] entry that
# waits on the [inbox, reductiontarget] testDone. +ck_fakemigrate rebuilds
# the coroutine state at every wait.
test: all
	@echo "Testing bulk construction of sparse arrays (coroutine)..."
	$(call run, ./sparse +p4 )
	$(call run, ./sparse +p1 )
	$(call run, ./sparse +p4 +ck_fakemigrate )

smptest: all
	@echo "Testing bulk construction of sparse arrays (coroutine)..."
	$(call run, ./sparse +p2 ++ppn 2 )
	$(call run, ./sparse +p4 ++ppn 2 )

.PHONY: all clean test smptest
