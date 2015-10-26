-include ../../common.mk
SRC=../../../src
CHARMC=../../../bin/charmc $(OPTS)

OBJS = sparse.o

all: sparse

sparse: $(OBJS)
	$(CHARMC) -language charm++ -o sparse $(OBJS)

cifiles: sparse.ci
	$(CHARMC) sparse.ci
	touch cifiles

clean:
	rm -f *.decl.h *.def.h *.o charmrun cifiles sparse

sparse.o: sparse.C sparse.h cifiles
	$(CHARMC) sparse.C

test: all
	@echo "Testing bulk construction of sparse arrays..."
	$(call run, ./sparse +p4 )
