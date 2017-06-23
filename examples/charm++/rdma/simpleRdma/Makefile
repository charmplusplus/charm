-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

all: rdma

rdma:  rdma.o
	$(CHARMC) rdma.o -o rdma -language charm++

cifiles: rdma.ci
	$(CHARMC) -c rdma.ci
	touch cifiles

rdma.o : rdma.C cifiles
	$(CHARMC) -c rdma.C

test: all
	$(call run, +p4 ./rdma 10)

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun rdma
