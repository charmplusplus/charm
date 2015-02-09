-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = jacobi1d.o

all: jacobi1d

jacobi1d: $(OBJS)
	$(CHARMC) -language charm++ -o jacobi1d $(OBJS)

proj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o jacobi1d.prj $(OBJS)

jacobi1d.o: jacobi1d.C jacobi1d.decl.h
	$(CHARMC) -c jacobi1d.C

jacobi1d.decl.h: jacobi1d.ci
	$(CHARMC)  jacobi1d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi1d jacobi1d.prj charmrun *~ *log *projrc *sts

test: all
	$(call run, +p4 ./jacobi1d 32 32 4 )

