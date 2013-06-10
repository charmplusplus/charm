-include ../../../common.mk
CHARMC=../../../../bin/charmc -O3 $(OPTS)

OBJS = jacobi2d.o

all: jacobi2d

jacobi2d: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -lrt -g -o jacobi2d $(OBJS)

jacobi2d.decl.h: jacobi2d.ci
	$(CHARMC)  jacobi2d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi2d charmrun *~

jacobi2d.o: jacobi2d.C jacobi2d.decl.h
	$(CHARMC) -c jacobi2d.C

test: all
	$(call run, ./jacobi2d +p4 200 20 +balancer GreedyLB +LBDebug 3 ++server ++server-port 1234)
