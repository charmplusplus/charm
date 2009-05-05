OPTS	= -O3 -DUSE_TOPOMAP=1
CHARMC	= ../../../../bin/charmc $(OPTS)

OBJS = jacobi2d.o

all: jacobi2d

jacobi2d: $(OBJS)
	$(CHARMC) -language charm++ -o jacobi2d $(OBJS)

proj: $(OBJS)
	$(CHARMC) -language charm++ -tracemode projections -o jacobi2d.prj $(OBJS)

jacobi2d.decl.h: jacobi2d.ci
	$(CHARMC)  jacobi2d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi2d jacobi2d.prj charmrun *~ *log *projrc *sts

jacobi2d.o: jacobi2d.C jacobi2d.decl.h
	$(CHARMC) -c jacobi2d.C
