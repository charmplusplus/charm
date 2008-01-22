CHARMC=../../../../bin/charmc $(OPTS)

OBJS = jacobi3d.o

all: jacobi3d

jacobi3d: $(OBJS)
	$(CHARMC) -O3 -language charm++ -o jacobi3d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -O3 -language charm++ -tracemode projections -lz -o jacobi3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -O3 -language charm++ -tracemode summary -lz -o jacobi3d.sum $(OBJS)

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC)  jacobi3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi3d jacobi3d.prj charmrun *~

jacobi3d.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -c -O3 jacobi3d.C
