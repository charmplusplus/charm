CHARMC=../../../../bin/charmc $(OPTS)

OBJS = jacobi3d.o

all: jacobi3d

jacobi3d: $(OBJS)
	$(CHARMC) -O3 -language charm++ -o jacobi3d $(OBJS)

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC)  jacobi3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi3d charmrun *~

jacobi3d.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -c -O3 jacobi3d.C
