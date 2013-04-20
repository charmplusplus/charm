CHARMC=../../../bin/charmc $(OPTS)

OBJS = jacobi2d.o

all: jacobi2d

jacobi2d: $(OBJS)
	$(CHARMC) -language charm++ -o jacobi2d $(OBJS)

jacobi2d.decl.h: jacobi2d.ci
	$(CHARMC)  jacobi2d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi2d charmrun

jacobi2d.o: jacobi2d.C jacobi2d.decl.h
	$(CHARMC) -c jacobi2d.C

test: all
	./charmrun ./jacobi2d +p4 10 $(TESTOPTS)

bgtest: all
	./charmrun ./jacobi2d +p4 10 +x2 +y2 +z2
