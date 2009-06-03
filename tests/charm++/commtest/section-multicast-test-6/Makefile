CHARMC=../../../../bin/charmc -module comlib  $(OPTS)

all: pgm


test: pgm
	./charmrun pgm +p1 2 2
	./charmrun pgm +p2 5 5 
	./charmrun pgm +p4 5 5
	./charmrun pgm +p8 5 5

pgm: test.o
	$(CHARMC) -language charm++ -o pgm test.o $(COMLIB)

test.decl.h: test.ci
	$(CHARMC)  test.ci

test.o: test.C test.decl.h
	$(CHARMC) -c test.C -o test.o $(OPTS)


clean:
	rm -f *.decl.h *.def.h conv-host *.o charmrun *~ lu lu-blas lu-mem lu-blas-proj.*.log lu-blas-proj.*.sum lu-blas-proj.*.sts lu-blas-proj.sts lu-blas-proj.projrc lu-blas-proj test.o test.decl.h test.def.h test pgm

