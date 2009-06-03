CHARMC=../../../../bin/charmc -module comlib  $(OPTS)

all: pgm


test: pgm
	./charmrun pgm +p1
	./charmrun pgm +p2
	./charmrun pgm +p4
	./charmrun pgm +p8

pgm: eachtomany.o
	$(CHARMC) -language charm++ -o pgm eachtomany.o $(COMLIB)

eachtomany.decl.h: eachtomany.ci
	$(CHARMC)  eachtomany.ci

eachtomany.o: eachtomany.C eachtomany.decl.h
	$(CHARMC) -c eachtomany.C -o eachtomany.o $(OPTS)


clean:
	rm -f *.decl.h *.def.h conv-host *.o charmrun *~ lu lu-blas lu-mem lu-blas-proj.*.log lu-blas-proj.*.sum lu-blas-proj.*.sts lu-blas-proj.sts lu-blas-proj.projrc lu-blas-proj test.o test.decl.h test.def.h test pgm eachtomany.o pgm

