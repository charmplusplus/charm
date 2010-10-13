CHARMC=../../../../bin/charmc $(OPTS)
FNAME=Jacobi2D

all: pgm

pgm:  $(FNAME).o
	$(CHARMC) $(FNAME).o -o pgm -language charm++ -module CommonLBs

$(FNAME).o : $(FNAME).C $(FNAME).h $(FNAME).ci
	$(CHARMC) -c $(FNAME).ci $(FNAME).C

clean:
	rm -f *.o conv-host pgm charmrun *.def.h *.decl.h

test:  pgm
	./charmrun +p4 ./pgm +LBPeriod 3 +balancer RotateLB +LBDebug $(TESTOPTS)

bgtest:  pgm
	./charmrun +p4 ./pgm +LBPeriod 3 +balancer RotateLB +LBDebug +x2 +y4 +z1 +cth1 +wth1
