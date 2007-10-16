CHARMC=../../../../bin/charmc -module comlib -module RefineLB $(OPTS)

all: hello bench benchmulti benchsectionmulti

hello: hello.o
	$(CHARMC) -language charm++ -o hello hello.o

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

bench: bench.o
	$(CHARMC) -language charm++ -o bench bench.o 

benchmulti: benchmulti.o
	$(CHARMC) -language charm++ -o benchmulti benchmulti.o 

benchsectionmulti: benchsectionmulti.o
	$(CHARMC) -language charm++ -o benchsectionmulti benchsectionmulti.o 


bench.decl.h: bench.ci
	$(CHARMC)  bench.ci

bench.o: bench.C bench.decl.h
	$(CHARMC) -c bench.C

benchmulti.o: benchmulti.C bench.decl.h
	$(CHARMC) -c benchmulti.C

benchsectionmulti.o: benchsectionmulti.C bench.decl.h
	$(CHARMC) -c benchsectionmulti.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello bench benchmulti benchsectionmulti charmrun *~

test_bench: all
	./charmrun bench +p4 10

test: all
	./charmrun hello +p1 1 $(TESTOPTS)
	./charmrun hello +p1 2 $(TESTOPTS)
	./charmrun hello +p2 2 $(TESTOPTS)
	./charmrun hello +p4 10 2 $(TESTOPTS)
	./charmrun hello +p4 10 3 $(TESTOPTS)
	./charmrun hello +p4 10 4 $(TESTOPTS)
	./charmrun hello +p4 10 5 $(TESTOPTS)
	./charmrun hello +p2 1 $(TESTOPTS)
	./charmrun bench +p2 100 1 +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 2 +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 4 +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p4 100 2 +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p4 100 4 +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 1 +strategy USE_GRID +balancer RefineLB  $(TESTOPTS)
	./charmrun bench +p2 100 2 +strategy USE_GRID +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 4 +strategy USE_GRID +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 1 +strategy USE_DIRECT +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 2 +strategy USE_DIRECT +balancer RefineLB $(TESTOPTS)
	./charmrun bench +p2 100 4 +strategy USE_DIRECT +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 1 +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 2 +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 4 +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p4 100 2 +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p4 100 4 +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 1 +strategy USE_GRID +balancer RefineLB  $(TESTOPTS)
	./charmrun benchmulti +p2 100 2 +strategy USE_GRID +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 4 +strategy USE_GRID +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 1 +strategy USE_DIRECT +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 2 +strategy USE_DIRECT +balancer RefineLB $(TESTOPTS)
	./charmrun benchmulti +p2 100 4 +strategy USE_DIRECT +balancer RefineLB $(TESTOPTS)
	./charmrun benchsectionmulti +p2 100 1 +balancer RefineLB $(TESTOPTS)
	./charmrun benchsectionmulti +p2 100 2 +balancer RefineLB $(TESTOPTS)
	./charmrun benchsectionmulti +p2 100 4 +balancer RefineLB $(TESTOPTS)
	./charmrun benchsectionmulti +p4 100 4 +balancer RefineLB $(TESTOPTS)

bgtest: all
	./charmrun hello +p1 1  +x1 +y1 +z1
	./charmrun hello +p1 2  +x1 +y1 +z1
	./charmrun hello +p2 2  +x2 +y1 +z1
	./charmrun hello +p4 10 2  +x2 +y2 +z1
	./charmrun hello +p4 10 3  +x2 +y2 +z1
	./charmrun hello +p4 10 4  +x2 +y2 +z1
	./charmrun hello +p4 10 5  +x2 +y2 +z1
	./charmrun hello +p2 1  +x2 +y1 +z1

