CHARMC=../../../../bin/charmc -module comlib $(OPTS)

all: hello bench benchmulti

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

bench.decl.h: bench.ci
	$(CHARMC)  bench.ci

bench.o: bench.C bench.decl.h
	$(CHARMC) -c bench.C

benchmulti.o: benchmulti.C bench.decl.h
	$(CHARMC) -c benchmulti.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello bench benchmulti charmrun *~

test_bench: all
	./charmrun bench +p4 10

test: all
	./charmrun hello +p1 1
	./charmrun hello +p1 2
	./charmrun hello +p2 2
	./charmrun hello +p4 10 2
	./charmrun hello +p4 10 3
	./charmrun hello +p4 10 4
	./charmrun hello +p4 10 5
	./charmrun hello +p2 1

bgtest: all
	./charmrun hello +p1 1  +x1 +y1 +z1
	./charmrun hello +p1 2  +x1 +y1 +z1
	./charmrun hello +p2 2  +x2 +y1 +z1
	./charmrun hello +p4 10 2  +x2 +y2 +z1
	./charmrun hello +p4 10 3  +x2 +y2 +z1
	./charmrun hello +p4 10 4  +x2 +y2 +z1
	./charmrun hello +p4 10 5  +x2 +y2 +z1
	./charmrun hello +p2 1  +x2 +y1 +z1

