OPTS=-g
CHARMC=$(HOME)/charm/net-linux/bin/charmc $(OPTS)

PGM=ring

all: ring maxReduce prime jacobi3D line

ring: blue.o ring.o
	$(CHARMC) -language converse++ -o ring blue.o ring.o

maxReduce: maxReduce.o blue.o
	$(CHARMC) -language converse++ -o maxReduce blue.o maxReduce.o

prime: prime.o blue.o
	$(CHARMC) -language converse++ -o prime blue.o prime.o

jacobi3D: jacobi3D.o blue.o
	$(CHARMC) -language converse++ -o jacobi3D blue.o jacobi3D.o

line: line.o blue.o
	$(CHARMC) -language converse++ -o line blue.o line.o

blue.o: blue.h blue.c
	$(CHARMC) -c blue.c

ring.o: ring.c blue.h
	$(CHARMC) -c ring.c

maxReduce.o: maxReduce.C blue.h
	$(CHARMC) -c maxReduce.C

prime.o: prime.C blue.h
	$(CHARMC) -c prime.C

jacobi3D.o: jacobi3D.C blue.h
	$(CHARMC) -c jacobi3D.C

line.o: line.C blue.h
	$(CHARMC) -c line.C

clean:
	rm -f core 
	rm -f TAGS *.o
	rm -f ring maxReduce prime jacobi3D line
	rm -f conv-host charmrun

