CHARMC=../../../../bin/charmc $(OPTS) 

all: pgm

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

clean:
	rm -f pgm *.o conv-host
