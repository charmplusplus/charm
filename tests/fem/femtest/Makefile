CHARMC=../../../../bin/charmc $(OPTS) 

all: pgm fpgm

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

fpgm: fpgm.o
	$(CHARMC) -o fpgm fpgm.o -language femf

fpgm.o: fpgm.f90
	$(CHARMC) -c fpgm.f90

clean:
	rm -f pgm fpgm *.o conv-host
