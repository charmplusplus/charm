CHARMDIR=$(HOME)/charm
CFLAGS=-I../
CHARMC=$(CHARMDIR)/bin/charmc $(CFLAGS) $(OPTS)

OBJS=pgm.o

all: $(DEST) pgm 

pgm: $(DEST) $(OBJS)
	$(CHARMC) -L. -o pgm $(OBJS) -language fem -memory paranoid

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

clean:	
	rm -f *.*de*.h *.o pgm 
