CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
DEST=$(CDIR)/lib/libddt.a

all: $(DEST) $(CDIR)/include/ddt.h
	
$(DEST): ddt.o
	$(CHARMC) -o $(DEST) ddt.o

ddt.o:ddt.C ddt.h
	$(CHARMC) -c ddt.C 

$(CDIR)/include/ddt.h: ddt.h
	/bin/cp ddt.h $(CDIR)/include

clean:
	/bin/rm -f *.o *.a		

