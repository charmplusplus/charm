CDIR=../../../../
CHARMC=$(CDIR)/bin/charmc $(OPTS)

all:libddt.a $(CDIR)/include/ddt.h
	
libddt.a:ddt.o
	@echo '.... building the drived datatypes libraries...'
	@$(CHARMC) -cp $(CDIR)/lib -o libddt.a ddt.o

ddt.o:ddt.C ddt.h
	@echo '.... ddt.C...'
	@$(CHARMC) -c ddt.C 

$(CDIR)/include/ddt.h: ddt.h
	@/bin/cp ddt.h $(CDIR)/include

clean:
	@/bin/rm -f *.o *.a		

