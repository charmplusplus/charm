CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc -fpic $(OPTS)
DEST=$(CDIR)/lib/libGOTManager.a

all: $(DEST) $(CDIR)/include/GOTManager.h

$(DEST): GOTManager.o
	$(CHARMC) -o $(DEST) GOTManager.o

GOTManager.o:GOTManager.C GOTManager.h
	$(CHARMC) -c GOTManager.C 

$(CDIR)/include/GOTManager.h: GOTManager.h
	/bin/cp GOTManager.h $(CDIR)/include

clean:
	/bin/rm -f *.o *.a