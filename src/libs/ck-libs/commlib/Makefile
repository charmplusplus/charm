CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
COMMLIBDIR=../../conv-libs/commlib

LIB = libcomm.a
OBJS = ComlibManager.o $(COMMLIBDIR)/commlib.o $(COMMLIBDIR)/gridrouter.o \
	$(COMMLIBDIR)/de.o $(COMMLIBDIR)/treerouter.o $(COMMLIBDIR)/petable.o \
	$(COMMLIBDIR)/overlapper.o $(COMMLIBDIR)/rsend.o $(COMMLIBDIR)/bcast.o \
	$(COMMLIBDIR)/charm_bind.o 

all: cifiles conversecom $(LIB) 
	rm -f $(CDIR)/include/ComlibManager.h
	cp  ComlibManager.h $(CDIR)/include
	rm -f $(CDIR)/include/ComlibModule.decl.h
	cp  ComlibModule.decl.h $(CDIR)/include
	rm -f $(CDIR)/include/ComlibModule.def.h
	cp  ComlibModule.def.h $(CDIR)/include
	rm -f $(CDIR)/tmp/ComlibManager.ci
	cp  ComlibManager.ci $(CDIR)/tmp

conversecom :
	(cd ../../conv-libs/commlib; make)

libcomm.a: $(OBJS)
	$(CHARMC) -cp ../../../../../lib -o libcomm.a $(OBJS)

ComlibManager.o:
	$(CHARMC) ComlibManager.C

cifiles: ComlibManager.ci
	$(CHARMC) ComlibManager.ci

clean:
	rm -f *~ ComlibModule.decl.h ComlibModule.def.h *.o 
