CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
COMMLIBDIR=../../conv-libs/commlib

LIB = libcomm.a

STRATEGY_OBJS = EachToManyStrategy.o MPIStrategy.o StreamingStrategy.o \
		DummyStrategy.o NodeMulticast.o

CONVERSE_COM_OBJS = $(COMMLIBDIR)/commlib.o $(COMMLIBDIR)/gridrouter.o \
	$(COMMLIBDIR)/de.o $(COMMLIBDIR)/treerouter.o $(COMMLIBDIR)/petable.o \
	$(COMMLIBDIR)/overlapper.o $(COMMLIBDIR)/rsend.o $(COMMLIBDIR)/bcast.o \
	$(COMMLIBDIR)/charm_bind.o $(COMMLIBDIR)/3dgridrouter.o

OBJS = ComlibManager.o $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)

all: cifiles conversecom $(LIB) 
	rm -f $(CDIR)/include/ComlibManager.h
	cp  ComlibManager.h $(CDIR)/include
	rm -f $(CDIR)/include/EachToManyStrategy.h
	cp  EachToManyStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/DummyStrategy.h
	cp  DummyStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/NodeMulticast.h
	cp  NodeMulticast.h $(CDIR)/include
	rm -f $(CDIR)/include/StreamingStrategy.h
	cp  StreamingStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/ComlibModule.decl.h
	cp  ComlibModule.decl.h $(CDIR)/include
	rm -f $(CDIR)/include/ComlibModule.def.h
	cp  ComlibModule.def.h $(CDIR)/include
	rm -f $(CDIR)/tmp/ComlibManager.ci
	cp  ComlibManager.ci $(CDIR)/tmp
	rm -f $(CDIR)/lib/libcomm.a
	cp libcomm.a $(CDIR)/lib/

conversecom :
	(cd ../../conv-libs/commlib; make)

libcomm.a: $(OBJS) $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)
	$(CHARMC) -cp ../../../../../lib -o libcomm.a $(OBJS)

ComlibManager.o: ComlibManager.C ComlibManager.h
	$(CHARMC) -c ComlibManager.C

EachToManyStrategy.o: EachToManyStrategy.C EachToManyStrategy.h ComlibManager.h
	$(CHARMC) -c EachToManyStrategy.C

MPIStrategy.o: MPIStrategy.C MPIStrategy.h ComlibManager.h
	$(CHARMC) -c MPIStrategy.C

StreamingStrategy.o: StreamingStrategy.C StreamingStrategy.h ComlibManager.h
	$(CHARMC) -c StreamingStrategy.C

DummyStrategy.o: DummyStrategy.C DummyStrategy.h ComlibManager.h
	$(CHARMC) -c DummyStrategy.C

NodeMulticast.o: NodeMulticast.C NodeMulticast.h ComlibManager.h
	$(CHARMC) -c NodeMulticast.C

cifiles: ComlibManager.ci
	$(CHARMC) ComlibManager.ci

clean:
	rm -f *~ ComlibModule.decl.h ComlibModule.def.h *.o 
