CHARMC=$(CDIR)/bin/charmc $(OPTS)
COMMLIBDIR=../../conv-libs/commlib
CDIR=../../../..
LIBDIR=$(CDIR)/lib

LIB = libmodulecommlib.a

STRATEGY_OBJS = EachToManyStrategy.o MPIStrategy.o StreamingStrategy.o \
		DummyStrategy.o NodeMulticast.o EachToManyMulticastStrategy.o

CONVERSE_COM_OBJS = $(COMMLIBDIR)/commlib.o $(COMMLIBDIR)/gridrouter.o \
	$(COMMLIBDIR)/de.o $(COMMLIBDIR)/treerouter.o $(COMMLIBDIR)/petable.o\
	$(COMMLIBDIR)/overlapper.o $(COMMLIBDIR)/rsend.o \
	$(COMMLIBDIR)/bcast.o $(COMMLIBDIR)/3dgridrouter.o  \
	$(COMMLIBDIR)/graphrouter.o  $(COMMLIBDIR)/hypercubetopology.o

OBJS = ComlibManager.o $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)

all: cifiles conversecom $(LIB) 
	rm -f $(CDIR)/include/ComlibManager.h
	cp  ComlibManager.h $(CDIR)/include
	rm -f $(CDIR)/include/EachToManyStrategy.h
	cp  EachToManyStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/EachToManyMulticastStrategy.h
	cp  EachToManyMulticastStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/DummyStrategy.h
	cp  DummyStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/NodeMulticast.h
	cp  NodeMulticast.h $(CDIR)/include
	rm -f $(CDIR)/include/StreamingStrategy.h
	cp  StreamingStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/commlib.decl.h
	cp  commlib.decl.h $(CDIR)/include
	rm -f $(CDIR)/include/commlib.def.h
	cp  commlib.def.h $(CDIR)/include

conversecom :
	(cd ../../conv-libs/commlib; make)

libmodulecommlib.a: $(OBJS) $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)
	$(CHARMC) -cp $(LIBDIR) -o libmodulecommlib.a $(OBJS)

ComlibManager.o: ComlibManager.C ComlibManager.h
	$(CHARMC) -c ComlibManager.C

EachToManyStrategy.o: EachToManyStrategy.C EachToManyStrategy.h ComlibManager.h
	$(CHARMC) -c EachToManyStrategy.C

EachToManyMulticastStrategy.o: EachToManyMulticastStrategy.C EachToManyMulticastStrategy.h \
	ComlibManager.h
	$(CHARMC) -c EachToManyMulticastStrategy.C

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
	rm -f *~ *.decl.h *.def.h *.o $(LIB) \
	$(CDIR)/include/ComlibManager.h      \
	$(CDIR)/include/EachToManyStrategy.h \
	$(CDIR)/include/EachToManyMulticastStrategy.h \
	$(CDIR)/include/DummyStrategy.h      \
	$(CDIR)/include/NodeMulticast.h      \
	$(CDIR)/include/StreamingStrategy.h  \
	$(CDIR)/include/commlib.decl.h       \
	$(CDIR)/include/commlib.def.h        \
        $(CDIR)/lib/libcomm.a                \
	$(CDIR)/lib/libmodulecomlib.a