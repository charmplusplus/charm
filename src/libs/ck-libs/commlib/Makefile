CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc -I../../.. $(OPTS)
CHARMINC=$(CDIR)/include
CHARMBIN=$(CDIR)/bin
LIBDIR=$(CDIR)/lib
COMMLIBDIR=../../conv-libs/commlib

DEPENDFILE = Make.depends

LIB = $(LIBDIR)/libmodulecommlib.a

STRATEGY_OBJS = MPIStrategy.o StreamingStrategy.o \
		DummyStrategy.o NodeMulticast.o EachToManyMulticastStrategy.o \
		MsgPacker.o DirectMulticastStrategy.o RingMulticastStrategy.o \
		PipeBroadcastStrategy.o MeshStreamingStrategy.o \
		PrioStreaming.o BroadcastStrategy.o

CONVERSE_COM_OBJS = $(COMMLIBDIR)/commlib.o $(COMMLIBDIR)/gridrouter.o \
	$(COMMLIBDIR)/de.o $(COMMLIBDIR)/treerouter.o $(COMMLIBDIR)/petable.o\
	$(COMMLIBDIR)/overlapper.o $(COMMLIBDIR)/rsend.o \
	$(COMMLIBDIR)/bcast.o $(COMMLIBDIR)/3dgridrouter.o  \
	$(COMMLIBDIR)/graphrouter.o  $(COMMLIBDIR)/hypercubetopology.o

CIFILES = ComlibManager.ci

OBJS = ComlibManager.o $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)

all: 
	$(MAKE) conversecom 
	$(MAKE) $(LIB) 

$(LIB): $(OBJS) $(STRATEGY_OBJS) $(CONVERSE_COM_OBJS)
	$(CHARMC) -o $(LIB) $(OBJS)
	rm -f $(CDIR)/include/ComlibManager.h
	cp  ComlibManager.h $(CDIR)/include
	rm -f $(CDIR)/include/EachToManyMulticastStrategy.h
	cp  EachToManyMulticastStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/BroadcastStrategy.h
	cp  BroadcastStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/DirectMulticastStrategy.h
	cp  DirectMulticastStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/RingMulticastStrategy.h
	cp  RingMulticastStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/DummyStrategy.h
	cp  DummyStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/NodeMulticast.h
	cp  NodeMulticast.h $(CDIR)/include
	rm -f $(CDIR)/include/StreamingStrategy.h
	cp  StreamingStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/PipeBroadcastStrategy.h
	cp  PipeBroadcastStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/MeshStreamingStrategy.h
	cp  MeshStreamingStrategy.h $(CDIR)/include
	rm -f $(CDIR)/include/PrioStreaming.h
	cp  PrioStreaming.h $(CDIR)/include
	rm -f $(CDIR)/include/commlib.decl.h
	cp  commlib.decl.h $(CDIR)/include
	rm -f $(CDIR)/include/commlib.def.h
	cp  commlib.def.h $(CDIR)/include

conversecom :
	(cd ../../conv-libs/commlib; $(MAKE) )

include $(DEPENDFILE)

commlib.decl.h : commlib.def.h

commlib.def.h: ComlibManager.ci
	$(CHARMC) ComlibManager.ci

clean:
	rm -f *~ *.decl.h *.def.h *.o $(LIB) \
	$(CDIR)/include/ComlibManager.h      \
	$(CDIR)/include/EachToManyMulticastStrategy.h \
	$(CDIR)/include/DummyStrategy.h      \
	$(CDIR)/include/NodeMulticast.h      \
	$(CDIR)/include/StreamingStrategy.h  \
	$(CDIR)/include/PipeBroadcastStrategy.h \
	$(CDIR)/include/BroadcastStrategy.h \
	$(CDIR)/include/MeshStreamingStrategy.h \
	$(CDIR)/include/PrioStreaming.h \
	$(CDIR)/include/commlib.decl.h       \
	$(CDIR)/include/commlib.def.h        \
	$(CDIR)/lib/libcomm.a                \
	$(CDIR)/lib/libmodulecomlib.a

depends:  $(CIFILES)
	echo "Creating " $(DEPENDFILE) " ..."; 
	if [ -f $(DEPENDFILE) ]; then \
           /bin/cp -f $(DEPENDFILE) $(DEPENDFILE).old; \
        fi; \
	echo > $(DEPENDFILE); \
        for i in ComlibManager.o $(STRATEGY_OBJS) ; do \
	      SRCFILE=`basename $$i .o`.C ; \
              echo "checking dependencies for $$SRCFILE" ; \
              g++ -MM  -I../../.. -I$(CHARMINC) $$SRCFILE | \
              perl $(CHARMBIN)/dep.pl $(CHARMINC)  ../../.. /usr/include /usr/local >> $(DEPENDFILE); \
              echo '	$$(CHARMC) -c -o '$$i $$SRCFILE >> $(DEPENDFILE) ; \
	      echo '' >> $(DEPENDFILE) ; \
        done; 

