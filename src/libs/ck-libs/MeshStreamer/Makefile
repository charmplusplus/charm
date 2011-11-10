CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=MeshStreamer.h MeshStreamer.decl.h MeshStreamer.def.h
OBJS=MeshStreamer.o
LIB=libmoduleMeshStreamer

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a

.SUFFIXES:
.SUFFIXES: .C .o
                                                                                
.C.o:
	$(CHARMC) -c $<

all: $(DEST)

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@
	#cp $(LIB).dep $(LIBDIR)/$(LIB).dep

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

MeshStreamer.o: MeshStreamer.C MeshStreamer.decl.h $(HEADERS)
	$(CHARMC) -c MeshStreamer.C

MeshStreamer.decl.h MeshStreamer.def.h: MeshStreamer.ci
	$(CHARMC) MeshStreamer.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) conv-host charmrun headers
