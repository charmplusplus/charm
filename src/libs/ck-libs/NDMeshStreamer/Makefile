CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=NDMeshStreamer.h DataItemTypes.h NDMeshStreamer.decl.h NDMeshStreamer.def.h
OBJS=NDMeshStreamer.o
LIB=libmoduleNDMeshStreamer

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

NDMeshStreamer.o: NDMeshStreamer.C NDMeshStreamer.decl.h $(HEADERS)
	$(CHARMC) -c NDMeshStreamer.C

NDMeshStreamer.decl.h NDMeshStreamer.def.h: NDMeshStreamer.ci
	$(CHARMC) NDMeshStreamer.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) conv-host charmrun headers
