CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

GENHEADERS=NDMeshStreamer.decl.h NDMeshStreamer.def.h
HEADERS=NDMeshStreamer.h DataItemTypes.h $(GENHEADERS)
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
	cp $^ $(CDIR)/include/
	touch $@

NDMeshStreamer.o: NDMeshStreamer.C $(HEADERS)
	$(CHARMC) -c $<

$(GENHEADERS): NDMeshStreamer.stamp

NDMeshStreamer.stamp: NDMeshStreamer.ci
	$(CHARMC) $<
	touch $@

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h *.stamp gmon.out $(DEST) conv-host charmrun headers
