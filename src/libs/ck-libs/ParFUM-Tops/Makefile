CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

#Headers to be copied to include directory so application users can see them
HEADERS= ParFUM_TOPS.h  $(INTERNALHEADERS)

HEADDEP= $(GENHEADERS) $(HEADERS) $(INTERNALHEADERS)

OBJS=ParFUM_TOPS.o

LIB=libmoduleParFUM_TOPS

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
PARFUMMAIN=$(LIBDIR)/libParFUMmain.o

INCS=-I.

all: $(DEST)


# Build libmoduleParFUM
$(DEST): $(OBJS) headers $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Copy header files to include directory
headers: $(HEADDEP)
	cp $(HEADERS) $(CDIR)/include/
	cp $(GENHEADERS) $(CDIR)/include/
	touch headers

ParFUM_TOPS.o: ParFUM_TOPS.C $(HEADDEP) ParFUM_TOPS.h
	$(CHARMC) -c ParFUM_TOPS.C $(INCS)


# clean up
clean:
	rm -rf *.a *.def.h *.decl.h *.o *~ SunWS_cache $(DEST)
