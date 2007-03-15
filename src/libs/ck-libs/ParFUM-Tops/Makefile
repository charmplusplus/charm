CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
CUDAC=$(CHARMC)

#Headers to be copied to include directory so application users can see them
HEADERS= ParFUM_TOPS.h ParFUM_TOPS_CUDA.h ParFUM_TOPS_Types.h $(INTERNALHEADERS)

#Headers generated from .ci files
GENHEADERS= ParFUM_TOPS.def.h ParFUM_TOPS.decl.h 

HEADDEP= $(GENHEADERS) $(HEADERS) $(INTERNALHEADERS)

OBJS=ParFUM_TOPS.o ParFUM_TOPS_Iterators.o ParFUM_TOPS_CUDA.o

LIB=libmoduleParFUM_TOPS

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a

INCS=-I.

all: $(DEST)


ParFUM_TOPS.decl.h: ParFUM_TOPS.ci
	$(CHARMC) ParFUM_TOPS.ci
ParFUM_TOPS.def.h: ParFUM_TOPS.ci
	$(CHARMC) ParFUM_TOPS.ci

# Build libmoduleParFUM
$(DEST): $(OBJS) headers $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Copy header files to include directory
.PHONY: headers
headers: $(HEADDEP)
	cp $(HEADERS) $(CDIR)/include/

ParFUM_TOPS.o: ParFUM_TOPS.C $(HEADDEP) ParFUM_TOPS.h
	$(CHARMC) -c ParFUM_TOPS.C $(INCS)

ParFUM_TOPS_Iterators.o: ParFUM_TOPS_Iterators.C $(HEADDEP)
	$(CHARMC) -c ParFUM_TOPS_Iterators.C $(INCS)

ParFUM_TOPS_CUDA.o: ParFUM_TOPS_CUDA.C $(HEADDEP)
	$(CUDAC) -c ParFUM_TOPS_CUDA.C $(INCS)


# clean up
clean:
	rm -rf *.a *.def.h *.decl.h *.o *~ SunWS_cache $(DEST) headers docs doxygen_warn_logfile
