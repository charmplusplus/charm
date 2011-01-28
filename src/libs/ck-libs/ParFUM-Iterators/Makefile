CDIR    = ../../../..

CUDADIR = /usr/local/cuda
CUDASDK = $(HOME)/TOPS-Tesla/NVIDIA_CUDA_SDK
CUDAC   = nvcc
CUDAINC = -I$(CUDASDK)/common/inc -I$(CUDADIR)/include

CHARMC  = $(CDIR)/bin/charmc $(OPTS) #-nobs -cc $(CUDAC) -c++ $(CUDAC)
CCFLAGS = -DFP_TYPE_FLOAT -DSHARED_NODES_ONLY_NEIGHBOR #-DCUDA


#Headers to be copied to include directory so application users can see them
HEADERS= ParFUM_Iterators.h ParFUM_Iterators_CUDA.h ParFUM_Iterators_Types.h $(INTERNALHEADERS)

#Headers generated from .ci files
GENHEADERS= ParFUM_Iterators.decl.h 

HEADDEP= $(GENHEADERS) $(HEADERS) $(INTERNALHEADERS)

OBJS=ParFUM_Iterators.o ParFUM_Iterators_CPU.o #ParFUM_Iterators_CUDA.o

LIB=libmoduleParFUM_Iterators

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a

INCS=-I. $(CUDAINC)

all: $(DEST)

ParFUM_Iterators.decl.h: ParFUM_Iterators.ci
	$(CHARMC) ParFUM_Iterators.ci

# Build libmoduleParFUM
$(DEST): $(OBJS) headers $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Copy header files to include directory
.PHONY: headers
headers: $(HEADDEP)
	cp $(HEADERS) $(CDIR)/include/

ParFUM_Iterators.o: ParFUM_Iterators.cc $(HEADDEP) ParFUM_Iterators.h
	$(CHARMC) -c ParFUM_Iterators.cc $(INCS) $(CCFLAGS)

ParFUM_Iterators_CPU.o: ParFUM_Iterators_CPU.cc $(HEADDEP)
	$(CHARMC) -c ParFUM_Iterators_CPU.cc $(CCFLAGS) $(INCS)

ParFUM_Iterators_CUDA.o: ParFUM_Iterators_CUDA.cu $(HEADDEP)
	$(CHARMC) -c ParFUM_Iterators_CUDA.cu $(CCFLAGS) $(INCS)


# clean up
clean:
	rm -rf *.a *.def.h *.decl.h *.o *~ SunWS_cache $(DEST) headers docs doxygen_warn_logfile
