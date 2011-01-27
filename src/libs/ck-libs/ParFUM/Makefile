
CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
MPICC=$(CDIR)/bin/ampiCC $(OPTS)

#Headers internal to ParFUM
INTERNALHEADERS= ParFUM_internals.h ParFUM_locking.h ParFUM_Mesh_Modify.h ParFUM_Adapt_Algs.h ParFUM_Adapt.h ParFUM_Interpolate.h ParFUM_SA.h import.h adapt_adj.h bulk_adapt_ops.h bulk_adapt.h

#Headers to be copied to include directory so application users can see them
HEADERS= ParFUM.h ParFUMf.h ParFUM_Types.h $(INTERNALHEADERS)

#Headers generated from .ci files
GENHEADERS= ParFUM.def.h ParFUM.decl.h ParFUM_Adapt.decl.h ParFUM_Adapt.def.h ParFUM_SA.def.h ParFUM_SA.decl.h

HEADDEP= $(GENHEADERS) $(HEADERS) $(INTERNALHEADERS)

OBJS=ParFUM.o mesh.o symmetries.o partition.o map.o compat.o call_init.o parallel_part.o mesh_modify.o mesh_adjacency.o adapt.o adapt_algs.o adapt_if.o interpolate.o lock.o util.o lock_node.o adapt_lock.o ParFUM_SA.o global_numbering.o import.o adapt_adj.o bulk_adapt_ops.o bulk_adapt.o mesh_feature_detect.o MsaHashtable.o

COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
LIB=libmoduleParFUM

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libParFUM-compat.a
PARFUMMAIN=$(LIBDIR)/libParFUMmain.o

INCS=-I.

.PHONY: all
all: $(DEST) $(COMPATLIB) $(PARFUMMAIN) headers

ParFUM.def.h: ParFUM.decl.h
ParFUM.decl.h: ParFUM.ci
	$(CHARMC) ParFUM.ci
ParFUM_Adapt.def.h: ParFUM_Adapt.decl.h
ParFUM_Adapt.decl.h: mesh_modify.ci
	$(CHARMC) mesh_modify.ci
ParFUM_SA.def.h: ParFUM_SA.decl.h
ParFUM_SA.decl.h: ParFUM_SA.ci
	$(CHARMC) ParFUM_SA.ci

# Build libmoduleParFUM
$(DEST): $(OBJS) $(COMPAT) $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Build libParFUM-compat
$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

# Build fortran file with definitions for init and driver. 
# This allows us to use a single library with either fortran or c
$(PARFUMMAIN): main.C $(HEADDEP)
	$(MPICC) main.C -o $(PARFUMMAIN)

# Copy header files to include directory
.PHONY: headers
headers: $(HEADDEP)
	cp $(HEADERS) $(CDIR)/include/
	cp $(GENHEADERS) $(CDIR)/include/

#Automatic rules for some reason break because they don't use the dependencies
#.SUFFIXES: .c .o .C
#.c.o: $(HEADDEP) $< 
#	$(CHARMC) -c $<
#.C.o: $(HEADDEP) $<
#	$(CHARMC) -c $<


MsaHashtable.o: MsaHashtable.C $(HEADDEP)
	$(MPICC) -DFOR_SUN_CC_ONLY -c MsaHashtable.C $(INCS)

mesh_modify.o: mesh_modify.C $(HEADDEP)
	$(MPICC) -c mesh_modify.C $(INCS)

ParFUM_SA.o: ParFUM_SA.C $(HEADDEP)
	$(MPICC) -c ParFUM_SA.C $(INCS)

ParFUM.o: ParFUM.C $(HEADDEP)
	$(MPICC) -c ParFUM.C $(INCS)

mesh.o: mesh.C $(HEADDEP)
	$(MPICC) -c mesh.C $(INCS)

symmetries.o: symmetries.C $(HEADDEP)
	$(MPICC) -c symmetries.C $(INCS)

partition.o: partition.C $(HEADDEP)
	$(MPICC) -c partition.C $(INCS)

map.o: map.C $(HEADDEP)
	$(MPICC) -c map.C $(INCS)

compat.o: compat.C $(HEADDEP)
	$(MPICC) -c compat.C $(INCS)

call_init.o: call_init.c $(HEADDEP)
	$(CHARMC) -c call_init.c $(INCS)

parallel_part.o: parallel_part.C $(HEADDEP) MsaHashtable.h
	$(MPICC) -DFOR_SUN_CC_ONLY -c parallel_part.C $(INCS)

mesh_adjacency.o: mesh_adjacency.C $(HEADDEP)
	$(MPICC) -c mesh_adjacency.C $(INCS)

mesh_feature_detect.o: mesh_feature_detect.C $(HEADDEP)
	$(MPICC) -c mesh_feature_detect.C $(INCS)

adapt.o: adapt.C $(HEADDEP)
	$(MPICC) -c adapt.C $(INCS)

adapt_if.o: adapt_if.C $(HEADDEP)
	$(MPICC) -c adapt_if.C $(INCS)

adapt_algs.o: adapt_algs.C $(HEADDEP)
	$(MPICC) -c adapt_algs.C $(INCS)

interpolate.o: interpolate.C $(HEADDEP)
	$(MPICC) -c interpolate.C $(INCS)

lock.o: lock.C $(HEADDEP)
	$(MPICC) -c lock.C $(INCS)

util.o: util.C $(HEADDEP)
	$(MPICC) -c util.C $(INCS)

lock_node.o: lock_node.C $(HEADDEP)
	$(MPICC) -c lock_node.C $(INCS)

adapt_lock.o: adapt_lock.C $(HEADDEP)
	$(MPICC) -c adapt_lock.C $(INCS)

collide.o: collide.C $(HEADDEP)
	$(CHARMC) -c collide.C $(INCS)

compat_init.o: compat_init.c $(HEADDEP)
	$(CHARMC) -c compat_init.c $(INCS)

compat_finit.o: compat_finit.c $(HEADDEP)
	$(CHARMC) -c compat_finit.c $(INCS)

compat_driver.o: compat_driver.c $(HEADDEP)
	$(CHARMC) -c compat_driver.c $(INCS)

compat_fdriver.o: compat_fdriver.c $(HEADDEP)
	$(CHARMC) -c compat_fdriver.c $(INCS)

global_numbering.o: global_numbering.C $(HEADDEP)
	$(MPICC) -c global_numbering.C $(INCS)

import.o: import.C $(HEADDEP)
	$(MPICC) -c import.C $(INCS)

adapt_adj.o: adapt_adj.C $(HEADDEP)
	$(MPICC) -c adapt_adj.C $(INCS)

bulk_adapt_ops.o: bulk_adapt_ops.C $(HEADDEP)
	$(MPICC) -c bulk_adapt_ops.C $(INCS)

bulk_adapt.o: bulk_adapt.C $(HEADDEP)
	$(MPICC) -c bulk_adapt.C $(INCS)

docs: ParFUM.doxy ParFUM_Adapt_Algs.h ParFUM_Adapt.h ParFUMf.h ParFUM.h ParFUM_internals.h ParFUM_Interpolate.h ParFUM_locking.h ParFUM_Mesh_Modify.h adapt_algs.C adapt.C adapt_if.C adapt_lock.C collide.C compat.C interpolate.C lock.C lock_node.C main.C map.C mesh_adjacency.C mesh.C mesh_modify.C MsaHashtable.C parallel_part.C ParFUM.C partition.C symmetries.C util.C call_init.c compat_driver.c compat_fdriver.c compat_finit.c compat_init.c ParFUM_SA.C global_numbering.C import.C import.h
	doxygen ParFUM.doxy

# clean up
.PHONY: clean
clean:
	rm -rf *.a *.def.h *.decl.h *.o *~ SunWS_cache $(DEST) $(COMPATLIB) $(GENHEADERS)
