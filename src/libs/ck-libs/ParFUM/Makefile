CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

#Headers internal to ParFUM
INTERNALHEADERS= ParFUM_internals.h ParFUM_locking.h ParFUM_Mesh_Modify.h ParFUM_Adapt_Algs.h ParFUM_Adapt.h ParFUM_Interpolate.h

#Headers to be copied to include directory so application users can see them
HEADERS= ParFUM.h ParFUMf.h $(INTERNALHEADERS)

#Headers generated from .ci files
GENHEADERS= ParFUM.def.h ParFUM.decl.h fem.decl.h fem.def.h

HEADDEP= $(GENHEADERS) $(HEADERS) $(INTERNALHEADERS)

OBJS=ParFUM.o mesh.o symmetries.o partition.o map.o compat.o call_init.o parallel_part.o mesh_modify.o mesh_adjacency.o adapt.o adapt_algs.o adapt_if.o interpolate.o lock.o util.o lock_node.o adapt_lock.o             

COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
LIB=libmoduleParFUM

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libParFUM-compat.a
PARFUMMAIN=$(LIBDIR)/libParFUMmain.o

all: $(DEST) $(COMPATLIB) $(PARFUMMAIN)

fem.decl.h: ParFUM.ci
	$(CHARMC) ParFUM.ci
fem.def.h: ParFUM.ci
	$(CHARMC) ParFUM.ci
ParFUM.def.h: mesh_modify.ci
	$(CHARMC) mesh_modify.ci
ParFUM.decl.h: mesh_modify.ci
	$(CHARMC) mesh_modify.ci


# Build libmoduleParFUM
$(DEST): $(OBJS) $(COMPAT) headers $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Build libParFUM-compat
$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

# Build fortran file with definitions for init and driver. 
# This allows us to use a single library with either fortran or c
$(PARFUMMAIN): main.C $(HEADDEP)
	$(CHARMC) main.C -o $(PARFUMMAIN)

# Copy header files to include directory
headers: $(HEADDEP)
	cp $(HEADERS) $(CDIR)/include/
	cp $(GENHEADERS) $(CDIR)/include/
	touch headers


#Automatic rules for some reason break because they don't use the dependencies
#.SUFFIXES: .c .o .C
#.c.o: $(HEADDEP) $< 
#	$(CHARMC) -c $<
#.C.o: $(HEADDEP) $<
#	$(CHARMC) -c $<


mesh_modify.o: mesh_modify.C $(HEADDEP)
	$(CHARMC) -c mesh_modify.C

ParFUM.o: ParFUM.C $(HEADDEP)
	$(CHARMC) -c ParFUM.C

mesh.o: mesh.C $(HEADDEP)
	$(CHARMC) -c mesh.C

symmetries.o: symmetries.C $(HEADDEP)
	$(CHARMC) -c symmetries.C

partition.o: partition.C $(HEADDEP)
	$(CHARMC) -c partition.C

map.o: map.C $(HEADDEP)
	$(CHARMC) -c map.C

compat.o: compat.C $(HEADDEP)
	$(CHARMC) -c compat.C

call_init.o: call_init.c $(HEADDEP)
	$(CHARMC) -c call_init.c

parallel_part.o: parallel_part.C $(HEADDEP)
	$(CHARMC) -c parallel_part.C

mesh_adjacency.o: mesh_adjacency.C $(HEADDEP)
	$(CHARMC) -c mesh_adjacency.C

adapt.o: adapt.C $(HEADDEP)
	$(CHARMC) -c adapt.C

adapt_if.o: adapt_if.C $(HEADDEP)
	$(CHARMC) -c adapt_if.C

adapt_algs.o: adapt_algs.C $(HEADDEP)
	$(CHARMC) -c adapt_algs.C

interpolate.o: interpolate.C $(HEADDEP)
	$(CHARMC) -c interpolate.C

lock.o: lock.C $(HEADDEP)
	$(CHARMC) -c lock.C

util.o: util.C $(HEADDEP)
	$(CHARMC) -c util.C

lock_node.o: lock_node.C $(HEADDEP)
	$(CHARMC) -c lock_node.C

adapt_lock.o: adapt_lock.C $(HEADDEP)
	$(CHARMC) -c adapt_lock.C

collide.o: collide.C $(HEADDEP)
	$(CHARMC) -c collide.C

compat_init.o: compat_init.c $(HEADDEP)
	$(CHARMC) -c compat_init.c

compat_finit.o: compat_finit.c $(HEADDEP)
	$(CHARMC) -c compat_finit.c

compat_driver.o: compat_driver.c $(HEADDEP)
	$(CHARMC) -c compat_driver.c

compat_fdriver.o: compat_fdriver.c $(HEADDEP)
	$(CHARMC) -c compat_fdriver.c

docs: ParFUM.doxy ParFUM_Adapt_Algs.h ParFUM_Adapt.h ParFUMf.h ParFUM.h ParFUM_internals.h ParFUM_Interpolate.h ParFUM_locking.h ParFUM_Mesh_Modify.h adapt_algs.C adapt.C adapt_if.C adapt_lock.C collide.C compat.C interpolate.C lock.C lock_node.C main.C map.C mesh_adjacency.C mesh.C mesh_modify.C parallel_part.C ParFUM.C partition.C symmetries.C util.C call_init.c compat_driver.c compat_fdriver.c compat_finit.c compat_init.c
	doxygen ParFUM.doxy

# clean up
clean:
	rm -rf *.a *.def.h *.decl.h *.o *~ SunWS_cache $(DEST) $(COMPATLIB) $(GENHEADERS)
