CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
AMPICC=$(CDIR)/bin/ampiCC $(OPTS)


HEADERS=fem.h femf.h fem_mesh.h fem_impl.h parallel_part.h map.h  fem_mesh_modify.h FEMMeshModify.decl.h fem_adapt_new.h fem_adapt_algs.h fem_interpolate.h fem_lock.h fem_util.h fem_lock_node.h fem_adapt_lock.h fem_adapt_if.h
HEADDEP=$(HEADERS) fem_impl.h 
OBJS=fem.o fem_mesh.o symmetries.o \
	partition.o map.o fem_compat.o call_init.o \
	parallel_part.o fem_mesh_modify.o fem_mesh_adjacency.o \
	fem_adapt_new.o fem_adapt_algs.o fem_interpolate.o \
	fem_lock.o fem_util.o fem_lock_node.o fem_adapt_lock.o \
	fem_adapt_if.o
COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
LIB=libmodulefem

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libfem-compat.a

FEMMAIN=$(LIBDIR)/libfemmain.o

.SUFFIXES:
.SUFFIXES: .c .o .C

.c.o:
	$(CHARMC) -c $<

all: $(DEST) $(COMPATLIB) $(FEMMAIN)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

compat_init.o: compat_init.c

compat_finit.o: compat_finit.c

compat_driver.o: compat_driver.c

compat_fdriver.o: compat_fdriver.c

fem.o: fem.C $(HEADDEP)
	$(CHARMC) -c fem.C

partition.o: partition.C $(HEADDEP)
	$(CHARMC) -c partition.C

fem.decl.h: fem.ci 
	$(CHARMC) fem.ci

parallel_part.o: parallel_part.C fem.decl.h $(HEADDEP)
	$(CHARMC) -c parallel_part.C

map.o: map.C $(HEADDEP)
	$(CHARMC) -c map.C

symmetries.o: symmetries.C $(HEADDEP)
	$(CHARMC) -c symmetries.C

datatype.o: datatype.C $(HEADDEP)
	$(CHARMC) -c datatype.C

fem_compat.o: fem_compat.C $(HEADDEP)
	$(CHARMC) -c fem_compat.C

fem_mesh.o: fem_mesh.C $(HEADDEP)
	$(AMPICC) -c fem_mesh.C

#fem_adapt.o: fem_adapt.C $(HEADDEP)
#	$(CHARMC) -c fem_adapt.C

fem_adapt_new.o: fem_adapt_new.C $(HEADDEP)
	$(AMPICC) -c fem_adapt_new.C

fem_adapt_algs.o: fem_adapt_algs.C $(HEADDEP)
	$(AMPICC) -c fem_adapt_algs.C

fem_interpolate.o: fem_interpolate.C $(HEADDEP)
	$(AMPICC) -c fem_interpolate.C

fem_lock.o: fem_lock.C $(HEADDEP)
	$(AMPICC) -c fem_lock.C

fem_util.o: fem_util.C $(HEADDEP)
	$(AMPICC) -c fem_util.C

fem_lock_node.o: fem_lock_node.C $(HEADDEP)
	$(AMPICC) -c fem_lock_node.C

fem_adapt_lock.o: fem_adapt_lock.C $(HEADDEP)
	$(AMPICC) -c fem_adapt_lock.C

fem_adapt_if.o: fem_adapt_if.C $(HEADDEP)
	$(AMPICC) -c fem_adapt_if.C

fem_comm.o: fem_comm.C $(HEADDEP)
	$(CHARMC) -c fem_comm.C

$(FEMMAIN): femmain.C $(HEADDEP)
	$(AMPICC) -c femmain.C -o $(FEMMAIN)

call_init.o: call_init.c
	$(CHARMC) -c call_init.c

FEMMeshModify.decl.h: fem_mesh_modify.ci
	$(CHARMC) fem_mesh_modify.ci

fem_mesh_modify.o: FEMMeshModify.decl.h fem_mesh_modify.C fem_mesh_modify.h $(HEADDEP)
	$(AMPICC) -c fem_mesh_modify.C

fem_mesh_adjacency.o: fem_mesh_adjacency.C $(HEADDEP)
	$(CHARMC) -c fem_mesh_adjacency.C

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) $(COMPATLIB) headers
