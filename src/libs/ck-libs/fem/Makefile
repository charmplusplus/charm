CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS= fem.h femf.h fem_mesh.h fem_impl.h parallel_part.h map.h \
		 fem_mesh_modify.h  fem_adapt_new.h fem_collide.h         \
		 fem_adapt_algs.h fem_interpolate.h fem_lock.h fem_util.h \
		 fem_lock_node.h fem_adapt_lock.h fem_adapt_if.h     

GENHEADERS= FEMMeshModify.decl.h fem.decl.h FEMMeshModify.def.h

HEADDEP= $(GENHEADERS) $(HEADERS)

OBJS=fem.o fem_mesh.o symmetries.o partition.o map.o fem_compat.o \
	 call_init.o parallel_part.o fem_mesh_modify.o                \
	 fem_mesh_adjacency.o fem_adapt_new.o fem_adapt_algs.o        \
	 fem_interpolate.o fem_lock.o fem_util.o fem_lock_node.o      \
	 fem_adapt_lock.o fem_collide.o                               

COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
LIB=libmodulefem

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libfem-compat.a

#Should this be here?:
FEMMAIN=$(LIBDIR)/libfemmain.o

#Automatic rules:
.SUFFIXES: .c .o .C .decl.h .ci .def.h
.ci.decl.h: $< $(HEADERS)
	$(CHARMC) $<
.ci.def.h: $< $(HEADERS)
	$(CHARMC) $<
.c.o: $< $(HEADDEP)
	$(CHARMC) -c $<
.C.o: $< $(HEADDEP)
	$(CHARMC) -c $<

all: $(DEST) $(COMPATLIB) $(FEMMAIN)


#Manual rules since some dependencies are a little screwy. These should be fixed
fem_mesh_modify.o: $(HEADDEP)
	$(CHARMC) -c fem_mesh_modify.C
FEMMeshModify.decl.h: $(HEADERS)
	$(CHARMC) fem_mesh_modify.ci
FEMMeshModify.def.h: $(HEADERS)
	$(CHARMC) fem_mesh_modify.ci

# Build libmodulefem
$(DEST): $(OBJS) $(COMPAT) headers $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Build libfem-compat
$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

# Copy header files to include directory
headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	cp vector2d.h $(CDIR)/include/
	touch headers

# clean up
clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) $(COMPATLIB) headers
