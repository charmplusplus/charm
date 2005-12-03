CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS= ParFUM.h ParFUMf.h

INTERNALHEADERS= mesh.h impl.h parallel_part.h map.h    \
		mesh_modify.h  ParFUM_adapt.h collide.h    \
		adapt_algs.h interpolate.h lock.h util.h   \
		lock_node.h adapt_lock.h ParFUM_adapt_if.h

GENHEADERS= FEMMeshModify.def.h	FEMMeshModify.decl.h   \
		fem.decl.h fem.def.h

HEADDEP= $(GENHEADERS) $(HEADERS) $(INTERNALHEADERS)

OBJS=ParFUM.o mesh.o symmetries.o partition.o map.o compat.o \
	 call_init.o parallel_part.o mesh_modify.o                \
	 mesh_adjacency.o adapt.o adapt_algs.o        \
	 interpolate.o lock.o util.o lock_node.o      \
	 adapt_lock.o collide.o                               

COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
LIB=libmoduleParFUM

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libParFUM-compat.a

#Should this be here?:
PARFUMMAIN=$(LIBDIR)/libParFUMmain.o

#Automatic rules:
.SUFFIXES: .c .o .C .decl.h .ci .def.h
.ci.decl.h: $< $(HEADDEP)
	$(CHARMC) $<
.ci.def.h: $< $(HEADDEP)
	$(CHARMC) $<
.c.o: $< $(HEADDEP)
	$(CHARMC) -c $<
.C.o: $< $(HEADDEP)
	$(CHARMC) -c $<

all: $(DEST) $(COMPATLIB) $(PARFUMMAIN)


#Manual rules since some dependencies are a little screwy. These should be fixed
fem.decl.h: ParFUM.ci
	$(CHARMC) ParFUM.ci
fem.def.h: ParFUM.ci
	$(CHARMC) ParFUM.ci

FEMMeshModify.def.h: mesh_modify.ci
	$(CHARMC) mesh_modify.ci
FEMMeshModify.decl.h: mesh_modify.ci
	$(CHARMC) mesh_modify.ci


# Build libmoduleParFUM
$(DEST): $(OBJS) $(COMPAT) headers $(HEADDEP)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

# Build libParFUM-compat
$(COMPATLIB): $(COMPAT)
	$(CHARMC) $(COMPAT) -o $@

$(PARFUMMAIN): main.C $(HEADDEP)
	$(CHARMC) main.C -o $(PARFUMMAIN)

# Copy header files to include directory
headers: $(HEADDEP)
	cp $(HEADERS) $(CDIR)/include/
	cp vector2d.h $(CDIR)/include/
	cp cktimer.h $(CDIR)/include/
	touch headers

# clean up
clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) $(COMPATLIB)
