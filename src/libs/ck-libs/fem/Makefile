CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=fem.h femf.h
HEADDEP=$(HEADERS) fem_impl.h fem_mesh.h fem.decl.h fem.def.h
OBJS=fem.o fem_mesh.o symmetries.o \
	partition.o map.o fem_compat.o call_init.o
COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
LIB=libmodulefem

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

$(COMPAT):
	@for o in $(COMPAT); \
	do \
	        file=`echo $$o | sed -e "s/\.o/.c/g"`; \
	        echo "$(CHARMC) -c $$file..."; \
	        $(CHARMC) -c $$file || exit 1; \
	done

fem.o: fem.C $(HEADDEP)
	$(CHARMC) -c fem.C

partition.o: partition.C $(HEADDEP)
	$(CHARMC) -c partition.C

map.o: map.C $(HEADDEP)
	$(CHARMC) -c map.C

symmetries.o: symmetries.C $(HEADDEP)
	$(CHARMC) -c symmetries.C

datatype.o: datatype.C $(HEADDEP)
	$(CHARMC) -c datatype.C

fem_compat.o: fem_compat.C $(HEADDEP)
	$(CHARMC) -c fem_compat.C

fem_mesh.o: fem_mesh.C $(HEADDEP)
	$(CHARMC) -c fem_mesh.C

fem_comm.o: fem_comm.C $(HEADDEP)
	$(CHARMC) -c fem_comm.C

fem.decl.h fem.def.h: fem.ci
	$(CHARMC) fem.ci

call_init.o: call_init.c
	$(CHARMC) -c call_init.c

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
