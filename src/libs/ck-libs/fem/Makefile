CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=fem.h femf.h
HEADDEP=$(HEADERS) fem_impl.h fem.decl.h fem.def.h
OBJS=fem.o \
	symmetries.o datatype.o \
	partition.o map.o call_init.o
COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o 
DEST=$(CDIR)/lib/libmodulefem.a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

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

.C.o:
	$(CHARMC) -c $<

fem.o: $(HEADDEP)
partition.o: $(HEADDEP)
map.o: $(HEADDEP)
symmetries.o: $(HEADDEP)
datatype.o: $(HEADDEP)

fem.decl.h fem.def.h: fem.ci
	$(CHARMC) fem.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
