CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=fem.h femf.h
HEADDEP=$(HEADERS) fem_impl.h fem.decl.h fem.def.h
OBJS=fem.o partition.o map.o symmetries.o call_init.o
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

fem.o: fem.C $(HEADDEP)
	$(CHARMC) -c fem.C

partition.o: partition.C $(HEADDEP)
	$(CHARMC) -c partition.C

map.o: map.C $(HEADDEP)
	$(CHARMC) -c map.C

symmetries.o: symmetries.C $(HEADDEP)
	$(CHARMC) -c symmetries.C

fem.decl.h fem.def.h: fem.ci
	$(CHARMC) fem.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
