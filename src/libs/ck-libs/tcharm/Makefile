CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=tcharm_impl.h tcharm.h tcharmc.h tcharmf.h tcharm.decl.h
OBJS=tcharm.o
COMPAT=compat_uns.o compat_us.o compat_funs.o compat_fus.o compat_regmm.o

LIBDIR=$(CDIR)/lib

LIB=libmoduletcharm
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libmoduletcharm-compat.a

all: $(DEST) $(COMPATLIB)

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

$(COMPATLIB): $(COMPAT) 
	$(CHARMC) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

$(COMPAT):
	@for o in $(COMPAT); \
	do \
		file=`echo $$o | sed -e "s/\.o/.c/g"`; \
		echo "$(CHARMC) -c $$file"; \
		$(CHARMC) -c $$file || exit 1; \
	done

tcharm.o: tcharm.C $(HEADERS)
	$(CHARMC) -c tcharm.C

tcharm.decl.h: tcharm.ci
	$(CHARMC) tcharm.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out headers $(DEST) $(COMPATLIB) conv-host charmrun
