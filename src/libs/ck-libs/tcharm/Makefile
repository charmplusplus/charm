CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=tcharm_impl.h tcharm.h tcharmc.h tcharmf.h tcharm.decl.h
OBJS=tcharm.o
COMPAT=compat_uns.o compat_us.o compat_funs.o compat_fus.o
DEST=$(CDIR)/lib/libmoduletcharm.a

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
		echo "$(CHARMC) -c $$file"; \
		$(CHARMC) -c $$file || exit 1; \
	done

tcharm.o: tcharm.C $(HEADERS)
	$(CHARMC) $(CFLAGS) -c $<

tcharm.decl.h: tcharm.ci
	$(CHARMC) tcharm.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) conv-host charmrun
