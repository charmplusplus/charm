CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=mblock.h mblockf.h
HEADDEP=$(HEADERS) patch.h gridutil.h mblock_impl.h \
        mblock.decl.h mblock.def.h
OBJS=mblock.o patch.o
COMPAT=compat_init.o compat_finit.o compat_driver.o compat_fdriver.o
LIB=libmodulemblock

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

patch.o: patch.C patch.h
	$(CHARMC) -c patch.C

mblock.o: mblock.C mblock.decl.h $(HEADDEP)
	$(CHARMC) -c mblock.C

mblock.decl.h mblock.def.h: mblock.ci
	$(CHARMC) mblock.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out $(DEST) conv-host charmrun headers
