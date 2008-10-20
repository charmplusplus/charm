include ../common.mk

HEADERS=$(CDIR)/include/tcharm_impl.h $(CDIR)/include/tcharm.h \
        $(CDIR)/include/tcharmc.h $(CDIR)/include/tcharmf.h \
        $(CDIR)/include/tcharm.decl.h
OBJS=tcharm.o
COMPAT=compat_uns.o compat_us.o compat_funs.o compat_fus.o compat_regmm.o

LIB=libmoduletcharm
DEST=$(LIBDIR)/$(LIB).a
COMPATLIB=$(LIBDIR)/libtcharm-compat.a

OBJS2=tcharmmain.o
LIB2=libmoduletcharmmain
DEST2=$(LIBDIR)/$(LIB2).a

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CHARMC) -c $<

all: $(DEST) $(COMPATLIB) $(DEST2)

$(DEST): $(OBJS) $(HEADERS)
	$(CHARMC) $(OBJS) -o $@
	cp $(LIB).dep $(LIBDIR)/$(LIB).dep

$(DEST2): $(OBJS2) $(HEADERS)
	$(CHARMC) $(OBJS2) -o $@
	cp $(LIB2).dep $(LIBDIR)/$(LIB2).dep

$(COMPATLIB): $(COMPAT) 
	$(CHARMC) $(COMPAT) -o $@

headers: $(HEADERS)

compat_uns.o: compat_uns.c

compat_us.o: compat_us.c

compat_funs.o: compat_funs.c

compat_fus.o: compat_fus.c

compat_regmm.o: compat_regmm.c

tcharm.o: tcharm.C $(HEADERS)
	$(CHARMC) -c tcharm.C

tcharm.decl.h: tcharm.ci
	$(CHARMC) tcharm.ci

tcharmmain.o: tcharmmain.C tcharmmain.decl.h $(HEADERS)
	$(CHARMC) -c tcharmmain.C

tcharmmain.decl.h: tcharmmain.ci
	$(CHARMC) tcharmmain.ci

clean: 
	-rm -fr *.o *~ *.decl.h *.def.h gmon.out headers conv-host charmrun

realclean: clean
	rm -f $(HEADERS) $(DEST) $(DEST2) $(COMPATLIB)
