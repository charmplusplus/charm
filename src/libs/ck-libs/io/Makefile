CDIR=../../../..
-include $(CDIR)/include/conv-mach-opt.mak
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=CkIO
LIB = $(CDIR)/lib/libmodule$(MODULE).a
DEP = $(CDIR)/lib/libmodule$(MODULE).dep
LIBOBJ = ckio.o fs_parameters.o

GENHEADERS = $(MODULE).decl.h $(MODULE).def.h $(MODULE)_impl.decl.h $(MODULE)_impl.def.h
HEADERS = ckio.h $(GENHEADERS)

all: $(LIBDEST)$(LIB)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ)
	echo "$(CMK_LUSTREAPI)" > $(DEP)

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

ckio.o: ckio.C headers
	$(CHARMC) -c $<

fs_parameters.o: fs_parameters.C
	$(CHARMC) -c $<

$(GENHEADERS): ckio.ci.stamp
%.ci.stamp: %.ci
	$(CHARMC) -c $<
	touch $@

clean:
	rm -f *.o *.decl.h *.def.h $(LIB) headers *.stamp
