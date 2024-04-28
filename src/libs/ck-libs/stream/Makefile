CDIR=../../../..
-include $(CDIR)/include/conv-mach-opt.mak
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=CkStream
LIB = $(CDIR)/lib/libmodule$(MODULE).a
DEP = $(CDIR)/lib/libmodule$(MODULE).dep
LIBOBJ = stream.o 

GENHEADERS = $(MODULE).decl.h $(MODULE).def.h $(MODULE)_impl.decl.h $(MODULE)_impl.def.h
HEADERS = stream.h streamtoken.h stream_msg_counter.h $(GENHEADERS)

all: $(LIBDEST)$(LIB)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ)

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

stream.o: stream.C headers
	$(CHARMC) -c $<

$(GENHEADERS): stream.ci.stamp
%.ci.stamp: %.ci
	$(CHARMC) -c $<
	touch $@

clean:
	rm -f *.o *.decl.h *.def.h $(LIB) headers *.stamp
