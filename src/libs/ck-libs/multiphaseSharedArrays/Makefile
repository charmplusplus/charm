CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=msa
LIB = $(CDIR)/lib/libmodule$(MODULE).a
LIBOBJ = msa-distArray.o

HEADERS = msa.h msa-common.h msa-distArray.h msa-DistPageMgr.h $(MODULE).decl.h \
		$(MODULE).def.h

all: $(LIBDEST)$(LIB)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

headers: $(HEADERS)
	test -d $(CDIR)/include/msa || mkdir $(CDIR)/include/msa/
	cp $(HEADERS) $(CDIR)/include/msa/
	touch headers

msa-distArray.o: msa-distArray.C msa-DistPageMgr.ci headers
	$(CHARMC) -c msa-distArray.C

$(MODULE).def.h: $(MODULE).decl.h

$(MODULE).decl.h: msa-DistPageMgr.ci
	$(CHARMC) -c msa-DistPageMgr.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core $(LIB) headers

c: clean
