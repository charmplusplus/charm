CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

MODULE=msa
LIB = libmodule$(MODULE).a
LIBOBJ = msa-distArray.o

HEADERS = msa.h msa-common.h msa-distArray.h msa-DistPageMgr.h $(MODULE).decl.h \
		$(MODULE).def.h
LIBDEST =  $(CDIR)/lib/

all: $(LIBDEST)

$(LIBDEST) : $(LIB)
	/bin/cp $(LIB) $(LIBDEST)

$(LIB): $(LIBOBJ)
	test -e $(CDIR)/include/msa || mkdir $(CDIR)/include/msa/
	/bin/cp $(HEADERS) $(CDIR)/include/msa/
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

msa-distArray.o: msa-distArray.C msa-DistPageMgr.ci $(HEADERS)
	$(CHARMC) -c msa-distArray.C

$(MODULE).def.h: $(MODULE).decl.h

$(MODULE).decl.h: msa-DistPageMgr.ci
	$(CHARMC) -c msa-DistPageMgr.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core $(LIB)

c: clean
