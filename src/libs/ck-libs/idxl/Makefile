CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

HEADERS=idxlc.h idxlf.h idxl.h idxl_layout.h idxl_comm.h
HEADDEP=$(HEADERS) idxl.decl.h idxl.def.h
OBJS=idxl.o idxl_layout.o idxl_comm.o idxl_api.o 
DEST=$(CDIR)/lib/libmoduleidxl.a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(CHARMC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

idxl.o: idxl.C $(HEADDEP)
	$(CHARMC) -c idxl.C

idxl_layout.o: idxl_layout.C $(HEADDEP)
	$(CHARMC) -c idxl_layout.C

idxl_comm.o: idxl_comm.C $(HEADDEP)
	$(CHARMC) -c idxl_comm.C

idxl_api.o: idxl_api.C $(HEADDEP)
	$(CHARMC) -c idxl_api.C

idxl.decl.h idxl.def.h: idxl.ci
	$(CHARMC) idxl.ci

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
