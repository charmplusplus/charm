CDIR=../../../..
AMPICC=$(CDIR)/bin/ampicxx $(OPTS)
-include $(CDIR)/include/conv-mach-opt.mak

HEADERS=idxlc.h idxl.h idxl_layout.h idxl_comm.h
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  HEADERS += idxlf.h
endif
HEADDEP=$(HEADERS)
OBJS=idxl.o idxl_layout.o idxl_comm.o idxl_api.o
DEST=$(CDIR)/lib/libidxl.a

all: $(DEST)

$(DEST): $(OBJS) $(COMPAT) headers
	$(AMPICC) $(OBJS) $(COMPAT) -o $@

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

idxl.o: idxl.C $(HEADDEP)
	$(AMPICC) -c idxl.C

idxl_layout.o: idxl_layout.C $(HEADDEP)
	$(AMPICC) -c idxl_layout.C

idxl_comm.o: idxl_comm.C $(HEADDEP)
	$(AMPICC) -c idxl_comm.C

idxl_api.o: idxl_api.C $(HEADDEP)
	$(AMPICC) -c idxl_api.C

clean:
	rm -rf *.a *.def.h *.decl.h *.o SunWS_cache $(DEST) headers
