CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
-include $(CDIR)/include/conv-mach-opt.mak

LIB = libirecv.a
LIBOBJ = receiver.o

HEADERS = $(CDIR)/include/receiver.decl.h \
          $(CDIR)/include/receiver.h
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  HEADERS +=$(CDIR)/include/receiverf.h
endif
LIBDEST =  $(CDIR)/lib/libirecv.a

all: $(LIBDEST)
	cp receiver.decl.h $(CDIR)/include/
	cp receiver.h $(CDIR)/include/
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
	cp receiverf.h $(CDIR)/include/
endif

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $@ $^

receiver.o: receiver.C receiver.h receiver.decl.h receiver.def.h
	$(CHARMC) -c receiver.C

receiver.def.h: receiver.decl.h

receiver.decl.h: receiver.ci
	$(CHARMC) -c receiver.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core 
