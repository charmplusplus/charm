CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libirecv.a
LIBOBJ = receiver.o

HEADERS = $(CDIR)/include/receiver.decl.h \
          $(CDIR)/include/receiver.h $(CDIR)/include/receiverf.h
LIBDEST =  $(CDIR)/lib/libirecv.a

all: $(LIBDEST)
	/bin/cp receiver.decl.h $(CDIR)/include
	/bin/cp receiver.h $(CDIR)/include
	/bin/cp receiverf.h $(CDIR)/include

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $@ $^

receiver.o: receiver.C receiver.h receiver.decl.h receiver.def.h
	$(CHARMC) -c receiver.C

receiver.def.h: receiver.decl.h

receiver.decl.h: receiver.ci
	$(CHARMC) -c receiver.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core 
