CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

LIB = libirecv.a
LIBOBJ = receiver.o

HEADERS = $(CDIR)/include/receiver.decl.h \
          $(CDIR)/include/receiver.h $(CDIR)/include/receiverf.h
LIBDEST =  $(CDIR)/lib/libirecv.a

all: $(LIBDEST) $(HEADERS)

$(HEADERS): receiver.decl.h receiver.h receiverf.h
	cp receiver.decl.h $(CDIR)/include
	cp receiver.h $(CDIR)/include
	cp receiverf.h $(CDIR)/include

$(LIBDEST) : $(LIB)
	cp $(LIB) $(LIBDEST)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

receiver.o: receiver.C receiver.h receiver.decl.h receiver.def.h
	$(CHARMC) -c receiver.C

main.decl.h: main.ci
	$(CHARMC) -c main.ci

receiver.decl.h receiver.def.h : receiver.ci
	$(CHARMC) -c receiver.ci

clean:
	rm -f conv-host *.o *.decl.h *.def.h core 
