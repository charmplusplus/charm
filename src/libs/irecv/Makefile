CHARMC=../../../bin/charmc $(OPTS)

LIB = libirecv.a
LIBOBJ = receiver.o

HEADERS = ../../../include/receiver.decl.h ../../../include/receiver.def.h \
          ../../../include/receiver.h ../../../include/receiverf.h
LIBDEST =  ../../../lib/libirecv.a

all: $(LIBDEST) $(HEADERS)

$(HEADERS): receiver.decl.h receiver.h receiverf.h
	/bin/cp receiver.decl.h ../../../include/receiver.decl.h
	/bin/cp receiver.def.h ../../../include/receiver.def.h
	/bin/cp receiver.h ../../../include/receiver.h
	/bin/cp receiverf.h ../../../include/receiverf.h

$(LIBDEST) : $(LIB)
	/bin/cp $(LIB) $(LIBDEST)

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

receiver.o: receiver.C receiver.h receiver.decl.h
	$(CHARMC) -c receiver.C

main.decl.h: main.ci
	$(CHARMC) -c main.ci

receiver.decl.h: receiver.ci
	$(CHARMC) -c receiver.ci

clean:
	/bin/rm -f conv-host *.o *.decl.h *.def.h core 
