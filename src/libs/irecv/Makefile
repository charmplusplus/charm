CHARMC=../../../bin/charmc $(OPTS)

LIB = libirecv.a
LIBOBJ = receiver.o

all: 
	echo 'Nothing to do here right now'

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
