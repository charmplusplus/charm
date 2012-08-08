include ../../Makefile.in
CHARMC:=../../${CHARMC}

all: libmodulehello.a

libmodulehello.a: hello.o
	$(CHARMC) -o libmodulehello.a hello.o

hello.decl.h: hello.ci
	$(CHARMC) hello.ci

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.a *.o charmrun *~ 
