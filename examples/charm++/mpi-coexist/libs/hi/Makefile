include ../../Makefile.in
CHARMC:=../../${CHARMC}

all: libmodulehi.a

libmodulehi.a: hi.o
	$(CHARMC) -o libmodulehi.a hi.o

hi.decl.h: hi.ci
	$(CHARMC) hi.ci

hi.o: hi.C hi.decl.h
	$(CHARMC) -c hi.C

clean:
	rm -f *.decl.h *.def.h conv-host *.a *.o charmrun *~ 
