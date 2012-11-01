include ../Makefile.in
CHARMC:=../${CHARMC}

all: libmodulempicollide.a

libmodulempicollide.a: mpicollide.o
	$(CHARMC) -o libmodulempicollide.a mpicollide.o

mpicollide.decl.h: mpicollide.ci
	$(CHARMC) mpicollide.ci

mpicollide.o: mpicollide.C mpicollide.decl.h
	$(CHARMC) -module collidecharm -c mpicollide.C

clean:
	rm -f *.decl.h *.def.h conv-host *.a *.o charmrun *~ 
