OPTS=-g
CHARMC=../../../../bin/charmc $(OPTS)

all: hello

hello:   hello.o
	$(CHARMC) hello.o -o hello -module CkMulticast -language charm++ #-memory paranoid

hello.o : hello.C hello.def.h hello.decl.h 
	$(CHARMC) -c hello.C

hello.decl.h hello.def.h : hello.ci
	$(CHARMC) hello.ci

clean:
	rm -f conv-host *.o charmrun
	rm -f *.def.h *.decl.h
	rm -f pgm pgm.*.log pgm.sts
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers hello

