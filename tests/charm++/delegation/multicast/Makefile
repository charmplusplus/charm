OPTS=-g
CHARMC=../../../../bin/charmc $(OPTS)

hello:   hello.o
	$(CHARMC) hello.o -o hello -lckmulticast -language charm++ #-memory paranoid

hello.o : hello.C hello.def.h hello.decl.h #ckmulticast.h
	$(CHARMC) -c hello.C

ckmulticast.o : ckmulticast.C ckmulticast.h CkMulticast.def.h CkMulticast.decl.h
	$(CHARMC) -c ckmulticast.C

CkMulticast.def.h CkMulticast.decl.h : ckmulticast.ci
	$(CHARMC) ckmulticast.ci

hello.decl.h hello.def.h : hello.ci
	$(CHARMC) hello.ci

clean:
	rm -f conv-host *.o charmrun
	rm -f *.def.h *.decl.h
	rm -f pgm pgm.*.log pgm.sts
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers hello

