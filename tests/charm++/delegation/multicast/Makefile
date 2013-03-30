CHARMC=../../../../bin/charmc $(OPTS)

all: hello

hello:   hello.o
	$(CHARMC) hello.o -o hello -module CkMulticast -language charm++ 

hello.o : hello.C hello.def.h hello.decl.h 
	$(CHARMC) -c hello.C

hello.decl.h hello.def.h : hello.ci.stamp

hello.ci.stamp: hello.ci
	$(CHARMC) $<
	touch $@

test: all
	./charmrun +p3 ./hello 10 $(TESTOPTS)

bgtest: all
	./charmrun +p3 ./hello 10 +x1 +y1 +z3 $(TESTOPTS)

clean:
	rm -f conv-host *.o charmrun charmrun.exe
	rm -f *.def.h *.decl.h
	rm -f hello hello.*.log hello.sts hello.exe hello.pdb hello.ilk
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers hello

