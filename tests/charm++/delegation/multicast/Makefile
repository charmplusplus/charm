CHARMC=../../../../bin/charmc $(OPTS)

hello:   hello.o
	$(CHARMC) hello.o -o hello -module CkMulticast -language charm++ 

hello.o : hello.C hello.def.h hello.decl.h 
	$(CHARMC) -c hello.C

hello.decl.h hello.def.h : hello.ci
	$(CHARMC) hello.ci

test: hello
	./charmrun +p2 ./hello 10

bgtest: hello
	./charmrun +p2 ./hello 10 +x1 +y1 +z2

clean:
	rm -f conv-host *.o charmrun
	rm -f *.def.h *.decl.h
	rm -f pgm pgm.*.log pgm.sts
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers hello

