CHARMC=../../../bin/charmc $(OPTS)

pgm:  hello.o
	$(CHARMC) hello.o  -o pgm -language charm++

hello.o : hello.C hello.h hello.decl.h hello.def.h
	$(CHARMC) -c hello.C

hello.decl.h  hello.def.h:  hello.ci
	$(CHARMC) hello.ci

clean:
	rm -f *.o conv-host *.def.h *.decl.h *~ pgm charmrun
