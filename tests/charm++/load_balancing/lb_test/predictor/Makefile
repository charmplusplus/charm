CHARMC=../../../../../bin/charmc $(OPTS)

test:	test.o
	$(CHARMC) test.o -o test -language charm++ -module RefineLB

test.o:	test.C test.h test.def.h test.decl.h
	$(CHARMC) -c test.C

test.def.h test.decl.h: test.ci
	$(CHARMC) test.ci

clean:
	rm -f charmrun test test.def.h test.decl.h test.o

new:	clean test
