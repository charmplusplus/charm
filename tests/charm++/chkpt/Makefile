CHARMC=../../../bin/charmc $(OPTS)

OBJS = hello.o

all: hello
	rm -rf log/

hello: $(OBJS)
	$(CHARMC) -language charm++ -o hello hello.o $(LIBS)

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

clean:
	rm -f *~ *.decl.h *.def.h conv-host *.o hello charmrun charmrun.exe hello.exe hello.pdb hello.ilk
	rm -rf log/

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

test: all
	-rm -fr log
	./charmrun ./hello +p3 $(TESTOPTS)
	-sync
	./charmrun ./hello +p3 +restart log $(TESTOPTS)
	./charmrun ./hello +p4 +restart log $(TESTOPTS)

bgtest: all
	-rm -fr log
	./charmrun ./hello +p2 +x3 +y1 +z1 $(TESTOPTS)
	./charmrun ./hello +p2 +restart log  +x3 +y1 +z1 $(TESTOPTS)
	./charmrun ./hello +p2 +restart log  +x2 +y2 +z1 $(TESTOPTS)
