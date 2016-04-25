-include ../../common.mk
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
	$(call run, ./hello +p4 )
	-sync
	$(call run, ./hello +p4 +restart log )
	$(call run, ./hello +p2 +restart log )
	-sync
	-rm -fr log
	$(call run, ./hello +p2 )
	-sync
	$(call run, ./hello +p4 +restart log )

bgtest: all
	-rm -fr log
	$(call run, ./hello +p2 +x3 +y1 +z1 )
	$(call run, ./hello +p2 +restart log  +x3 +y1 +z1 )
	$(call run, ./hello +p2 +restart log  +x2 +y2 +z1 )
