CHARMRUN=./charmrun
-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language armci

test-local: hello
	$(call run, ./hello +p2 )

bgtest: hello
	$(call run, +p1 ./hello +vp2 +x2 +y1 +z1 )

clean:
	rm -f hello charmrun *.o charmrun.exe hello.exe hello.pdb hello.ilk
