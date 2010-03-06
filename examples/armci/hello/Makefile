CHARMC=../../../bin/charmc $(OPTS)

all: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language armci

test: hello
	./charmrun +p2 ./hello $(TESTOPTS)

bgtest: hello
	./charmrun +p1 ./hello +vp2 +x2 +y1 +z1 $(TESTOPTS)

clean:
	rm -f hello charmrun *.o charmrun.exe hello.exe hello.pdb hello.ilk
