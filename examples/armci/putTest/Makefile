CHARMC=../../../bin/charmc $(OPTS)

all: put

put: put.c
	$(CHARMC) -o put put.c -language armci 

test: put
	./charmrun +p2 ./put $(TESTOPTS)

bgtest: put
	./charmrun +p1 ./put +vp 2 +x2 +y1 +z1 $(TESTOPTS)

clean:
	rm -f put charmrun *.o charmrun.exe put.exe put.pdb put.ilk
