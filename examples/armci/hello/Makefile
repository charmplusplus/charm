CHARMC=../../../bin/charmc $(OPTS)

all: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language armci

test: hello
	./charmrun +p2 ./hello

bgtest: hello
	./charmrun +p1 ./hello +vp2 +x2 +y1 +z1

clean:
	rm -f hello charmrun
