CHARMC=../../../bin/charmc $(OPTS)

all: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language armci

test: hello
	./charmrun +p2 ./hello

clean:
	rm -f hello charmrun
