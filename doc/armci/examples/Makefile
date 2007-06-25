CHARMBIN = ../../../bin

hello: hello.c
	$(CHARMBIN)/charmc -o hello hello.c -language armci

run: hello
	./charmrun +p2 ./hello

clean:
	rm -f hello charmrun
