-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language armci

test: hello
	$(call run, +p2 ./hello )

clean:
	rm -f hello charmrun *.o charmrun.exe hello.exe hello.pdb hello.ilk
