CHARMC=../../../bin/charmc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c -language ampi $(OPTS)

test: hello
	rm -rf log
	./charmrun +p3 ./hello +vp8 $(TESTOPTS)
	-sync
	./charmrun +p3 ./hello +vp8 +restart log $(TESTOPTS)
	./charmrun +p2 ./hello +vp8 +restart log $(TESTOPTS)

bgtest: clean ampi
	./charmrun +p3 ./hello +vp8 +x3 +y1 +z1
	./charmrun +p3 ./hello +vp8 +restart log +x3 +y1 +z1
	./charmrun +p2 ./hello +vp8 +restart log +x2 +y1 +z1

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun
	rm -rf log
