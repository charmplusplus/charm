CHARMC=../../../bin/charmc

all: ampi

ampi: hello.c
	$(CHARMC) -o hello hello.c -language ampi $(OPTS)

test: clean ampi
	./charmrun +p3 ./hello +vp8
	./charmrun +p3 ./hello +vp8 +restart log
	./charmrun +p2 ./hello +vp8 +restart log

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun
	rm -rf log
