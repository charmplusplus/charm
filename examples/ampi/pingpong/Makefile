OPTS=-O3
CHARMC=../../../bin/charmc -language ampi $(OPTS) 

all : pingpongmpi.c
	$(CHARMC) pingpongmpi.c -o pgm

test: all
	./charmrun +p2 ./pgm 100000 16 +vp2

clean:
	rm -rf charmrun conv-host moduleinit* *.o pingpong *~ *.sts core

