CHARMBIN = ../../../bin

all: put

put: put.c
	$(CHARMBIN)/charmc -o put put.c -language armci -memory os

run: put
	./charmrun +p2 ./put

clean:
	rm -f put charmrun
