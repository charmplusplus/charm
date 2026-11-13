-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: neighbor

neighbor: neighbor.o
	$(CHARMC) -o neighbor neighbor.o

neighbor.o: neighbor.C
	$(CHARMC) -std=c++20 -c neighbor.C

clean:
	rm -f *.o neighbor charmrun conv-host ampirun

test: neighbor
	$(call run, ./neighbor +p1 +vp4)
	$(call run, ./neighbor +p2 +vp4)
	$(call run, ./neighbor +p2 +vp5)
	$(call run, ./neighbor +p2 +vp7)
