-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: kNeighbors

kNeighbors: kNeighbors.o
	$(CHARMC) -language converse++ -o kNeighbors kNeighbors.o

kNeighbors.o: kNeighbors.C
	$(CHARMC) -language converse++ -c kNeighbors.C

test: kNeighbors
	$(call run, ./kNeighbors +p4 2 100 100)
	$(call run, ./kNeighbors +p6 5 100 100)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f kNeighbors
	rm -f conv-host charmrun
