CHARMC=../../../bin/charmc $(OPTS)

all: pingpong

pingpong: pingpong.o
	$(CHARMC) -language converse++ -pg -o pingpong pingpong.o

pingpong.o: pingpong.C
	$(CHARMC) -language converse++ -pg -c pingpong.C

test: pingpong
	./charmrun ./pingpong +p2 $(TESTOPTS)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f pingpong
	rm -f conv-host charmrun
