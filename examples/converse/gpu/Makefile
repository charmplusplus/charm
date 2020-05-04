-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pingpong pingpong

pingpong: pingpong.o
	$(CHARMC) -language converse++ -o pingpong pingpong.o

pingpong.o: pingpong.C
	$(CHARMC) -language converse++ -c pingpong.C

clean:
	rm -f *.o pingpong charmrun
