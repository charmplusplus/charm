-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pingpong

pingpong: pingpong.o
	$(CHARMC) -language converse++ -o pingpong pingpong.o

pingpong.o: pingpong.C
	$(CHARMC) -language converse++ -c pingpong.C

test: pingpong
	$(call run, ./pingpong +p2 )

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f pingpong
	rm -f conv-host charmrun
