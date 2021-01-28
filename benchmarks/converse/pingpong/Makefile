-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pingpong pingpong_multipairs

pingpong: pingpong.o
	$(CHARMC) -language converse++ -o pingpong pingpong.o

pingpong.o: pingpong.C
	$(CHARMC) -language converse++ -c pingpong.C

pingpong_multipairs: pingpong_multipairs.o
	$(CHARMC) -language converse++ -o pingpong_multipairs pingpong_multipairs.o

pingpong_multipairs.o: pingpong_multipairs.C
	$(CHARMC) -language converse++ -c pingpong_multipairs.C

test: pingpong pingpong_multipairs
	$(call run, ./pingpong +p2 )
	$(call run, ./pingpong_multipairs +p2 )
 
testp: pingpong pingpong_multipairs
	$(call run, ./pingpong_multipairs +p$(P))

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f pingpong pingpong_multipairs
	rm -f conv-host charmrun
