CHARMC=../../../bin/charmc $(OPTS)

pingpong: pingpong.C
	$(CHARMC) -language converse++ -o pingpong pingpong.C

test: pingpong
	./charmrun pingpong +p2

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f pingpong
	rm -f conv-host charmrun
