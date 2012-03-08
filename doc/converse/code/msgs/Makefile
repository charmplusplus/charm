CHARMC=../../../../bin/charmc $(OPTS)

all: interNodeMsg 

interNodeMsg: interNodeMsg.o
	$(CHARMC) -language converse++ -o interNodeMsg interNodeMsg.o

interNodeMsg.o: interNodeMsg.C
	$(CHARMC) -language converse++ -c interNodeMsg.C


test: interNodeMsg
	./charmrun ./interNodeMsg +p2 $(TESTOPTS)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f interNodeMsg 
	rm -f conv-host charmrun
