-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: randomttl

randomttl: randomttl.o
	$(CHARMC) -language converse++ -o randomttl randomttl.o

randomttl.o: randomttl.C
	$(CHARMC) -language converse++ -c randomttl.C

test: randomttl
	$(call run, ./randomttl +p4 2 1000)
	$(call run, ./randomttl +p6 5 1000)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f randomttl
	rm -f conv-host charmrun
