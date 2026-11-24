-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: calls

calls: calls.o
	$(CHARMC) -o calls calls.o

calls.o: calls.C
	$(CHARMC) -std=c++20 -c calls.C

clean:
	rm -f *.o calls charmrun conv-host ampirun

test: calls
	$(call run, ./calls +p1 +vp4)
	$(call run, ./calls +p2 +vp4)
	$(call run, ./calls +p2 +vp5)
