OPTS = -g
CHARMC=../../../../bin/charmc $(OPTS)

all: text2log

text2log: text2log.o
	$(CHARMC) -seq -o text2log text2log.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++

text2log.o: text2log.C 
	$(CHARMC) -seq -c text2log.C

clean:
	rm -f *.o text2log *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

help: text2log
	./text2log -help

test: text2log
	./text2log 1 1 1 1 1 1
