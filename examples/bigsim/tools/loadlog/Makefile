CHARMC=../../../../bin/charmc $(OPTS)

all: loadlog

loadlog: loadlog.o
	$(CHARMC) -host -o loadlog loadlog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++

loadlog.o: loadlog.C 
	$(CHARMC) -host -c loadlog.C

clean:
	rm -f *.o loadlog *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

test: loadlog
	./loadlog
