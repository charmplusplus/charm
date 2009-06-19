CHARMC=../../../../bin/charmc $(OPTS)

all: changelog

changelog: changelog.o
	$(CHARMC) -seq -o changelog changelog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++

changelog.o: changelog.C 
	$(CHARMC) -seq -c changelog.C

clean:
	rm -f *.o loadlog *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

test: loadlog
	./loadlog
