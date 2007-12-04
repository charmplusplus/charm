OPTS = -g
CHARMC=../../../../bin/charmc $(OPTS)

all: buildlog

buildlog: buildlog.o
	$(CHARMC) -seq -o buildlog buildlog.o -lconv-bigsim-logs -lconv-util -lblue-standalone -language c++

buildlog.o: buildlog.C 
	$(CHARMC) -seq -c buildlog.C

clean:
	rm -f *.o buildlog *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

test: buildlog
	./buildlog
