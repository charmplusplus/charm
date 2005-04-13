CHARMC=../../../../bin/charmc $(OPTS)

loadlog: loadlog.o
	$(CHARMC) -seq -o loadlog loadlog.o -lconv-bluegene-logs -lconv-util -lckstandalone -language c++

loadlog.o: loadlog.C 
	$(CHARMC) -c loadlog.C

clean:
	rm -f *.o loadlog *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

test: loadlog
	./loadlog
