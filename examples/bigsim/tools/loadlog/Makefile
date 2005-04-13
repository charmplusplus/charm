CHARMC=../../../../bin/charmc $(OPTS)

loadlog: loadlog.o
	$(CHARMC) -o loadlog loadlog.o -lconv-bluegene-logs -language converse++

loadlog.o: loadlog.C 
	$(CHARMC) -c loadlog.C

clean:
	rm -f *.o loadlog *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

test: loadlog
	./charmrun +p1 ./loadlog
