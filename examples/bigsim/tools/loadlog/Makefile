CHARMC=../../../../../bin/charmc $(OPTS)

loadlog: loadlog.o
	$(CHARMC) -o loadlog loadlog.o -lconv-bluegene-logs -language charm++
	
loadlog.o: loadlog.C loadlog.decl.h
	$(CHARMC) -c loadlog.C

loadlog.def.h: loadlog.decl.h

loadlog.decl.h:  loadlog.ci
	$(CHARMC) loadlog.ci

clean:
	rm -f *.o loadlog *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log
