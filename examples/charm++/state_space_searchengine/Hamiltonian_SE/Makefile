
override OPTS += -DVALUE_HEURISTIC #-DUSERSOLVER 

include ../Makefile.common


all: main main.nb

projections: main.nb.prj main.prj

main: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main -module searchEngine

main.prj:  main.o 
	$(CHARMC) -language charm++  -tracemode projections main.o searchEngineAPI.o -o main.prj -module searchEngine

main.nb: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main.nb -balance neighbor -module searchEngine

main.nb.prj:  main.o 
	$(CHARMC) -language charm++  -tracemode projections main.o searchEngineAPI.o -o main.nb.prj -balance neighbor -module searchEngine

main.decl.h: main.ci
	$(CHARMC)  main.ci

main.o: main.C main.decl.h
	$(CHARMC) -c  main.C

searchEngineAPI.o: searchEngineAPI.C  
	$(CHARMC) -c  searchEngineAPI.C


clean:
	rm -f *.decl.h *.def.h conv-host *.o main main.nb main.nb.prj main.prj charmrun $(ALL) *~
