
include ../Makefile.common


all: main main_nb main_ws 

projections: main.prj main_nb.prj main_ws.prj

main: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main -module searchEngine

main.prj:  main.o 
	$(CHARMC) -language charm++  -tracemode projections main.o searchEngineAPI.o -o main.prj -module searchEngine

main_nb: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main_nb -balance neighbor -module searchEngine

main_nb.prj:  main.o 
	$(CHARMC) -language charm++  -tracemode projections main.o searchEngineAPI.o -o main_nb.prj -balance neighbor -module searchEngine

main_ws: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main_ws -balance workstealing -module searchEngine

main_ws.prj: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main_ws.prj -balance workstealing -module searchEngine -tracemode projections

main.decl.h: main.ci
	$(CHARMC)  main.ci

main.o: main.C main.decl.h
	$(CHARMC) -c  main.C

searchEngineAPI.o: searchEngineAPI.C  
	$(CHARMC) -c  searchEngineAPI.C


clean:
	rm -f main main_nb main_ws  *.decl.h *.def.h conv-host *.o charmrun $(all) $(projections) *~
