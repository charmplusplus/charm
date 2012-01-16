OPTS =  -DBRANCHBOUND #-DSTATISTIC#  -DOPTIMIZE #-DUSEPRIORITY -DOPTIMIZE #-DADPATIVE

include ../Makefile.common


#all: main main_prj
all: main

main: searchEngineAPI.o main.o 
	$(CHARMC) -language charm++  main.o searchEngineAPI.o -o main -module searchEngine_bound

main_prj:  main.o 
	$(CHARMC) -language charm++  -tracemode projections main.o searchEngine.o -o main_prj

main.decl.h: main.ci
	$(CHARMC)  main.ci

main.o: main.C main.decl.h
	$(CHARMC) -c  main.C

searchEngineAPI.o: searchEngineAPI.C  
	$(CHARMC) -c  searchEngineAPI.C


clean:
	rm -f *.decl.h *.def.h conv-host *.o main charmrun *~
