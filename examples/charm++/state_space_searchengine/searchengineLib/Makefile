
include ../Makefile.common

# remove -DBRANCHBOUND so that it can build a non-branchbound library
override CHARMC := $(subst -DBRANCHBOUND, , $(CHARMC))


OBJS = searchEngine.o

LIB1 = libmodulesearchEngine.a

LIB2 = libmodulesearchEngine_bound.a

#all: main main_prj
all: lib

main: searchEngine.o main.o 
	$(CHARMC) -language charm++  main.o searchEngine.o -o main

main_prj: searchEngine.o main.o 
	$(CHARMC) -language charm++  -tracemode projections main.o searchEngine.o -o main_prj

main.decl.h: main.ci
	$(CHARMC) main.ci

main.o: main.C main.decl.h
	$(CHARMC) -c  main.C

searchEngine.o: searchEngine_impl.h searchEngine.h searchEngine.C searchEngine.decl.h
	$(CHARMC) -c  searchEngine.C

searchEngine.decl.h: searchEngine.ci
	$(CHARMC) -E searchEngine.ci

lib: $(LIB1) $(LIB2)

$(LIB1):  $(OBJS)
	$(CHARMC) -o $(LIB1) $(OBJS)

$(LIB2):  searchEngine_bound.o
	$(CHARMC) -o $(LIB2)  searchEngine_bound.o

searchEngine_bound.o: searchEngine_impl.h searchEngine.h searchEngine.C searchEngine.decl.h searchEngine.ci
	sed -e 's/module searchEngine/module searchEngine_bound/' searchEngine.ci > searchEngine_bound.ci && $(CHARMC) -DBRANCHBOUND -E  searchEngine_bound.ci
	$(CHARMC) -DBRANCHBOUND -c  searchEngine.C -o searchEngine_bound.o

clean:
	rm -f *.decl.h *.def.h conv-host *.o main charmrun *~ $(LIB1) $(LIB2) searchEngine_bound.ci
