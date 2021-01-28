
#override OPTS += -DUSING_CONTROLPOINTS
#override OPTS += -DUSEBITPRIORITY

###########  Don't change below this line ############

CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

# remove -DBRANCHBOUND so that it can build a non-branchbound library
# override CHARMC := $(subst -DBRANCHBOUND, , $(CHARMC))

HEADERS = searchEngine.h searchEngine_impl.h searchEngine.decl.h
OBJS = searchEngine.o

LIB1 = libmodulesearchEngine.a
LIB2 = libmodulesearchEngine_bound.a

all: $(CDIR)/lib/$(LIB1)   $(CDIR)/lib/$(LIB2)
	/bin/cp $(HEADERS) $(CDIR)/include

$(CDIR)/lib/$(LIB1):  $(OBJS)
	$(CHARMC) -o $@ $^

$(CDIR)/lib/$(LIB2): searchEngine_bound.o
	$(CHARMC) -o $@ $^
	/bin/cp searchEngine_bound.decl.h  $(CDIR)/include

searchEngine.o: $(HEADERS) searchEngine.C
	$(CHARMC) -c  searchEngine.C

searchEngine.decl.h: searchEngine.ci
	$(CHARMC) -E searchEngine.ci

searchEngine_bound.o: $(HEADERS) searchEngine.C searchEngine.ci
	sed -e 's/module searchEngine/module searchEngine_bound/' searchEngine.ci > searchEngine_bound.ci && $(CHARMC) -DBRANCHBOUND -E  searchEngine_bound.ci
	$(CHARMC) -DBRANCHBOUND -c  searchEngine.C -o searchEngine_bound.o

clean:
	rm -f *.decl.h *.def.h conv-host *.o main charmrun *~ searchEngine_bound.ci
