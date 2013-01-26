CHARMC=../../../../bin/charmc $(OPTS)
ALL=caseTest

.PHONY: all clean

all: $(ALL)

clean:
	-rm -rf *.o *~ core *.decl.h *.def.h *.txt charmrun $(ALL)

caseTest.decl.h: caseTest.ci
	$(CHARMC) $(OPTS) caseTest.ci

caseTest.o: caseTest.cc caseTest.decl.h
	$(CHARMC) $(OPTS) -c $<

caseTest: caseTest.o
	$(CHARMC) -language charm++ $(OPTS) $^ -o $@

