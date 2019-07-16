-include ../../../../../common.mk
CHARMC= ../../../../../../bin/charmc $(OPTS)

all: simpleZCVec

simpleZCVec:  simpleZCVec.o
	$(CHARMC) simpleZCVec.o -o simpleZCVec -language charm++ -module CommonLBs

cifiles: simpleZCVec.ci
	$(CHARMC) -c simpleZCVec.ci
	touch cifiles

simpleZCVec.o : simpleZCVec.C cifiles
	$(CHARMC) -c simpleZCVec.C

test: all
	$(call run, +p4 ./simpleZCVec 32)
	$(call run, +p4 ./simpleZCVec 32 +noCMAForZC)

clean:
	rm -f *.def.h *.decl.h *.o *~ *.exe cifiles charmrun simpleZCVec
