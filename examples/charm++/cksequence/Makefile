-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = cksequence_test.o

all: cksequence_test

cksequence_test: $(OBJS)
	$(CHARMC) -language charm++ -o cksequence_test $(OBJS)

cksequence_test.decl.h: cksequence_test.ci
	$(CHARMC)  cksequence_test.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o cksequence_test charmrun

cksequence_test.o: cksequence_test.C cksequence_test.decl.h
	$(CHARMC) -c cksequence_test.C

test: all
	$(call run, ./cksequence_test +p4 10 )
