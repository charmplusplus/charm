-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = alignmentCheck.o

all: alignmentCheck

alignmentCheck: $(OBJS)
	$(CHARMC) -language charm++ -o alignmentCheck $(OBJS)

alignmentCheck.decl.h: alignmentCheck.ci
	$(CHARMC)  alignmentCheck.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o alignmentCheck charmrun *.log *.sum *.sts

alignmentCheck.o: alignmentCheck.C alignmentCheck.decl.h
	$(CHARMC) -c alignmentCheck.C

test: all
	$(call run, +p2 ./alignmentCheck )
