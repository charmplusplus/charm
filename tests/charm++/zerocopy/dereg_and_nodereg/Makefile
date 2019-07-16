-include ../../../common.mk
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: dereg_and_nodereg

OBJS = dereg_and_nodereg.o

dereg_and_nodereg: $(OBJS)
	$(CHARMC) -language charm++ -o dereg_and_nodereg $(OBJS)

cifiles: dereg_and_nodereg.ci
	$(CHARMC)  dereg_and_nodereg.ci
	touch cifiles

dereg_and_nodereg.o: dereg_and_nodereg.C cifiles
	$(CHARMC) -c dereg_and_nodereg.C

test: all
	$(call run, +p1 ./dereg_and_nodereg 10)
	$(call run, +p4 ./dereg_and_nodereg 50)
	$(call run, +p6 ./dereg_and_nodereg 100)
	$(call run, +p4 ./dereg_and_nodereg 50 +noCMAForZC)
	$(call run, +p6 ./dereg_and_nodereg 100 +noCMAForZC)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f dereg_and_nodereg charmrun cifiles
