-include ../../../common.mk
CHARMDIR = ../../../..
CHARMC = $(CHARMDIR)/bin/charmc $(OPTS)

all: direct_api

OBJS = direct_api.o

direct_api: $(OBJS)
	$(CHARMC) -language charm++ -o direct_api $(OBJS)

cifiles: direct_api.ci
	$(CHARMC)  direct_api.ci
	touch cifiles

direct_api.o: direct_api.C cifiles
	$(CHARMC) -c direct_api.C

test: all
	$(call run, +p1 ./direct_api 60)
	$(call run, +p4 ./direct_api 100)
	$(call run, +p4 ./direct_api 100 +noCMAForZC)

clean:
	rm -f *.decl.h *.def.h *.o
	rm -f direct_api charmrun cifiles
