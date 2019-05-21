-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: check1d check2d

check1d: check1d.o
	$(CHARMC) $(OPTS) -language charm++ -o check1d check1d.o

check1d.o: check1d.C check1d.decl.h
	$(CHARMC) $(OPTS) -language charm++ -o check1d.o check1d.C

check1d.decl.h: check1d.ci
	$(CHARMC) $(OPTS) -language charm++ check1d.ci

check2d: check2d.o
	$(CHARMC) $(OPTS) -language charm++ -o check2d check2d.o

check2d.o: check2d.C check2d.decl.h
	$(CHARMC) $(OPTS) -language charm++ -o check2d.o check2d.C

check2d.decl.h: check2d.ci
	$(CHARMC) $(OPTS) -language charm++ check2d.ci


test: check1d check2d
	$(call run, +p4 ./check1d 20 )
	$(call run, +p4 ./check2d 20 20)

clean:
	rm -f *.decl.h *.def.h *.o check1d check2d charmrun
