-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: iotest

test: iotest

iotest: iotest.ci iotest.C
	$(CHARMC) iotest.ci
	$(CHARMC) iotest.C -o $@ -module CkIO

testp: iotest
	$(call run, ./iotest +p$(P) $(P) $(TESTOPTS))

smptest: iotest
	$(call run, ./iotest 4 +p2 ++ppn 2 $(TESTOPTS))
	$(call run, ./iotest 4 +p4 ++ppn 2 $(TESTOPTS) )

clean:
	rm -f *.o *.decl.h *.def.h iotest test*

