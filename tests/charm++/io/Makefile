-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: iotest.ci iotest.C
	$(CHARMC) iotest.ci
	$(CHARMC) iotest.C -o $@ -module CkIO

test: pgm
	$(call run, ./pgm +p4 4 )

smptest: pgm
	$(call run, ./pgm 4 +p2 ++ppn 2)
	$(call run, ./pgm 4 +p4 ++ppn 2)

clean:
	rm -f *.o *.decl.h *.def.h pgm test*
