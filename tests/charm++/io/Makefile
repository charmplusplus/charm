-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: iotest.ci iotest.C
	$(CHARMC) iotest.ci
	$(CHARMC) iotest.C -o $@ -module CkIO

test: pgm
	$(call run, ./pgm +p4 4 )

clean:
	rm -f *.o *.decl.h *.def.h pgm test*
