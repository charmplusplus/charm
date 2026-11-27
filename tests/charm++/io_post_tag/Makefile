-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: iopost

iopost: iopost.ci iopost.C
	$(CHARMC) iopost.ci
	$(CHARMC) iopost.C -o $@ -module CkIO

test: iopost
	$(call run, ./iopost +p1)
	$(call run, ./iopost +p2)

smptest: iopost
	$(call run, ./iopost +p2)

clean:
	rm -f *.o *.decl.h *.def.h iopost iopost.dat charmrun
