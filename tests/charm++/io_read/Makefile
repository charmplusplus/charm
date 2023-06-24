-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: iotest

iotest: iotest.ci iotest.C
	$(CHARMC) iotest.ci
	$(CHARMC) iotest.C -o $@ -module CkIO

test: iotest
	# dd if=/dev/urandom of=large_test.txt bs=64M count=16 iflag=fullblock
	$(call run, ./iotest +p2 4)
	$(call run, ./iotest +p4 4)

testp: iotest
	$(call run, ./iotest +p$(P) $(P))
	$(call run, ./iotest +p$(P) $$(( $(P) * 2 )))

smptest: iotest
	$(call run, ./iotest +p2 ++ppn 2 4)
	$(call run, ./iotest +p4 ++ppn 2 4)

clean:
	rm -f *.o *.decl.h *.def.h iotest test*

