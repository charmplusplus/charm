-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: reduction_empty_kid

reduction_empty_kid.decl.h reduction_empty_kid.def.h: reduction_empty_kid.ci
	$(CHARMC) reduction_empty_kid.ci

reduction_empty_kid: reduction_empty_kid.C reduction_empty_kid.decl.h reduction_empty_kid.def.h
	$(CHARMC) -language charm++ reduction_empty_kid.C -o reduction_empty_kid

# PE 0 with kids 1 and 2, then with kids 1, 2 and 3 (also runs in two processes).
test: reduction_empty_kid
	$(call run, ./reduction_empty_kid +p3)
	$(call run, ./reduction_empty_kid +p4)

smptest: reduction_empty_kid
	$(call run, ./reduction_empty_kid +p4 ++ppn 4)

clean:
	rm -f *.decl.h *.def.h *.o reduction_empty_kid charmrun
