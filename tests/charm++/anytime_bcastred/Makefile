-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: bcastred

bcastred: bcastred.decl.h bcastred.C
	$(CHARMC) bcastred.C -o bcastred

bcastred.decl.h: bcastred.ci
	$(CHARMC) bcastred.ci

test: bcastred
	$(call run, +p2 ./bcastred)
	$(call run, +p2 ./bcastred -n 8 -i 200 -m 1 -c 8 )

smptest: bcastred
	$(call run, ./bcastred +p4 ++ppn 2)
	$(call run, ./bcastred +p4 ++ppn 2 -n 8 -i 200 -m 1 -c 8 )

clean:
	rm -rf *.decl.h *.def.h bcastred charmrun
