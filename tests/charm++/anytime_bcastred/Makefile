-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: bcastred

bcastred: bcastred.decl.h bcastred.C
	$(CHARMC) bcastred.C -o bcastred

bcastred.decl.h: bcastred.ci
	$(CHARMC) bcastred.ci

test: bcastred
	$(call run, +p2 ./bcastred)

clean:
	rm -rf *.decl.h *.def.h bcastred charmrun
