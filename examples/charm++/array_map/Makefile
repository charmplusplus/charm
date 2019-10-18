-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: array_map

array_map: array_map.C array_map.decl.h
	$(CHARMC) -language charm++ array_map.C -o array_map

array_map.decl.h: array_map.ci
	$(CHARMC) array_map.ci

clean:
	rm -f *.decl.h *.def.h *.o conv-host array_map charmrun

test: all
	$(call run, ./array_map +p4)