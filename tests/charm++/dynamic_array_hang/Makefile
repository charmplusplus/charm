-include ../../../common.mk
CHARMC	= ../../../bin/charmc $(OPTS)

all: dynamic_array_hang

dynamic_array_hang: dynamic_array_hang.decl.h dynamic_array_hang.C
	$(CHARMC) dynamic_array_hang.C -o dynamic_array_hang 

dynamic_array_hang.decl.h: dynamic_array_hang.ci
	$(CHARMC) dynamic_array_hang.ci

test: dynamic_array_hang
	$(call run, +p3 ./dynamic_array_hang)

testp: dynamic_array_hang
	$(call run, +p$(P) ./dynamic_array_hang)

smptest: dynamic_array_hang
	$(call run, +p2 ./dynamic_array_hang)
	$(call run, +p3 ./dynamic_array_hang)
	$(call run, +p4 ./dynamic_array_hang)

clean:
	rm -rf *.decl.h *.def.h dynamic_array_hang charmrun
