-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: insert_chareidx

insert_chareidx: insert_chareidx.decl.h insert_chareidx.C
	$(CHARMC) insert_chareidx.C -o insert_chareidx

insert_chareidx.decl.h: insert_chareidx.ci
	$(CHARMC) insert_chareidx.ci

test: insert_chareidx
	$(call run, +p1 ./insert_chareidx)
	$(call run, +p2 ./insert_chareidx)
	$(call run, +p4 ./insert_chareidx)

testp: insert_chareidx
	$(call run, +p$(P) ./insert_chareidx)

smptest: insert_chareidx
	$(call run, +p2 ./insert_chareidx)
	$(call run, +p4 ./insert_chareidx)

clean:
	rm -rf *.decl.h *.def.h insert_chareidx charmrun
