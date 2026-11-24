-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: proxy_delctor

proxy_delctor.decl.h proxy_delctor.def.h: proxy_delctor.ci
	$(CHARMC) proxy_delctor.ci

proxy_delctor: proxy_delctor.C proxy_delctor.decl.h proxy_delctor.def.h
	$(CHARMC) -language charm++ proxy_delctor.C -o proxy_delctor

test: proxy_delctor
	$(call run, ./proxy_delctor +p1)
	$(call run, ./proxy_delctor +p2)
	$(call run, ./proxy_delctor +p4)

smptest: proxy_delctor
	$(call run, ./proxy_delctor +p4 ++ppn 4)

clean:
	rm -f *.decl.h *.def.h *.o proxy_delctor charmrun
