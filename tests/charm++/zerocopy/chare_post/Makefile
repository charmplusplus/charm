-include ../../../common.mk
-include ../../../../include/conv-mach-opt.mak
CHARMC = ../../../../bin/charmc $(OPTS)

all: chare_post

chare_post.decl.h chare_post.def.h: chare_post.ci
	$(CHARMC) chare_post.ci

chare_post: chare_post.C chare_post.decl.h chare_post.def.h
	$(CHARMC) -language charm++ chare_post.C -o chare_post

test: all
	$(call run, ./chare_post +p1)
	$(call run, ./chare_post +p2)

testp: all
	$(call run, ./chare_post +p$(P))

smptest: all
	$(call run, ./chare_post +p2 ++ppn 2)

clean:
	rm -f *.decl.h *.def.h *.o chare_post charmrun
