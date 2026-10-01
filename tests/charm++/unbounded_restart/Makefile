-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: restart
	rm -rf ckptlog/

restart: restart.decl.h restart.C
	$(CHARMC) restart.C -o restart

restart.decl.h: restart.ci
	$(CHARMC) restart.ci

test: all
	-rm -rf ckptlog
	$(call run, ./restart +p4 )
	-sync
	$(call run, ./restart +p4 +restart ckptlog )
	$(call run, ./restart +p2 +restart ckptlog )
	-rm -rf ckptlog

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o restart charmrun
	rm -rf ckptlog/
