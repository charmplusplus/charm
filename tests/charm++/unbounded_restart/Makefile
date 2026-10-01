-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)
# lcrun of a reconverse build (testprocs is reconverse-only).
LCRUN ?= ../../../reconverse-darwin-arm8/_deps/lci-src/lcrun

all: restart
	rm -rf ckptlog/

restart: restart.decl.h restart.C
	$(CHARMC) restart.C -o restart

restart.decl.h: restart.ci
	$(CHARMC) restart.ci

# Restart with the same and with a different number of PEs.
test: all
	-rm -rf ckptlog
	$(call run, ./restart +p4 )
	-sync
	$(call run, ./restart +p4 +restart ckptlog )
	$(call run, ./restart +p2 +restart ckptlog )
	$(call run, ./restart +p3 +restart ckptlog )
	-rm -rf ckptlog

smptest: test

# Reconverse only: checkpoint from 2 processes x 1 PE, restart on 1 process x 2 PEs
# and on 2 processes x 1 PE. (Not 2 processes x 2 PEs: a reconverse restart hang.)
testprocs: all
	-rm -rf ckptlog
	$(LCRUN) -n 2 ./restart +pe 2
	./restart +pe 2 +restart ckptlog
	$(LCRUN) -n 2 ./restart +pe 2 +restart ckptlog
	-rm -rf ckptlog

clean:
	rm -f *.decl.h *.def.h *.o restart charmrun
	rm -rf ckptlog/
