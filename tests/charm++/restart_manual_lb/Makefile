-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: restart_manual_lb
	rm -rf log/

restart_manual_lb: restart_manual_lb.decl.h restart_manual_lb.C
	$(CHARMC) -language charm++ restart_manual_lb.C -o restart_manual_lb -module CommonLBs

restart_manual_lb.decl.h: restart_manual_lb.ci
	$(CHARMC) restart_manual_lb.ci

# Each pair checkpoints to log/ after step 4 and exits, then restarts from log/
# on the same PEs and must reach step 8.
test: all
	-rm -rf log
	$(call run, ./restart_manual_lb +p1 )
	-sync
	$(call run, ./restart_manual_lb +p1 +restart log )
	-rm -rf log
	$(call run, ./restart_manual_lb +p4 )
	-sync
	$(call run, ./restart_manual_lb +p4 +restart log )
	-rm -rf log
	$(call run, ./restart_manual_lb +p4 +balancer RotateLB )
	-sync
	$(call run, ./restart_manual_lb +p4 +balancer RotateLB +restart log )
	-rm -rf log

testp: all
	-rm -rf log
	$(call run, ./restart_manual_lb +p$(P) )
	-sync
	$(call run, ./restart_manual_lb +p$(P) +restart log )
	-rm -rf log

# make test already covers SMP builds; see ckpt_restart.
smptest:
	@echo "restart_manual_lb: no separate smptest"

clean:
	rm -f *.decl.h *.def.h *.o restart_manual_lb charmrun
	rm -rf log/
