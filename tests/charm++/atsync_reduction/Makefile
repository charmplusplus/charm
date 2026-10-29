-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: redmig

redmig: redmig.decl.h redmig.C
	$(CHARMC) redmig.C -o redmig -module CommonLBs

redmig.decl.h: redmig.ci
	$(CHARMC) redmig.ci

# The -s runs create the arrays as TCharm and AMPI do (static insertion,
# anytime migration off), under which broadcasts are not stored for migrating
# elements beyond what every local element has received; -d keeps the promise
# those flags make by calling AtSync only after the step's results arrived, and
# -o checks that the results still arrive in reduction order. -s without -d
# hangs by design.
test: redmig
	$(call run, ./redmig +p2 -A)
	$(call run, ./redmig +p2 -s -d -o -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -s -d -o -b -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -s -d -o -k -b -n 4 -i 20 +balancer RandCentLB -M)
	$(call run, ./redmig +p4 -s -d -o -n 16 -r 3 -i 40 +balancer RandCentLB -M)
	$(call run, ./redmig +p1 -s -d -z -n 1 -i 5)
	$(call run, ./redmig +p2 -s -d -o -z -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p4 -s -d -o -z -n 4 -i 20 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -z -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -d -o -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -m +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -b -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -b +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -k -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -k -b -n 4 -i 5 +balancer RandCentLB -M)
	$(call run, ./redmig +p2 -r 5 -i 40 +balancer RandCentLB -M)
	$(call run, ./redmig +p4 -o -n 16 -r 3 -i 40 +balancer RandCentLB -M)

testp: redmig
	$(call run, ./redmig +p$(P) -n 16 -r 3 -i 40 +balancer RandCentLB -M)
	$(call run, ./redmig +p$(P) -s -d -o -n 16 -r 3 -i 40 +balancer RandCentLB -M)
	$(call run, ./redmig +p$(P) -s -d -o -z -n 16 -r 3 -i 20 +balancer RandCentLB -M)

smptest: redmig
	$(call run, ./redmig +p2 ++ppn 2 +balancer RandCentLB -M)
	$(call run, ./redmig +p4 ++ppn 2 -s -d -o -n 16 -r 3 -i 40 +balancer RandCentLB -M)

clean:
	rm -rf *.decl.h *.def.h redmig charmrun
