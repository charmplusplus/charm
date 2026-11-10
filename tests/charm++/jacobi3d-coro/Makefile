-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) $(MOPTS)

OBJS = jacobi3d.o

all: jacobi3d

jacobi3d: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -o jacobi3d $(OBJS)

jacobi3d.decl.h: jacobi3d.ci
	$(CHARMC) jacobi3d.ci

jacobi3d.o: jacobi3d.C jacobi3d.decl.h
	$(CHARMC) -std=c++20 -c jacobi3d.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi3d charmrun *~
	rm -f ref.log p4.log lb.log fake.log chkpt.log restart*.log
	rm -rf log

# The final line "jacobi3d: N iterations, checksum ..." must be the same in
# every run: the iteration count and the arithmetic do not depend on the PE
# count, on migration or on a restart.
define check_same
	grep "^jacobi3d: " $(1) && test "$$(grep '^jacobi3d: ' $(1))" = "$$(grep '^jacobi3d: ' $(2))"
endef

# 32^3 array, 16^3 blocks (8 elements), 20 iterations, a report every 10,
# load balancing every 4 (where given), a checkpoint after iteration 15.
test: all
	$(call run, ./jacobi3d 32 16 20 10 +p1 ) 2>&1 | tee ref.log
	grep "^jacobi3d: 20 iterations" ref.log
	$(call run, ./jacobi3d 32 16 20 10 +p4 ) 2>&1 | tee p4.log
	$(call check_same,p4.log,ref.log)
	$(call run, ./jacobi3d 32 16 20 10 4 +p4 +balancer RotateLB ) 2>&1 | tee lb.log
	$(call check_same,lb.log,ref.log)
	$(call run, ./jacobi3d 32 16 20 10 4 +p2 +ck_fakemigrate ) 2>&1 | tee fake.log
	$(call check_same,fake.log,ref.log)
	rm -rf log
	$(call run, ./jacobi3d 32 16 20 10 4 -chkpt 15 log +p4 +balancer RotateLB ) 2>&1 | tee chkpt.log
	$(call check_same,chkpt.log,ref.log)
	-sync
	$(call run, ./jacobi3d 32 16 20 10 4 -chkpt 15 log +p4 +balancer RotateLB +restart log ) 2>&1 | tee restart4.log
	$(call check_same,restart4.log,ref.log)
	$(call run, ./jacobi3d 32 16 20 10 4 -chkpt 15 log +p2 +balancer RotateLB +restart log ) 2>&1 | tee restart2.log
	$(call check_same,restart2.log,ref.log)

.PHONY: all clean test
