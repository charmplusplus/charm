-include ../../common.mk
CHARMC = ../../../bin/charmc $(OPTS)

all: heat1d

heat1d: heat1d.o
	$(CHARMC) -language charm++ -module RotateLB -o heat1d heat1d.o

heat1d.o: heat1d.C heat1d.decl.h heat1d.def.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 -c heat1d.C

heat1d.decl.h heat1d.def.h: heat1d.ci
	$(CHARMC) heat1d.ci

# The checksum line is the same on any number of PEs, with the elements
# migrating every 40 iterations under RotateLB, and when every wait rebuilds
# the coroutines in place (+ck_fakemigrate).
define check_same
	grep "^heat1d: 200 iterations" $(1) && test "$$(grep '^heat1d: 200 iterations' $(1))" = "$$(grep '^heat1d: 200 iterations' $(2))"
endef

test: all
	$(call run, ./heat1d 8 64 200 50 +p1 ) 2>&1 | tee ref.log
	grep "^heat1d: 200 iterations" ref.log
	$(call run, ./heat1d 8 64 200 50 +p4 ) 2>&1 | tee p4.log
	$(call check_same,p4.log,ref.log)
	$(call run, ./heat1d 8 64 200 50 40 +p4 +balancer RotateLB ) 2>&1 | tee lb.log
	$(call check_same,lb.log,ref.log)
	$(call run, ./heat1d 8 64 200 50 +p4 +ck_fakemigrate ) 2>&1 | tee fake.log
	$(call check_same,fake.log,ref.log)

clean:
	rm -f *.decl.h *.def.h *.o heat1d charmrun ref.log p4.log lb.log fake.log

.PHONY: all test clean
