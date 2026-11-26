-include ../../common.mk
CHARMC         = ../../../bin/charmc

OPTS            = -O3

all: leanmd

leanmd: Main.o Cell.o Compute.o leanmd.decl.h
	$(CHARMC) $(OPTS) \
	-language charm++ -o leanmd Main.o Cell.o Compute.o

Main.o: Main.C Main.h leanmd.decl.h defs.h ../../../include/ckcoro.h
	$(CHARMC) $(OPTS) -std=c++20 -o $@ -c $<

Cell.o: Cell.C Cell.h leanmd.decl.h defs.h ../../../include/ckcoro.h
	$(CHARMC) $(OPTS) -std=c++20 -o $@ -c $<

leanmd.decl.h:	leanmd.ci
	$(CHARMC) leanmd.ci

Compute.o: Compute.C Compute.h leanmd.decl.h defs.h physics.h ../../../include/ckcoro.h
	$(CHARMC) $(OPTS) -std=c++20 -o $@ -c $<

# The line "leanmd: N steps, checksum ..." is an integer sum of the particle
# positions over all cells: it does not depend on the PE count, on the
# reduction order or on +ck_fakemigrate, and it is the same as the one
# printed by the SDAG version in ../leanmd.
define check_same
	grep "^leanmd: " $(1) && test "$$(grep '^leanmd: ' $(1))" = "$$(grep '^leanmd: ' $(2))"
endef

test: leanmd
	$(call run, ./leanmd 3 3 3 20 +p4 ) 2>&1 | tee p4.log
	grep "^leanmd: 20 steps, checksum" p4.log
	$(call run, ./leanmd 3 3 3 20 +p1 ) 2>&1 | tee p1.log
	$(call check_same,p1.log,p4.log)
	$(call run, ./leanmd 3 3 3 20 +p4 +ck_fakemigrate ) 2>&1 | tee fake.log
	$(call check_same,fake.log,p4.log)

testp: leanmd
	$(call run, +p$(P) ./leanmd $(P) $(P) $(P) 20 )

clean:
	rm -f *.decl.h *.def.h *.o leanmd leanmd.prj charmrun
	rm -f p4.log p1.log fake.log

.PHONY: all clean test testp
