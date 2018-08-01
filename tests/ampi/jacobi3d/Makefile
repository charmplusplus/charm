-include ../../common.mk
AMPICC=../../../bin/ampicxx $(OPTS)

OBJS = jacobi.o

CANDIDATES := jacobi
TARGETS :=

PARAMETERS := -memory isomalloc

# Ignore this test on platforms that don't support Isomalloc:
define TEST_TARGET_AVAILABILITY
  AMPICC_OUTPUT_$1 := $$(shell $$(AMPICC) -print-building-blocks $$(PARAMETERS) 2>&1)
  ifeq (,$$(findstring Error:,$$(AMPICC_OUTPUT_$1)))
    TARGETS += $1
  endif
endef

$(foreach i,$(CANDIDATES),$(eval $(call TEST_TARGET_AVAILABILITY,$i)))

all: $(TARGETS)

jacobi: $(OBJS)
	$(AMPICC) -o $@ $(OBJS) -lm $(PARAMETERS)

jacobi.o: jacobi.C
	$(AMPICC) -c $(PARAMETERS) $<

syncfttest causalfttest: $(foreach i,$(TARGETS),test-$i)

test-jacobi: jacobi
	$(call run, ./jacobi 2 2 2 200 +vp8 +p8 +isomalloc_sync +killFile kill_01.txt )
	$(call run, ./jacobi 4 2 2 200 +vp16 +p8 +isomalloc_sync +killFile kill_02.txt )

clean:
	rm -f *.o jacobi jacobi.o *~ moduleinit.C charmrun conv-host ampirun
	rm -rf 40 80 120
