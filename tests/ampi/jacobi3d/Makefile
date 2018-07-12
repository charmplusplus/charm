-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS) $(MOPTS)

OBJS = jacobi.o

CANDIDATES := jacobi
TARGETS :=

PARAMETERS := -memory isomalloc -swapglobals

define TEST_TARGET_AVAILABILITY
  AMPICC_OUTPUT_$1 := $$(shell $$(CHARMC) -print-building-blocks $$(PARAMETERS))
  ifeq (,$$(findstring Error:,$$(AMPICC_OUTPUT_$1)))
    ifeq (,$$(findstring Warning:,$$(AMPICC_OUTPUT_$1)))
      TARGETS += $1
    endif
  endif
endef

$(foreach i,$(CANDIDATES),$(eval $(call TEST_TARGET_AVAILABILITY,$i)))


all: $(TARGETS)

jacobi: $(OBJS)
	$(CHARMC) -o $@ $(OBJS) -lm $(PARAMETERS) -module DummyLB

jacobi.o: jacobi.C
	$(CHARMC) -language ampic -c $(PARAMETERS) $<

syncfttest causalfttest: $(foreach i,$(TARGETS),test-$i)

test-jacobi: jacobi
	$(call run, ./jacobi 2 2 2 200 +vp8 +p8 +balancer DummyLB +isomalloc_sync +killFile kill_01.txt )
	$(call run, ./jacobi 4 2 2 200 +vp16 +p8 +balancer DummyLB +isomalloc_sync +killFile kill_02.txt )

clean:
	rm -f *.o jacobi jacobi.o *~ moduleinit.C charmrun conv-host jacobi-cpp jacobi.iso jacobi-get jacobi.tls ampirun
	rm -rf 40 80 120
