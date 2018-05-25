-include ../../common.mk
AMPICC=../../../bin/ampicxx $(OPTS)


VARIANTS := \
  isomalloc \
  os-isomalloc \


# for determining `all`, `test` rules
CANDIDATES := $(VARIANTS)


# Determine which methods to test in the current environment and target.
TARGETS :=

define TEST_TARGET_AVAILABILITY
  AMPICC_OUTPUT_$1 := $$(shell $$(AMPICC) -print-building-blocks -memory $1)
  ifeq (,$$(findstring Error:,$$(AMPICC_OUTPUT_$1)))
    ifeq (,$$(findstring Warning:,$$(AMPICC_OUTPUT_$1)))
      TARGETS += $1
    endif
  endif
endef

$(foreach i,$(CANDIDATES),$(eval $(call TEST_TARGET_AVAILABILITY,$i)))


all: $(foreach i,$(TARGETS),$i)
test: $(foreach i,$(TARGETS),test-$i)

everything: $(foreach i,$(VARIANTS),$i)
test-everything: $(foreach i,$(VARIANTS),test-$i)


define VARIANT_RULES

$1: $1.o
	$$(AMPICC) -o $$@ $$^ -balancer RotateLB -memory $1

$1.o: test.C
	$$(AMPICC) -o $$@ -memory $1 -Disomalloc_method=$1 -c $$<

test-$1: $1
	$$(call run, ./$1 +p1 +vp1 )
	$$(call run, ./$1 +p1 +vp2 )
	$$(call run, ./$1 +p1 +vp4 )
	$$(call run, ./$1 +p2 +vp2 )
	$$(call run, ./$1 +p2 +vp4 )

endef

$(foreach i,$(VARIANTS),$(eval $(call VARIANT_RULES,$i)))


clean:
	rm -f *.o *.a *.so $(foreach i,$(VARIANTS),$i) charmrun ampirun

.SUFFIXES:
.PHONY: all test everything test-everything $(foreach i,$(VARIANTS),test-$i)
