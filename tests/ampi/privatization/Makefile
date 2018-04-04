-include ../../common.mk
AMPICC=../../../bin/ampicxx $(OPTS)


# for constructing individual build rules
VARIANTS := \
  control \
  swapglobals \
  tlsglobals \
  roseomptlsglobals \


# for determining `all`, `test` rules
CANDIDATES := \
  swapglobals \
  tlsglobals \

TARGETS :=

define TEST_TARGET_AVAILABILITY
  AMPICC_OUTPUT_$1 := $$(shell $$(AMPICC) -print-building-blocks -$1)
  ifeq (,$$(findstring Error: -$1,$$(AMPICC_OUTPUT_$1)))
    ifeq (,$$(findstring Warning: -$1,$$(AMPICC_OUTPUT_$1)))
      TARGETS += $1
    endif
  endif
endef

$(foreach i,$(CANDIDATES),$(eval $(call TEST_TARGET_AVAILABILITY,$i)))


EMPTY_MESSAGE := "No privatization methods supported for current AMPI target."

all: $(foreach i,$(TARGETS),$i)
ifeq (,$(TARGETS))
	@echo $(EMPTY_MESSAGE)
endif
test: $(foreach i,$(TARGETS),test-$i)
ifeq (,$(TARGETS))
	@echo $(EMPTY_MESSAGE)
endif

everything: $(foreach i,$(VARIANTS),$i)
test-everything: $(foreach i,$(VARIANTS),test-$i)

# Define the features that each privatization method supports.
FEATURES_control := staticvars sharedlib dynamiclib
FEATURES_swapglobals :=
FEATURES_tlsglobals := staticvars sharedlib
FEATURES_roseomptlsglobals := staticvars sharedlib


add_if_contains = $(subst $2,$3,$(findstring $2,$1))

define VARIANT_RULES

# link
$1: $1.o $1-other-object.o | lib$1-static-library.a $$(call add_if_contains,$$(FEATURES_$1),sharedlib,lib$1-shared-library.so) $$(call add_if_contains,$$(FEATURES_$1),dynamiclib,lib$1-shared-library-dynamic.so)
	$$(AMPICC) -o $$@ $$(subst -control,,-$1) $$^ -balancer RotateLB -rpath-origin -L. -l$1-static-library $$(call add_if_contains,$$(FEATURES_$1),sharedlib,-l$1-shared-library) $$(call add_if_contains,$$(FEATURES_$1),dynamiclib,-ldl)

lib$1-%.so: $1-%-so.o
	$$(AMPICC) -o $$@ $$(subst -control,,-$1) -shared -standalone $$<

lib$1-%.a: $1-%.o
	ar cr $$@ $$^
	ranlib $$@

$1-%.o: %.C
	$$(AMPICC) -o $$@ $$(subst -control,,-$1) -Dtest_using_$1 -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_$1),-Dtest_$$i) -c $$<

$1-%-so.o: %.C
	$$(AMPICC) -o $$@ $$(subst -control,,-$1) -Dtest_using_$1 -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_$1),-Dtest_$$i) -shared -c $$<

$1.o: test.C
	$$(AMPICC) -o $$@ $$(subst -control,,-$1) -Dtest_using_$1 -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_$1),-Dtest_$$i) -c $$<

# test
test-$1: $1
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
