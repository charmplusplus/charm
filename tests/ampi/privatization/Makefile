-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
AMPICC=../../../bin/ampicxx $(OPTS)
AMPIF90=../../../bin/ampif90 $(OPTS)


LANGUAGES := \
  cxx \
  f90 \


# for constructing individual build rules
VARIANTS := \
  control \
  swapglobals \
  tlsglobals \
  roseomptlsglobals \
  fsglobals \
  pipglobals \


# for determining `all`, `test` rules
CANDIDATES := \
  swapglobals \
  tlsglobals \

# Don't enable these for universal testing because they are not 100% portable to all platforms and toolchain versions.
# ifeq (1,$(CMK_SUPPORTS_FSGLOBALS))
#   CANDIDATES += fsglobals
# endif
# ifeq (1,$(CMK_SUPPORTS_PIPGLOBALS))
#   CANDIDATES += pipglobals
# endif


# Define what options to pass to charmc for each method.
$(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),$(eval OPTS_$j_$i := -$i)))
OPTS_cxx_control :=
OPTS_f90_control :=
OPTS_f90_tlsglobals += -fopenmp

# Define the features that each privatization method supports.
FEATURES_cxx_control := migration staticvars sharedlib dynamiclib
FEATURES_cxx_swapglobals := migration
FEATURES_cxx_tlsglobals := migration staticvars sharedlib
FEATURES_cxx_roseomptlsglobals := migration staticvars sharedlib
FEATURES_cxx_fsglobals := staticvars
FEATURES_cxx_pipglobals := staticvars sharedlib


# Determine which methods to test in the current environment and target.
TARGETS :=

define TEST_TARGET_AVAILABILITY
  AMPICC_OUTPUT_$1 := $$(shell $$(AMPICC) -print-building-blocks $$(OPTS_cxx_$1) 2>&1)
  ifeq (,$$(findstring Error:,$$(AMPICC_OUTPUT_$1)))
    ifeq (,$$(findstring Warning: -$1,$$(AMPICC_OUTPUT_$1)))
      TARGETS += cxx-$1
    endif
  endif

  AMPIF90_OUTPUT_$1 := $$(shell $$(AMPIF90) -print-building-blocks $$(OPTS_f90_$1) 2>&1)
  ifeq (,$$(findstring Error:,$$(AMPIF90_OUTPUT_$1)))
    ifeq (,$$(findstring Warning: -$1,$$(AMPIF90_OUTPUT_$1)))
      TARGETS += f90-$1
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

everything: $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),$j-$i))
test-everything: $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),test-$j-$i))


add_if_contains = $(subst $2,$3,$(findstring $2,$1))

define VARIANT_RULES

$1: $$(foreach j,$$(LANGUAGES),$$j-$1)

# link
cxx-$1: $1-test.o $1-test-cxx.o cxx-$1-other-object.o | libcxx-$1-static-library.a $$(call add_if_contains,$$(FEATURES_cxx_$1),sharedlib,libcxx-$1-shared-library.so) $$(call add_if_contains,$$(FEATURES_cxx_$1),dynamiclib,libcxx-$1-shared-library-dynamic.so)
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) $$^ -balancer TreeLB -rpath-origin -L. -lcxx-$1-static-library $$(call add_if_contains,$$(FEATURES_cxx_$1),sharedlib,-lcxx-$1-shared-library) $$(call add_if_contains,$$(FEATURES_cxx_$1),dynamiclib,-ldl)

f90-$1: $1-test.o $1-test-f90.o
	$$(AMPIF90) -o $$@ $$(OPTS_f90_$1) $$^ -balancer TreeLB

libcxx-$1-%.so: cxx-$1-%-so.o
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) -shared -standalone $$<

libcxx-$1-%.a: cxx-$1-%.o
	ar cr $$@ $$^
	ranlib $$@

$1-%.o cxx-$1-%.o: %.C
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) -Dtest_using_$1 -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_cxx_$1),-Dtest_$$i) -c $$<

$1-%-so.o cxx-$1-%-so.o: %.C
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) -Dtest_using_$1 -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_cxx_$1),-Dtest_$$i) -shared -c $$<

$1-%.o f90-$1-%.o: %.f90
	$$(AMPIF90) -o $$@ $$(OPTS_f90_$1) -c $$<

# test
test-cxx-$1: cxx-$1
	$$(call run, ./cxx-$1 +p1 +vp2 +balancer RotateLB)
	$$(call run, ./cxx-$1 +p1 +vp4 +balancer RotateLB)
	$$(call run, ./cxx-$1 +p2 +vp2 +balancer RotateLB)
	$$(call run, ./cxx-$1 +p2 +vp4 +balancer RotateLB)

test-f90-$1: f90-$1
	$$(call run, ./f90-$1 +p1 +vp2 +balancer RotateLB)
	$$(call run, ./f90-$1 +p1 +vp4 +balancer RotateLB)
	$$(call run, ./f90-$1 +p2 +vp2 +balancer RotateLB)
	$$(call run, ./f90-$1 +p2 +vp4 +balancer RotateLB)

test-$1: $$(foreach j,$$(LANGUAGES),test-$$j-$1)

endef

$(foreach i,$(VARIANTS),$(eval $(call VARIANT_RULES,$i)))

# overrides
tlsglobals-test-f90.o: test-f90-tlsglobals.f90
	$(AMPIF90) -o $@ $(OPTS_f90_tlsglobals) -c $<


FUNCPTR_SHIM_SUFFIX := .user

clean:
	rm -f *.o *.a *.so *.mod $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),$(strip $j-$i $(wildcard $j-$i$(FUNCPTR_SHIM_SUFFIX)*)))) charmrun ampirun

.SUFFIXES:
.PHONY: all test everything test-everything $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),test-$j-$i) test-$i $i)
