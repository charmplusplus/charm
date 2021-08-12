-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
AMPICC=../../../bin/ampicxx $(OPTS)
AMPIF90=../../../bin/ampif90 $(OPTS)


LANGUAGES := cxx
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  LANGUAGES += f90
endif


# for constructing individual build rules
VARIANTS := \
  control \
  swapglobals \
  tlsglobals \
  roseomptlsglobals \
  fsglobals \
  pipglobals \

PHONIES := \
  everything \

# for determining `all`, `test` rules
TARGETS :=

ifeq (1,$(CMK_SUPPORTS_SWAPGLOBALS))
  TARGETS += swapglobals
endif
ifeq (1,$(CMK_SUPPORTS_TLSGLOBALS))
  TARGETS += tlsglobals
endif
ifeq (1,$(CMK_SUPPORTS_FSGLOBALS))
  TARGETS += fsglobals
endif
ifeq (1,$(CMK_SUPPORTS_PIPGLOBALS))
  TARGETS += pipglobals
endif

$(foreach i,$(VARIANTS),$(eval LANGUAGES_$i := $(LANGUAGES)))
ifneq (1,$(CMK_HAS_OPENMP))
  LANGUAGES_tlsglobals := $(filter-out f90,$(LANGUAGES_tlsglobals))
endif


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


# Universal settings
LINKOPTS := -balancer TreeLB


# Rule construction

EMPTY_MESSAGE := "No privatization methods supported for current AMPI target."

all: $(foreach i,$(TARGETS),$(foreach j,$(LANGUAGES_$i),$j-$i))
ifeq (,$(TARGETS))
	@echo $(EMPTY_MESSAGE)
endif

test: $(foreach i,$(TARGETS),$(foreach j,$(LANGUAGES_$i),test-$j-$i))
ifeq (,$(TARGETS))
	@echo $(EMPTY_MESSAGE)
endif

testp: $(foreach i,$(TARGETS),$(foreach j,$(LANGUAGES_$i),testp-$j-$i))
ifeq (,$(TARGETS))
	@echo $(EMPTY_MESSAGE)
endif

everything: $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),$j-$i))
test-everything: $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),test-$j-$i))

add_if_contains = $(subst $2,$3,$(findstring $2,$1))

define VARIANT_RULES

$1: $$(foreach j,$$(LANGUAGES_$1),$$j-$1)
test-$1: $$(foreach j,$$(LANGUAGES_$1),test-$$j-$1)

# link
cxx-$1: $1-test.o $1-test-cxx.o cxx-$1-other-object.o | libcxx-$1-static-library.a $$(call add_if_contains,$$(FEATURES_cxx_$1),sharedlib,libcxx-$1-shared-library.so) $$(call add_if_contains,$$(FEATURES_cxx_$1),dynamiclib,libcxx-$1-shared-library-dynamic.so)
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) $$^ $$(LINKOPTS) -rpath-origin -L. -lcxx-$1-static-library $$(call add_if_contains,$$(FEATURES_cxx_$1),sharedlib,-lcxx-$1-shared-library) $$(call add_if_contains,$$(FEATURES_cxx_$1),dynamiclib,-ldl)

f90-$1: $1-test.o $1-test-f90.o
	$$(AMPIF90) -o $$@ $$(OPTS_f90_$1) $$^ $$(LINKOPTS)

libcxx-$1-%.so: cxx-$1-%-so.o
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) -shared -standalone $$<

libcxx-$1-%.a: cxx-$1-%.o
	ar cr $$@ $$^
	ranlib $$@

$1-%.o cxx-$1-%.o: %.C
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) -Dtest_using_$$(subst -,_,$1) -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_cxx_$1),-Dtest_$$i) -c $$<

$1-%-so.o cxx-$1-%-so.o: %.C
	$$(AMPICC) -o $$@ $$(OPTS_cxx_$1) -Dtest_using_$$(subst -,_,$1) -Dprivatization_method=$1 $$(foreach i,$$(FEATURES_cxx_$1),-Dtest_$$i) -shared -c $$<

$1-%.o f90-$1-%.o: %.f90
	$$(AMPIF90) -o $$@ $$(OPTS_f90_$1) -c $$<

# test
test-cxx-$1: cxx-$1
	$$(call run, ./cxx-$1 +p1 +vp2 +balancer RandCentLB)
	$$(call run, ./cxx-$1 +p1 +vp4 +balancer RandCentLB)
	$$(call run, ./cxx-$1 +p2 +vp2 +balancer RandCentLB)
	$$(call run, ./cxx-$1 +p2 +vp4 +balancer RandCentLB)

test-f90-$1: f90-$1
	$$(call run, ./f90-$1 +p1 +vp2 +balancer RandCentLB)
	$$(call run, ./f90-$1 +p1 +vp4 +balancer RandCentLB)
	$$(call run, ./f90-$1 +p2 +vp2 +balancer RandCentLB)
	$$(call run, ./f90-$1 +p2 +vp4 +balancer RandCentLB)

testp-cxx-$1: cxx-$1
	$$(call run, ./cxx-$1 +p$(P) +vp$(P) +balancer RandCentLB)
	$$(call run, ./cxx-$1 +p$(P) +vp$$$$(( $(P) * 2 )) +balancer RandCentLB)
	$$(call run, ./cxx-$1 +p$(P) +vp$$$$(( $(P) * 4 )) +balancer RandCentLB)

testp-f90-$1: f90-$1
	$$(call run, ./f90-$1 +p$(P) +vp$(P) +balancer RandCentLB)
	$$(call run, ./f90-$1 +p$(P) +vp$$$$(( $(P) * 2 )) +balancer RandCentLB)
	$$(call run, ./f90-$1 +p$(P) +vp$$$$(( $(P) * 4 )) +balancer RandCentLB)


endef

$(foreach i,$(VARIANTS),$(eval $(call VARIANT_RULES,$i)))

# overrides
tlsglobals-test-f90.o: test-f90-tlsglobals.f90
	$(AMPIF90) -o $@ $(OPTS_f90_tlsglobals) -c $<

clean:
	rm -f *.o *.a *.so *.dylib *.mod $(foreach i,$(VARIANTS),$(foreach j,$(LANGUAGES),$(strip $j-$i $(wildcard $j-$i$(CMK_USER_SUFFIX)*)))) charmrun ampirun

.SUFFIXES:
.PHONY: \
  all \
  test \
  clean \
  $(LANGUAGES) \
  $(addprefix test-,$(LANGUAGES)) \
  $(foreach i,$(VARIANTS) $(PHONIES),$(foreach j,$(LANGUAGES),test-$j-$i) test-$i $i) \
