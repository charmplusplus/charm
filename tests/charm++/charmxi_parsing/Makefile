-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)


TARGETS = \
  Typename \
  std_array \
  ParameterPack \

all: $(TARGETS)

test: $(foreach i,$(TARGETS),test-$i)


define TARGETRULES

# link
$1: $1.o
	$$(CHARMC) -language charm++ -o $$@ $$^

# test
test-$1: $1
	$$(call run,./$1)

# avoid deleting decl/def after building them
.PRECIOUS: $1.decl.h $1.def.h

endef

$(foreach i,$(TARGETS),$(eval $(call TARGETRULES,$i)))


%.decl.h: %.ci
	$(CHARMC) $^

%.o: %.C %.decl.h
	$(CHARMC) -c $<


clean:
	rm -f *.decl.h *.def.h *.o $(TARGETS) charmrun

.SUFFIXES:
.PHONY: all
