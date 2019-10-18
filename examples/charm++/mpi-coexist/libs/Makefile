DIRS = \
  hi \
  hello \
  kNeighbor \

all: $(foreach i,$(DIRS),build-$i)
	for d in $(DIRS); do \
		cp $$d/"libmodule$$d.a" ../; \
	done

clean: $(foreach i,$(DIRS),clean-$i)

$(foreach i,$(DIRS),build-$i):
	$(MAKE) -C $(subst build-,,$@) all OPTS='$(OPTS)'

$(foreach i,$(DIRS),clean-$i):
	$(MAKE) -C $(subst clean-,,$@) clean OPTS='$(OPTS)'

libs: all
