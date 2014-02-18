CHARMC=../../../bin/charmc $(OPTS)

SIMPLE_DIRS= master-slave packlib ddt
DIRS= $(SIMPLE_DIRS)

all: $(SIMPLE_DIRS)

.PHONY: $(DIRS)

$(DIRS):
	$(MAKE) -C $@ all OPTS='$(OPTS)'

clean:
	for d in $(DIRS); do make -C $$d clean OPTS='$(OPTS)'; done
