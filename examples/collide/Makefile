CHARMRUN=../../bin/./charmrun
-include ../common.mk
DIRS = collidecharm collidethread

all test-local clean:
	for d in $(DIRS); do  \
	    $(MAKE) -C $$d $@ || exit 1; \
        done
