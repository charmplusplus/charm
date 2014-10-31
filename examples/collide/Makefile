DIRS = collidecharm collidethread

all test clean:
	for d in $(DIRS); do  \
	    $(MAKE) -C $$d $@ || exit 1; \
        done
