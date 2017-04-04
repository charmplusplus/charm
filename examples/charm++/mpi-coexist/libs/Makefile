all: libs

DIRS=hi hello kNeighbor

libs:
	for d in $(DIRS); do \
		($(MAKE) -C $$D all || exit 1) || exit 1; \
	done
	for d in $(DIRS); do \
		cp $$d/"libmodule$$d.a" ../; \
	done

clean:
	for d in $(DIRS); do ($(MAKE) -C $$d clean); done
