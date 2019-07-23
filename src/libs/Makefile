CHARMC=../../bin/charmc $(OPTS)

all: convlibs charmlibs

convlibs:
	$(MAKE) -C conv-libs all OPTS='$(OPTS)'

charmlibs: convlibs
	$(MAKE) -C ck-libs all OPTS='$(OPTS)'

clean:
	$(MAKE) -C conv-libs clean
	$(MAKE) -C ck-libs clean
