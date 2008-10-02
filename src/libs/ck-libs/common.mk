CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
LIBDIR=$(CDIR)/lib

$(CDIR)/include/%.h: %.h
	cp $< $@