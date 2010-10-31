CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
LIBDIR=$(CDIR)/lib
CHARMINC=$(CDIR)/include

$(CDIR)/include/%.h: %.h
	cp $< $@

