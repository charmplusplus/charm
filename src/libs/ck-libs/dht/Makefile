include ../common.mk

MODULE = CkDhtManager
LIB = libmodule$(MODULE).a
LIBOBJ = $(MODULE).o

CK_HEADER_PATH = $(CDIR)/include/ck
HEADERS = $(CDIR)/include/$(MODULE).decl.h \
          $(CDIR)/include/$(MODULE).def.h \
          $(CDIR)/include/$(MODULE).h \
          $(CK_HEADER_PATH)/dht.h 
LIBDEST =  $(LIBDIR)/$(LIB)
CHARMXI_FLAGS = -E -I$(CDIR)/tmp

CIFILES = $(MODULE).ci

all: $(HEADERS)

$(MODULE).def.h $(MODULE).decl.h: INTERFACE

INTERFACE: $(CIFILES)
	$(CHARMC) $(CHARMXI_FLAGS) -c $(MODULE).ci
	touch INTERFACE

$(CK_HEADER_PATH):
	mkdir -p $(CK_HEADER_PATH)

$(CK_HEADER_PATH)/dht.h: dht.h $(CK_HEADER_PATH)
	cp dht.h $(CK_HEADER_PATH)/dht.h

clean:
	rm -f conv-host *.o *.decl.h *.def.h core $(LIB) INTERFACE

realclean: clean
	rm -f $(HEADERS)
