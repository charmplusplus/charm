include ../common.mk

LIB = libmoduleCkCache.a
LIBOBJ = CkCache.o

HEADERS = $(CDIR)/include/CkCache.decl.h \
          $(CDIR)/include/CkCache.def.h \
          $(CDIR)/include/CkCache.h 
LIBDEST =  $(LIBDIR)/$(LIB)
CHARMXI_FLAGS = -E

CIFILES = CkCache.ci

all: $(LIBDEST) $(HEADERS)

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ) 

CkCache.def.h CkCache.decl.h: INTERFACE

BUILD_INT16=$(shell CHARMINC=../../../; if test -f ../../../conv-config.sh; then . ../../../conv-config.sh; echo $$CMK_HAS_INT16; fi )
USING_CUDA=$(shell CHARMINC=../../../; if test -f ../../../conv-config.sh; then . ../../../conv-config.sh; echo $$BUILD_CUDA; fi )

ifeq "$(BUILD_INT16)" "1"
ifneq "$(USING_CUDA)" "1"
CHARMXI_FLAGS += -DCMK_HAS_INT16=1
endif
endif

INTERFACE: $(CIFILES)
	$(CHARMC) $(CHARMXI_FLAGS) -c CkCache.ci
	touch INTERFACE

CkCache.o: CkCache.C $(HEADERS)
	$(CHARMC) -I../../.. -c -o CkCache.o CkCache.C

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB) INTERFACE

realclean: clean
	rm -f $(LIBDEST) $(HEADERS)
