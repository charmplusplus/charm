include ../conv-mach-opt.mak
include ../uFcontext.mk

CDIR=../../
CHARMC=../../bin/charmc

.SUFFIXES : .S .asm .o

ifneq ($(findstring windows,$(TARGET)),)
  ifeq ($(CMK_COMPILER),msvc)
    TARGET := $(subst sysv_windows,ms_pe_masm,$(TARGET))
  endif
endif

CALLS := \
  jump \
  make \
  ontop \

all: $(addsuffix _$(TARGET).o, $(CALLS))

%.o: %.S
	$(CHARMC) -c $<

%.o: %.asm
	$(CHARMC) -c $< -DBOOST_CONTEXT_EXPORT

clean:
	rm *.o
