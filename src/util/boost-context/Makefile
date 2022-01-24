include ../conv-mach-opt.mak

CDIR=../../
CHARMC=../../bin/charmc

.SUFFIXES : .S .asm .o

ABI := sysv
SYNTAX := gas
ifneq ($(findstring darwin,$(CMK_VDIR)),)
  FORMAT := macho
else ifneq ($(findstring win,$(CMK_VDIR)),)
  ABI := ms
  FORMAT := pe
  ifeq ($(CMK_COMPILER),msvc)
    SYNTAX := masm
  endif
else
  FORMAT := elf
endif
ifneq ($(findstring arm7,$(CMK_VDIR)),)
  ARCHITECTURE := arm
  ABI := aapcs
else ifneq ($(findstring arm8,$(CMK_VDIR)),)
  ARCHITECTURE := arm64
  ABI := aapcs
else ifneq ($(findstring mips,$(CMK_VDIR)),)
  ARCHITECTURE := mips32
  ABI := o32
else ifneq ($(findstring pami,$(CMK_VDIR))$(findstring ppc,$(CMK_VDIR)),)
  ARCHITECTURE := ppc64
else ifneq ($(findstring cray,$(CMK_VDIR))$(findstring 64,$(CMK_VDIR)),)
  ARCHITECTURE := x86_64
else
  ARCHITECTURE := i386
endif

CALLS := \
  jump \
  make \
  ontop \

all: $(addsuffix _$(ARCHITECTURE)_$(ABI)_$(FORMAT)_$(SYNTAX).o, $(CALLS))

%.o: %.S
	$(CHARMC) -c $<

%.o: %.asm
	$(CHARMC) -c $< -DBOOST_CONTEXT_EXPORT

clean:
	rm *.o
