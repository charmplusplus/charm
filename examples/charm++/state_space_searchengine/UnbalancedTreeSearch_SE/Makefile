ifndef RNG
	RNG=BRG
endif

ifeq ($(RNG), Devine)
	RNG_SRC = rng/devine_sha1.c
	RNG_INCL= rng/devine_sha1.h
	RNG_DEF = -DDEVINE_RNG
endif
ifeq ($(RNG), BRG)
	RNG_SRC = rng/brg_sha1.c
	RNG_INCL= rng/brg_sha1.h
	RNG_DEF = -DBRG_RNG
endif
ifeq ($(RNG), ALFG)
	RNG_SRC = rng/alfg.c
	RNG_INCL= rng/alfg.h
	RNG_DEF = -DUTS_ALFG
endif

OPTS = -DBRG_C99_TYPES #-std=c99 

include ../Makefile.common

#COMMON_SRCS = uts-charm.C  uts.C searchEngineAPI.h    uts-charm.decl.h uts-charm.def.h
COMMON_SRCS =  main.C uts.C searchEngineAPI.C
COMMON_DEF  =    main.decl.h main.def.h uts.h

OBJS = main.o uts.o

all: main main.nb

projections:
	main.prj main.nb.prj

main: $(COMMON_SRCS) $(RNG_SRC) $(COMMON_DEF)
	$(CHARMC) -language charm++  $(RNG_DEF) -o $@  $(COMMON_SRCS) $(RNG_SRC) -module searchEngine

main.prj: $(COMMON_SRCS) $(RNG_SRC) $(COMMON_DEF)
	$(CHARMC) -language charm++  $(RNG_DEF) -o $@  $(COMMON_SRCS) $(RNG_SRC) -module searchEngine -tracemode projections

main.nb: $(COMMON_SRCS) $(RNG_SRC) $(COMMON_DEF)
	$(CHARMC) -language charm++  $(RNG_DEF) -o $@  $(COMMON_SRCS) $(RNG_SRC) -module searchEngine -balance neighbor

main.nb.prj: $(COMMON_SRCS) $(RNG_SRC) $(COMMON_DEF)
	$(CHARMC) -language charm++  $(RNG_DEF) -o $@  $(COMMON_SRCS) $(RNG_SRC) -module searchEngine -balance neighbor -tracemode projections

main.decl.h main.def.h: main.ci
	$(CHARMC)  main.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o main charmrun main main.nb main.prj main.nb.prj *~
