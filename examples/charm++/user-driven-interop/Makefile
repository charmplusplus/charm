-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

# User-driven interop works on the LRTS machine layers and on reconverse
# (which sets CMK_USE_LRTS to 0).
ifeq ($(CMK_USE_LRTS), 1)
  INTEROP := 1
else ifeq ($(CMK_GDIR),reconverse)
  INTEROP := 1
  # +p<n> through testrun starts one process with n PEs; the ++ppn runs below
  # ask for two processes instead, so interop is tested across processes too.
  RUN_PROCS = TESTRUN_PROCS=2
endif

ifeq ($(INTEROP), 1)
  TARGETS := hello_user hello_user_with_main
else
  TARGETS :=
endif

all: $(TARGETS)

hello_user.decl.h: hello_user.ci
	$(CHARMC) $<

hello_user.o: hello_user.cpp hello_user.decl.h
	$(CHARMC) -c $< -o $@

hello_user: hello_user.o
	$(CHARMC) $^ -o $@

hello_user_with_main.decl.h: hello_user.ci
	$(CHARMC) -DHELLO_USE_MAINCHARE $<

hello_user_with_main.o: hello_user.cpp hello_user_with_main.decl.h
	$(CHARMC) -DHELLO_USE_MAINCHARE -c $< -o $@

hello_user_with_main: hello_user_with_main.o
	$(CHARMC) -DHELLO_USE_MAINCHARE $^ -o $@

clean:
	rm -f hello_user hello_user_with_main *.o charmrun *.decl.h *.def.h

test: all
ifeq ($(INTEROP), 1)
	$(call run, +p2 ./hello_user 8 )
	$(call run, +p2 ./hello_user_with_main 8 )
ifeq ($(CMK_SMP), 1)
ifneq ($(CMK_MULTICORE), 1)
	$(RUN_PROCS) $(call run, +p4 ./hello_user 8 ++ppn 2 )
	$(RUN_PROCS) $(call run, +p4 ./hello_user_with_main 8 ++ppn 2 )
endif
endif
endif

testp: all
ifeq ($(INTEROP), 1)
	$(call run, +p$(P) ./hello_user $$(( $(P) * 8 ))  )
	$(call run, +p$(P) ./hello_user_with_main $$(( $(P) * 8 )) )
endif
