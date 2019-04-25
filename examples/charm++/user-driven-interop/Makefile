-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: hello_user hello_user_with_main

hello_user.decl.h: hello_user.ci
	$(CHARMC) $<

hello_user.o: hello_user.cpp hello_user.decl.h
	$(CHARMC) -c $< -o $@

hello_user: hello_user.o
	$(CHARMC) -c $^ -o $@

hello_user_with_main.decl.h: hello_user.ci
	$(CHARMC) -DHELLO_USE_MAINCHARE $<

hello_user_with_main.o: hello_user.cpp hello_user_with_main.decl.h
	$(CHARMC) -DHELLO_USE_MAINCHARE -c $< -o $@

hello_user_with_main: hello_user_with_main.o
	$(CHARMC) -DHELLO_USE_MAINCHARE -c $^ -o $@

clean:
	rm -f hello_user hello_user_with_main *.o charmrun *.decl.h *.def.h

test: all
	$(call run, +p2 ./hello_user 8 )
	$(call run, +p2 ./hello_user_with_main 8 )
ifeq ($(CMK_SMP), 1)
ifneq ($(CMK_MULTICORE), 1)
	$(call run, +p4 ./hello_user 8 ++ppn 2 )
	$(call run, +p4 ./hello_user_with_main 8 ++ppn 2 )
endif
endif
