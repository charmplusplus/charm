-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pingpong

pingpong: pingpong.o
	$(CHARMC) -language charm++ -o pingpong pingpong.o

pingpong.decl.h: pingpong.ci
	$(CHARMC) pingpong.ci

pingpong.o: pingpong.C pingpong.decl.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 -c pingpong.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o pingpong charmrun *~

# Short runs: the numbers are for the measurement recipe in
# doc/coroutine-sdag-design.md, not for the test.
test: all
	$(call run, ./pingpong plain 20000 +p1 )
	$(call run, ./pingpong sdag 20000 +p1 )
	$(call run, ./pingpong coro 20000 +p1 )
	$(call run, ./pingpong plain 20000 +p2 )
	$(call run, ./pingpong sdag 20000 +p2 )
	$(call run, ./pingpong coro 20000 +p2 )

.PHONY: all clean test
