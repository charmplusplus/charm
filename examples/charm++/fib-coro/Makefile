-include ../../common.mk
CHARMC = ../../../bin/charmc
BINARY = fib

CHARMCFLAGS = $(OPTS) -std=c++20
CHARMCLINKFLAGS = -language charm++ $(OPTS)

%.o: %.C

all: $(BINARY)
$(BINARY): $(patsubst %.C,%.o,$(wildcard *.C))
	$(CHARMC) $(CHARMCLINKFLAGS) -o $@ $+

.SECONDARY: $(patsubst %.C,%.decl.h,$(wildcard *.C))
.SECONDARY: $(patsubst %.C,%.def.h,$(wildcard *.C))

%.o: %.C %.decl.h %.def.h ../../../include/ckcoro.h
	$(CHARMC) $(CHARMCFLAGS) $<

%.decl.h %.def.h: %.ci
	$(CHARMC) $(CHARMCFLAGS) $<

# Each run must print the 7th Fibonacci number; +ck_fakemigrate rebuilds the
# coroutine state at every wait.
test: $(BINARY)
	$(call run, ./$(BINARY) +p4 7 ) 2>&1 | tee p4.log
	grep "Fibonacci number is: 13" p4.log
	$(call run, ./$(BINARY) +p1 7 ) 2>&1 | tee p1.log
	grep "Fibonacci number is: 13" p1.log
	$(call run, ./$(BINARY) +p4 7 +ck_fakemigrate ) 2>&1 | tee fake.log
	grep "Fibonacci number is: 13" fake.log

testp: $(BINARY)
	$(call run, ./$(BINARY) +p$(P) $$(( $(P) + 10 )) )

clean:
	rm -f *.o *.decl.h *.def.h charmrun $(BINARY) p4.log p1.log fake.log

.PHONY: all test testp clean
