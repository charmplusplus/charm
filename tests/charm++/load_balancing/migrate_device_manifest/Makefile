OPTS = -O2

CHARM_DIR = ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)

LD_LIBS = -module EveryLB

TARGET = migmanifest
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -o $@ $(TARGET).o $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

clean:
	rm -f *.decl.h *.def.h *.o $(TARGET) charmrun

# The injection runs pass by aborting with the right message; see migmanifest.C.
test: all
	$(call run, ./$(TARGET) +p2 +balancer RotateLB +gpushm +gpulbbuffer 32)
