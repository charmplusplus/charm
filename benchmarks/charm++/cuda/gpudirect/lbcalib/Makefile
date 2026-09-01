-include ../../../../common.mk

OPTS = -O3

CHARM_DIR = ../../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include
LD_LIBS =

TARGET = lbcalib
all: $(TARGET)

OBJS = $(TARGET).o

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).def.h: $(TARGET).ci

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) -c $<

# Exercises every tier: 2 hosts, 2 processes per host, 2 PEs per process, and
# two processes sharing a device so ipc_same_gpu is reachable.
test: all
	$(call run, +p8 ./lbcalib -o lbcost.conf +gpushm)

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
