OPTS = -O0 -g

CHARM_DIR = ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
CHARM_INC = -I$(CHARM_DIR)/include

LD_LIBS =

TARGET = iterative
all: $(TARGET)

OBJS = $(TARGET).o

$(TARGET): $(OBJS)
		$(CHARMC) -language charm++ -module CommonLBs -o $@ $(OBJS) $(LD_LIBS)

$(TARGET).decl.h: $(TARGET).ci
		$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h $(TARGET).def.h
		$(CHARMC) -c $<

clean:
		rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun
