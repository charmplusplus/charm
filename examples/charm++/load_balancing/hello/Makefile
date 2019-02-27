-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)

TARGET = hello

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -module CommonLBs -o $@ $^

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

test: $(TARGET)
	$(call run, +p4 ./$(TARGET) +balancer GreedyRefineLB +LBDebug 1)

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun
