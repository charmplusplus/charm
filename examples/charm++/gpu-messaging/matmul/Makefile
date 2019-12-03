OPTS = -O3 -DUSE_NVTX

CHARM_DIR = $(HOME)/charm
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)

TARGET = matmul
all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -o $@ $< $(OPTS)

$(TARGET).o: $(TARGET).C $(TARGET).decl.h
	$(CHARMC) -c $<

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

clean:
	rm -f *.o *.decl.h *.def.h $(TARGET) charmrun
