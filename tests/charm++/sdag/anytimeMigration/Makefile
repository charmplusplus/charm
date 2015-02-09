-include ../../../common.mk
-include ../../../common.mk
CHARMC = ../../../../bin/charmc $(OPTS)
CXX = $(CHARMC)
CC = $(CXX)
TARGET = migration

ifndef $(PROCS)
	PROCS=2
endif

all: $(TARGET)

%.decl.h: %.ci
	$(CXX) $(CXXFLAGS) $^

$(TARGET).C: $(TARGET).decl.h

clean:
	-rm -rf $(wildcard *.decl.h *.def.h *.o $(TARGET) charmrun *~)

test: migration
	$(call run, +p$(PROCS) ./$< )
