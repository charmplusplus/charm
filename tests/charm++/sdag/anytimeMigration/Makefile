CHARMC = ../../../../bin/charmc $(OPTS)
CXX = $(CHARMC)
CC = $(CXX)
TARGET = migration
RUN=./charmrun

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
	$(RUN) +p$(PROCS) ./$<  $(TESTOPTS)
