-include ../../../../common.mk
CHARMC = ../../../../../bin/charmc $(OPTS)

TARGET = vecadd

all: $(TARGET)

$(TARGET): $(TARGET)_charm.o $(TARGET)_openacc.o
	$(CHARMC) -language charm++ -o $@ $^

$(TARGET)_charm.o: $(TARGET)_charm.C $(TARGET).decl.h $(TARGET).h
	$(CHARMC) -c $<

$(TARGET).decl.h: $(TARGET).ci
	$(CHARMC) $<

$(TARGET)_openacc.o: $(TARGET)_openacc.cpp $(TARGET).h
	pgc++ -acc -std=c++11 -ta=tesla,cuda9.2 -c $<

clean:
	rm -f *.decl.h *.def.h *.o $(TARGET) charmrun

test: all
	$(call run, ./$(TARGET) +p2 ++local)
