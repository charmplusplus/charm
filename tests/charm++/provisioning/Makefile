-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

TARGET = launch

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -o $@ $^

%.decl.h: %.ci
	$(CHARMC) $<

%.o: %.C %.decl.h
	$(CHARMC) -c $<

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(TARGET) charmrun *.log *.sum *.sts $(TARGET).exe $(TARGET).pdb $(TARGET).ilk

test: all
	$(call run, ./$(TARGET))

testp: all
	$(call run, +p$(P) ./$(TARGET))

smptest:

.PRECIOUS: %.decl.h %.def.h
.SUFFIXES:
