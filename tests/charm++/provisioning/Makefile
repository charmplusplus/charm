-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

MACHINE_LAYER := $(CMK_GDIR)

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
ifeq (,$(findstring +p,$(TESTOPTS)))
ifneq (,$(filter netlrts verbs,$(MACHINE_LAYER)))
	$(call run, ./$(TARGET) ++processPerHost 1)
	$(call run, ./$(TARGET) ++processPerSocket 1)
	$(call run, ./$(TARGET) ++processPerCore 1)
	$(call run, ./$(TARGET) ++processPerPU 1)
	$(call run, ./$(TARGET) ++autoProvision)
endif
endif

smptest:
ifeq (,$(findstring +p,$(TESTOPTS)))
ifneq (,$(filter netlrts verbs,$(MACHINE_LAYER)))
	$(call run, ./$(TARGET) ++oneWthPerHost)
	$(call run, ./$(TARGET) ++oneWthPerSocket)
	$(call run, ./$(TARGET) ++oneWthPerCore)
	$(call run, ./$(TARGET) ++oneWthPerPU)
	$(call run, ./$(TARGET) ++processPerHost 1 ++oneWthPerHost)
	$(call run, ./$(TARGET) ++processPerHost 1 ++oneWthPerSocket)
	$(call run, ./$(TARGET) ++processPerHost 1 ++oneWthPerCore)
	$(call run, ./$(TARGET) ++processPerHost 1 ++oneWthPerPU)
	$(call run, ./$(TARGET) ++processPerSocket 1 ++oneWthPerSocket)
	$(call run, ./$(TARGET) ++processPerSocket 1 ++oneWthPerCore)
	$(call run, ./$(TARGET) ++processPerSocket 1 ++oneWthPerPU)
	$(call run, ./$(TARGET) ++processPerCore 1 ++oneWthPerCore)
	$(call run, ./$(TARGET) ++processPerCore 1 ++oneWthPerPU)
	$(call run, ./$(TARGET) ++processPerPU 1 ++oneWthPerPU)
endif
endif

.PRECIOUS: %.decl.h %.def.h
.SUFFIXES:
