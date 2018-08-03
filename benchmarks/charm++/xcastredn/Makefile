-include ../../common.mk
# Point this to your charm installation
CHARMHOME = ../../..

# Charm directory structure
CHARMBIN := $(CHARMHOME)/bin
CHARMINC := $(CHARMHOME)/include

# The relevant source files for this project
HDR       = myChareArray.h testController.h convMsgHandlers.h reductionBenchmark.h
SRC       = myChareArray.C testController.C convMsgHandlers.C
OBJ       = $(SRC:.C=.o)
INTF      = reductionBenchmark.ci   

# Specify the exe name and the arguments to run it with
NP        = 4
TARGET    = testReduction
ARGS      = 10 8 128

# Specify the compilers, run script, flags etc.
CXX       = $(CHARMBIN)/charmc
OPTS     ?= -optimize -production
CXXFLAGS += -language charm++ $(OPTS)
CPPFLAGS +=
LDFLAGS  += -module CkMulticast
EXECFLAGS = +p$(NP)
ifdef $(NODELIST)
  EXECFLAGS += ++nodelist $(NODELIST)
endif


########### This stuff should be able take care of itself ############

.PHONY: all clean again test bgtest translateInterface

all: $(TARGET)

$(TARGET): $(OBJ) 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	$(RM) $(wildcard *.decl.h *.def.h *.d *.di *.o *.ci.stamp) $(TARGET) charmrun

again: 
	$(MAKE) clean; $(MAKE)

test: all
	@echo "########################################################################################"
	$(call run, $(EXECFLAGS) ./$(TARGET) $(ARGS))

%.ci.stamp: %.ci
	$(CXX) $< && touch $@

# Dependencies

convMsgHandlers.o: \
  convMsgHandlers.C \
  reductionBenchmark.h \
  reductionBenchmark.decl.h \
  convMsgHandlers.h \
  myChareArray.h \
  testController.h \
  reductionBenchmark.def.h \

myChareArray.o: \
  myChareArray.C \
  myChareArray.h \
  reductionBenchmark.h \
  reductionBenchmark.decl.h \
  convMsgHandlers.h \

testController.o: \
  testController.C \
  testController.h \
  reductionBenchmark.h \
  reductionBenchmark.decl.h \

reductionBenchmark.decl.h reductionBenchmark.def.h: reductionBenchmark.ci.stamp
