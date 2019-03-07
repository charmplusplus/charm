-include ../../../common.mk
# Point this to your charm installation
CHARM ?= ../../../..

# Charm directory structure
CHARMBIN := $(CHARM)/bin
CHARMINC := $(CHARM)/include

# The relevant source files for this project
HDR       =
SRC       = hello.C
OBJ       = $(SRC:.C=.o)

# Specify the exe name and the arguments to run it with
NP        = 4
TARGET    = xarraySection
ARGS      = 10 3 3

# Specify the compilers, run script, flags etc.
CXX       = $(CHARMBIN)/charmc
CHARMC    = $(CHARMBIN)/charmc
CXXFLAGS += -language charm++ $(OPT)
EXECFLAGS = +p$(NP)
ifdef $(NODELIST)
  EXECFLAGS += ++nodelist $(NODELIST)
endif


########### This stuff should be able take care of itself ############

.PHONY: all clean again test bgtest translateInterface

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	$(RM) $(wildcard *.decl.h *.def.h *.o *.log *.projrc *.sts *.stamp) $(TARGET) charmrun

again:
	$(MAKE) clean; $(MAKE)

test: all
	@echo "########################################################################################"
	$(call run, $(EXECFLAGS) ./$(TARGET) $(ARGS))

hello.decl.h hello.def.h: hello.ci.stamp

# Compilation rule for ci files
%.ci.stamp: %.ci
	$(info-ci)
	$q$(CHARMC) $< && touch $@

hello.o: hello.C hello.decl.h hello.def.h
	$(CHARMC) -c hello.C
