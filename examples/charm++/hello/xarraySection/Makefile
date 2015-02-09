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
INTF      = hello.ci

# Specify the exe name and the arguments to run it with
NP        = 4
TARGET    = xarraySection
ARGS      = 10 3 3

# Specify the compilers, run script, flags etc.
CXX       = $(CHARMBIN)/charmc
CHARMC    = $(CHARMBIN)/charmc
OPT       = -O0 -g
CPPFLAGS +=
CXXFLAGS += -language charm++ $(OPT)
LDFLAGS  += -module CkMulticast
LDLIBS   +=
EXECFLAGS = +p$(NP)
ifdef $(NODELIST)
  EXECFLAGS += ++nodelist $(NODELIST)
endif


########### This stuff should be able take care of itself ############

.PHONY: all clean again test bgtest translateInterface

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	$(RM) $(wildcard *.decl.h *.def.h *.d *.di *.o *.log *.projrc *.sts *.stamp) $(TARGET) charmrun

again:
	$(MAKE) clean; $(MAKE)

test: all
	@echo "########################################################################################"
	$(call run, $(EXECFLAGS) $(TARGET) $(ARGS))

####### Pattern rules
# Rule to generate dependency information for C++ source files
%.d: %.C
	$(info Generating dependencies for $<)
	@g++ -MM -MG $(CPPFLAGS) $(INCDIRS:%=-I%) -I$(CHARMINC) $< | perl $(CHARMBIN)/dep.pl $(CHARMINC) > $@
#	@$(SHELL) -ec 'g++ -MM -MG $(CPPFLAGS) $(INCDIRS:%=-I%) $< \
#	| sed '\''s/\($*\)\.o[ :]*/\1.o $@ : /g'\'' > $@; \
#	[ -s $@ ] || rm -f $@'

# Rule to generate dependency info for charm++ interface (ci) definition files
%.di: %.ci
	$(info Generating dependencies for $<)
	@$(CHARMC) -M $< > $@

# Compilation rule for ci files
%.ci.stamp: %.ci
	$(info-ci)
	$q$(CHARMC) $< && touch $@

# Include the generated files containing dependency info
-include $(SRC:.C=.d)
-include $(INTF:.ci=.di)

