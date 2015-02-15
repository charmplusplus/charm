-include ../../common.mk
# Point this to your charm installation
CHARMHOME = ../../..

# Charm directory structure
CHARMBIN := $(CHARMHOME)/bin
CHARMINC := $(CHARMHOME)/include

# The relevant source files for this project
HDR       = mylib.h utils.h
SRC       = mylib.C pgm.C
OBJ       = $(SRC:.C=.o)
INTF      = mylib.ci pgm.ci   

# Specify the exe name and the arguments to run it with
NP        = 4
TARGET    = testTemplateEP
ARGS      =

# Specify the compilers, run script, flags etc.
CXX       = $(CHARMBIN)/charmc
CXXFLAGS += -language charm++ $(OPTS)
CPPFLAGS += $(OPTS)
LDFLAGS  +=
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
	$(RM) $(wildcard *.decl.h *.def.h *.d *.di *.ci.stamp *.o) $(TARGET) charmrun 

again: 
	$(MAKE) clean; $(MAKE)

test: all
	@echo "########################################################################################"
	$(call run, $(EXECFLAGS) ./$(TARGET) $(ARGS))


mylib.o: mylib.C mylib.h mylib.decl.h mylib.def.h
pgm.o: pgm.C client.decl.h mylib.h mylib.decl.h mylib.def.h utils.h client.def.h

# Rule to generate dependency info for charm++ interface (ci) definition files
%.di: %.ci
	$(info Generating dependencies for $<)
	@$(CXX) -M $< > $@

%.ci.stamp: %.ci
	$(CXX) $< && touch $@

# Include the generated files containing dependency info
-include $(INTF:.ci=.di)

