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

.PHONY: all clean again test translateInterface

all: $(TARGET)

$(TARGET): $(OBJ) 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	$(RM) $(wildcard *.decl.h *.def.h *.ci.stamp *.o) $(TARGET) charmrun

again: 
	$(MAKE) clean; $(MAKE)

test: all
	@echo "########################################################################################"
	$(call run, $(EXECFLAGS) ./$(TARGET) $(ARGS))

testp: all
	@echo "########################################################################################"
	$(call run, +p$(P) ./$(TARGET) $(ARGS))

smptest: all
	@echo "########################################################################################"
	$(call run, $(EXECFLAGS) ./$(TARGET) $(ARGS) ++ppn 2)
	$(call run, $(EXECFLAGS) ./$(TARGET) $(ARGS) ++ppn 4)


mylib.o: mylib.C mylib.h mylib.decl.h mylib.def.h
pgm.o: pgm.C client.decl.h mylib.h mylib.decl.h mylib.def.h utils.h client.def.h

%.ci.stamp: %.ci
	$(CXX) $< && touch $@

client.decl.h client.def.h: pgm.ci.stamp
mylib.decl.h mylib.def.h: mylib.ci.stamp
