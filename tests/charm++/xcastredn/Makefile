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
ARGS      = 10 8 128 10 1

# Specify the compilers, run script, flags etc.
CXX       = $(CHARMBIN)/charmc
OPTS     ?= -O2
CXXFLAGS += -language charm++ $(OPTS)
CPPFLAGS +=
LDFLAGS  += -module CkMulticast
EXEC      = ./charmrun
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
	$(RM) $(wildcard *.decl.h *.def.h *.d *.di *.o) $(TARGET) charmrun 

again: 
	$(MAKE) clean; $(MAKE)

test: all
	@echo "########################################################################################"
	$(EXEC) $(EXECFLAGS) ./$(TARGET) $(ARGS)


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
	@grep "module" $< | sed -e "s|.*module[ ]\+\([^ ]\+\).\+|\1\.decl\.h \1\.def\.h $@: $<\n\t$(CXX) -c $<\n|g" > $@

# Include the generated files containing dependency info
-include $(SRC:.C=.d)
-include $(INTF:.ci=.di)

