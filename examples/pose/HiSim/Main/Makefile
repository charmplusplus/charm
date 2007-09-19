# NOTE: This is the basic Makefile template that includes the additional
#       Makefile.simulation makefile that is found in the location of the
#       current installation of POSE.
#
#	The Makefile.simulation set of rules allows the compilation of
#       a SINGLE simulation module. If multiple simulation modules are
#	desired, some "surgery" of Makefile.simulation (and this file)
#	will be required. 
#
# ***********************************************************************

# Default charmc options
# - changes here will be exported to upper level makefile
# *******************************************************
OPTS=-g#-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

include ../Makefile.common


# Default Location variables
# **************************
#POSE_ROOT=$(HOME)/charm/net-linux/lib

# Default module names
# - MAKE MODIFICATIONS HERE
# *************************
PGM_MODULE=Pgm
PGM=pgm
OTHER=-lconv-bigsim-logs
LIBS=-Lstdc++ -lm
#LIBS=/expand8/home/guna/charm/src/langs/bluegene/blue_seq.o

# Simulation object definitions 
# *****************************
SIM_OBJECTS=$(OBJDIR)/BgSim_sim.o $(OBJDIR)/TCsim_sim.o

# EXECUTABLE GENERATION RULES 
# - DO NOT MODIFY
#****************************

RedStorm: $(OBJDIR)/$(PGM).o
HypCubeArch: $(OBJDIR)/$(PGM).o
HypCubeArchOB: $(OBJDIR)/$(PGM).o
BlueGene: $(OBJDIR)/$(PGM).o
IB: $(OBJDIR)/$(PGM).o
OB: $(OBJDIR)/$(PGM).o

$(OBJDIR)/$(PGM).o :  $(SIM_OBJECTS)	$(PGM).C $(PGM).h $(PGM_MODULE).def.h $(PGM_MODULE).decl.h
	-test -d $(OBJDIR) || mkdir $(OBJDIR)
	$(CHARMC) -c -o $@ -I$(CHARMINC) $(INCLUDES) $(LIBS) $(PGM).C -I$(MACHINE)
 
$(PGM_MODULE).def.h $(PGM_MODULE).decl.h : $(PGM).ci
	$(CHARMC) $(INCLUDES) $(LIBS) $(PGM).ci -I$(MACHINE)
 
$(OBJDIR)/TCsim_sim.o :	BgSim.def.h TCsim_sim.C TCsim_sim.h \
			TCsim.def.h TCsim.decl.h
	-test -d $(OBJDIR) || mkdir $(OBJDIR)
	$(CHARMC) -o $@ -c $(LIBS) $(INCLUDES) -I$(CHARMINC) TCsim_sim.C -I$(MACHINE)
 
TCsim.def.h TCsim.decl.h : TCsim_sim.ci
	$(CHARMC) $(INCLUDES) $(LIBS) TCsim_sim.ci -I$(MACHINE)
 
TCsim_sim.C TCsim_sim.h TCsim_sim.ci: TCsim.C \
		TCsim.h TCsim.ci 
	$(ETRANS) TCsim

$(OBJDIR)/BgSim_sim.o :	TCsim.def.h BgSim_sim.C BgSim_sim.h \
			BgSim.def.h BgSim.decl.h
	-test -d $(OBJDIR) || mkdir $(OBJDIR)
	$(CHARMC) -o $@ -c $(LIBS) $(INCLUDES) -I$(CHARMINC) BgSim_sim.C -I$(MACHINE)
 
BgSim.def.h BgSim.decl.h : BgSim_sim.ci
	$(CHARMC) $(INCLUDES) $(LIBS) BgSim_sim.ci -I$(MACHINE)
 
BgSim_sim.C BgSim_sim.h BgSim_sim.ci: BgSim.C \
		BgSim.h BgSim.ci Switch.C Nic.C Channel.C
	$(ETRANS) BgSim Switch.C Nic.C Channel.C

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
