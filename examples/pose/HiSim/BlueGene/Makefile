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
#LIBS=/expand8/home/guna/charm/src/langs/bluegene/blue_seq.o

# Simulation object definitions 
# *****************************

ROUTINGDIR = ../Routing
TOPOLOGYDIR = ../Topology
OUTPUT_VCSELECTIONDIR = ../OutputVcSelection
INPUT_VCSELECTIONDIR = ../InputVcSelection
MAINDIR = ../Main
SUBDIRS = $(MAINDIR) $(ROUTINGDIR) $(TOPOLOGYDIR) $(OUTPUT_VCSELECTIONDIR) $(INPUT_VCSELECTIONDIR)
PGM = BGHiSim
OTHER = -lconv-bluegene-logs
LIBS = -Lstdc++ -lm
MACHINE = BlueGene

all:  
	@echo
	test -d ../tmp || mkdir ../tmp;
	cd ..;
	for i in $(SUBDIRS) ; do \
	( cd $$i ; $(MAKE) $(MACHINE) MACHINE="../$(MACHINE)" ) || exit 1; \
	done

	$(MAKE) $(OBJDIR)/InitNetwork.o  $(OBJDIR)/util.o

	cd ../tmp; /bin/rm -f moduleinit*.o; $(CHARMC) *.o $(OTHER) $(LIBS) -o $(PGM) $(POSEMODULE) -ltrace-projections -language charm++ 

$(OBJDIR)/InitNetwork.o: InitNetwork.C
	$(CHARMC) -o $@ -seq InitNetwork.C -I../Main -I./

$(OBJDIR)/util.o: util.C
	$(CHARMC) -o $@ -seq util.C


docs: ../HiSim.doxy ../BlueGene/InitNetwork.C ../BlueGene/util.C ../HypCubeArch/InitNetwork.C ../HypCubeArch/util.C ../HypCubeArchOB/InitNetwork.C ../HypCubeArchOB/util.C ../IB/InitNetwork.C ../IB/util.C ../InputVcSelection/RoundRobin.C ../InputVcSelection/SLQ.C ../InputVcSelection/SLQ_Switch.C ../InputVcSelection/outputBufferIn.C ../Main/BgSim.C ../Main/Channel.C ../Main/Nic.C ../Main/Switch.C ../Main/TCsim.C ../Main/pgm.C ../Main/TCsim_sim.C ../Main/BgSim_sim.C ../OB/InitNetwork.C ../OB/util.C ../OutputVcSelection/maxAvailBuffer.C ../OutputVcSelection/maxAvailBufferBubbleVc.C ../OutputVcSelection/maxAvailBufferSwitch.C ../OutputVcSelection/outputBuffer.C ../RedStorm/InitNetwork.C ../Routing/DirectionOrdered3D.C ../Routing/HammingDistance.C ../Routing/TorusRouting.C ../Routing/UpDown.C ../Topology/FatTree.C ../Topology/HyperCube.C ../Topology/Mesh3D.C ../BlueGene/InitNetwork.h ../BlueGene/util.h ../HypCubeArch/InitNetwork.h ../HypCubeArch/util.h ../HypCubeArchOB/InitNetwork.h ../HypCubeArchOB/util.h ../IB/InitNetwork.h ../IB/util.h ../InputVcSelection/MainInputVcSelection.h ../InputVcSelection/RoundRobin.h ../InputVcSelection/SLQ.h ../InputVcSelection/SLQ_Switch.h ../InputVcSelection/outputBufferIn.h ../Main/BgSim.h ../Main/TCsim.h ../Main/pgm.h ../Main/TCsim_sim.h ../Main/TCsim.decl.h ../Main/TCsim.def.h ../Main/BgSim_sim.h ../Main/BgSim.decl.h ../Main/BgSim.def.h ../Main/Pgm.decl.h ../Main/Pgm.def.h ../OB/InitNetwork.h ../OB/util.h ../OutputVcSelection/MainOutputVcSelection.h ../OutputVcSelection/maxAvailBuffer.h ../OutputVcSelection/maxAvailBufferBubbleVc.h ../OutputVcSelection/maxAvailBufferSwitch.h ../OutputVcSelection/outputBuffer.h ../RedStorm/InitNetwork.h ../Routing/DirectionOrdered3D.h ../Routing/HammingDistance.h ../Routing/MainRouting.h ../Routing/TorusRouting.h ../Routing/UpDown.h ../Topology/FatTree.h ../Topology/HyperCube.h ../Topology/MainTopology.h ../Topology/Mesh3D.h ../Main/BgSim.ci ../Main/TCsim_sim.ci ../Main/TCsim.ci ../Main/pgm.ci ../Main/BgSim_sim.ci
	doxygen ../HiSim.doxy

clean:
	@echo
	cd ..;
	for i in $(SUBDIRS) ; do \
	( cd $$i ; $(MAKE) clean ) ; \
	done
	-cd ../tmp; rm -f *.o; rm -f charmrun; rm -f module*; rm -f $(PGM)
	rm -f *.o
