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
PGM = HypCubeSim
OTHER = -lconv-bluegene-logs
LIBS = -Lstdc++ -lm
MACHINE = HypCubeArch

all:
	@echo
	for i in $(SUBDIRS) ; do \
	( cd $$i ; make $(MACHINE) MACHINE="../$(MACHINE)" ) || exit 1; \
	done

	$(CHARMC) -o InitNetwork.o -seq InitNetwork.C -I../Main -I./
	$(CHARMC) -o util.o -seq util.C
	test -d ../tmp || mkdir ../tmp; cp *.o ../tmp

	cd ../tmp; /bin/rm -f moduleinit.o; $(CHARMC) *.o $(OTHER) $(LIBS) -o $(PGM) -module pose -ltrace-projections -language charm++ -module commlib

clean:
	@echo
	for i in $(SUBDIRS) ; do \
	( cd $$i ; make clean ) ; \
	done
	cd ../tmp; rm -f *.o; rm -f charmrun; rm -f module*; rm -f $(PGM)
	rm -f *.o
