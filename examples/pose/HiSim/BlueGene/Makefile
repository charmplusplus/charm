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

all:  sqtest
	@echo
	test -d ../tmp || mkdir ../tmp;
	for i in $(SUBDIRS) ; do \
	( cd $$i ; $(MAKE) $(MACHINE) MACHINE="../$(MACHINE)" ) || exit 1; \
	done

	make $(OBJDIR)/InitNetwork.o  $(OBJDIR)/util.o

	cd ../tmp; /bin/rm -f moduleinit.o; $(CHARMC) *.o $(OTHER) $(LIBS) -o $(PGM) -module pose -ltrace-projections -language charm++ 

$(OBJDIR)/InitNetwork.o: InitNetwork.C
	$(CHARMC) -o $@ -seq InitNetwork.C -I../Main -I./

$(OBJDIR)/util.o: util.C
	$(CHARMC) -o $@ -seq util.C


sqtest:
ifeq ($(SEQUENTIAL),1)
	cp  $(CHARMBASE)/lib/libmoduleseqpose.a $(CHARMBASE)/lib/libmodulepose.a
	cp  $(CHARMBASE)/lib_so/libmoduleseqpose.so $(CHARMBASE)/lib_so/libmodulepose.so
else
	cp  $(CHARMBASE)/lib/libmoduleorigpose.a $(CHARMBASE)/lib/libmodulepose.a
	cp  $(CHARMBASE)/lib_so/libmoduleorigpose.so $(CHARMBASE)/lib_so/libmodulepose.so
endif

clean:
	@echo
	for i in $(SUBDIRS) ; do \
	( cd $$i ; $(MAKE) clean ) ; \
	done
	-cd ../tmp; rm -f *.o; rm -f charmrun; rm -f module*; rm -f $(PGM)
	rm -f *.o
