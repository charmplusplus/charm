CHARM_BASE_DIR = ../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

OBJS = main.o jacobi.o
GENERAL_DEPENDS = Makefile

default : all
all : jacobi


################################################################################
## General Charm++ Stuff

# Rule to combine all the object files
jacobi : $(OBJS) $(GENERAL_DEPENDS)
	$(CHARMC) -language charm++ -o jacobi $(OBJS)

# Rule to compile the Main class
main.o : main.C main.h jacobi_config.h main.decl.h jacobi.decl.h $(GENERAL_DEPENDS)
	$(CHARMC) -c -o main.o main.C

# Rule to compile the Jacobi class
jacobi.o : jacobi.C jacobi.h jacobi_config.h jacobi.decl.h main.decl.h $(GENERAL_DEPENDS)
	$(CHARMC) -c -o jacobi.o jacobi.C

# General rule for creating decl and def files
%.decl.h : %.ci $(GENERAL_DEPENDS)
	$(CHARMC) $*.ci


################################################################################
## Clean-up Stuff

clean :
	$(RM) *.decl.h *.def.h *.o jacobi_shared jacobi charmrun *genSPECode*
