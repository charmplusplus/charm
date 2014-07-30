include ../common.mk

LIB = libmoduleCkSmpCoordination.a
LIBOBJ = CkSmpCoordinator.o

HEADERS = $(CDIR)/include/CkSmpCoordination.decl.h \
          $(CDIR)/include/CkSmpCoordination.def.h \
          $(CDIR)/include/CkSmpCoordinationMsg.h \
          $(CDIR)/include/CkSmpCoordinator.h 
LIBDEST =  $(LIBDIR)/$(LIB)

CIFILES = CkSmpCoordination.ci

all: $(LIBDEST) $(HEADERS)

$(LIBDEST): $(LIBOBJ)
	$(CHARMC) -o $(LIBDEST) $(LIBOBJ) 

CkSmpCoordination.def.h CkSmpCoordination.decl.h: INTERFACE

INTERFACE: $(CIFILES)
	$(CHARMC) -c $<
	touch $@

CkSmpCoordinator.o: CkSmpCoordinator.cpp $(HEADERS)
	$(CHARMC) -I../../.. -I$(HASHTABLE_MT_DIR) -c -o $@ $< 

clean:
	rm -f conv-host *.o *.decl.h *.def.h core  $(LIB) INTERFACE

realclean: clean
	rm -f $(LIBDEST) $(HEADERS)
