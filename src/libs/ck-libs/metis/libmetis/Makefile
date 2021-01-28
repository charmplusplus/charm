# This file was added to metis by Kavitha Chandrasekar at UIUC
# in order to skip ./configure step and directly make the metis library

CDIR=../../../../..
LIBDIR=$(CDIR)/lib
INCDIR=$(CDIR)/include
CC=$(CDIR)/bin/charmc -I. -I../GKlib $(OPTS)
LD = $(CC)
RANLIB = ar -ts

OBJS = auxapi.o balance.o bucketsort.o checkgraph.o \
	coarsen.o compress.o contig.o debug.o fm.o fortran.o \
	frename.o gklib.o graph.o initpart.o kmetis.o kwayfm.o \
	kwayrefine.o mcutil.o mesh.o meshpart.o minconn.o mincover.o \
	mmd.o ometis.o options.o parmetis.o pmetis.o refine.o \
	separator.o sfm.o srefine.o stat.o timing.o util.o wspace.o

.o:
	$(CC) -c $*.c

DEST=$(LIBDIR)/libckmetis.a

GKlib = ../GKlib/*.o

all: $(DEST)
	cp metis.h $(INCDIR)/

$(DEST): $(OBJS)
	$(CC)  -o $@ $(OBJS) $(GKlib)

clean:
	rm -f *.o 
