# Modify CHAMRDIR to point to your charm installation!
#
# Actually we need the include directory, so we can make sure
# we correctly load the PUPed NetFEM file format used by your 
# programs
#
CHARMDIR = ../../../../..

#Target Executable:
DEST = NetFEM_To_Paraview

CC   = g++
FLAGS = -O -w

# Shouldn't need to modify anything else below
CHARMINC=-I$(CHARMDIR)/include
CHARMLIB=-L$(CHARMDIR)/lib -lconv-util

OBJS = 	NetFEM_To_Paraview.o 								\
		ckstandalone.o 										\
		netfem_update_vtk.o 								\
		$(CHARMDIR)/tmp/libs/ck-libs/netfem/netfem_data.o

all: $(DEST)

ckstandalone.o : $(CHARMDIR)/tmp/ckstandalone.C
	$(CC) -c $(CHARMDIR)/tmp/ckstandalone.C -o ckstandalone.o $(FLAGS) $(CHARMINC)

$(DEST): $(OBJS)
	$(CC) $(OBJS) -o $(DEST) $(FLAGS) $(CHARMINC) $(CHARMLIB)

clean:
	rm -rf $(DEST) *.o *~

.cpp.o: $(HEADERS)
	$(CC) -c $< $(FLAGS) $(CHARMINC)


