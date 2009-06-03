CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc -I. $(OPTS)
MPICC=$(CDIR)/bin/ampiCC -I. $(OPTS)

# Directory where Magic software headers live:
MGC=.
MGCOBJS=$(MGC)/MgcIntr3DTetrTetr.o  $(MGC)/MgcMath.o  $(MGC)/MgcPlane.o \
	$(MGC)/MgcTetrahedron.o  $(MGC)/MgcVector3.o

HEADERS=paralleltransfer.h tetmesh.h prismMesh.h
# HEADERS=paralleltransfer.h parallelsurfacetransfer.h tetmesh.h prismMesh.h triSurfMesh.h mapbasic.h
HEADDEP=$(HEADERS) 
TOBJS=$(MGCOBJS) GenericElement.o transfer.o tetmesh.o tetmesh_noboite.o 
# TOBJS=$(MGCOBJS) GenericElement.o transfer.o tetmesh.o tetmesh_noboite.o geom_util.o
OBJS=$(TOBJS) tetmesh_fem.o paralleltransfer.o prismMesh.o
#OBJS=$(TOBJS) tetmesh_fem.o paralleltransfer.o prismMesh.o parallelsurfacetransfer.o
LIB=libdatatransfer

LIBDIR=$(CDIR)/lib
DEST=$(LIBDIR)/$(LIB).a

test1=test_GenericElement
test1o=$(test1).C GenericElement.o
test2=test_noboite
test2o=$(test2).C $(TOBJS) charm_standalone.o
test3=test_transfer
test3o=$(test3).C $(TOBJS) charm_standalone.o
test4=test_vol
test4o=$(test4).C cg3d.o \
	test_vol_mgc.o $(MGCOBJS) \
	test_vol_planes.o volume_planes.o
tests=$(test1) $(test2) $(test3) $(test4)

all: $(DEST)

$(DEST): $(OBJS) headers
	$(CHARMC) $(OBJS) -o $@ $(LIBS)

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

$(test1): $(test1o) $(HEADDEP)
	$(CHARMC) $(test1o) -o $(test1) -seq -language c++

$(test2): $(test2o) $(HEADDEP)
	$(CHARMC) $(test2o) -o $(test2) -seq -language c++

$(test3): $(test3o) $(HEADDEP)
	$(CHARMC) $(test3o) -o $(test3) -seq -language c++

$(test4): $(test4o) $(HEADDEP)
	$(CHARMC) $(test4o) -o $(test4) -language ampi

test: $(tests)
	./$(test1) 1 10000
	./$(test2) test_transfer_dir/src.noboite
	./$(test3) test_transfer_dir/src.noboite test_transfer_dir/src.noboite
	./$(test3) test_transfer_dir/dest.noboite test_transfer_dir/dest.noboite
	./$(test3) test_transfer_dir/src.noboite test_transfer_dir/dest.noboite
	./$(test3) test_transfer_dir/dest.noboite test_transfer_dir/src.noboite

clean:
	rm -rf *.a *.def.h *.decl.h *.o $(MGC)/*.o \
		SunWS_cache $(DEST) $(tests) headers

# Trick make into compiling .C's into .o's.
.SUFFIXES: .C .cpp

.C.o:
	$(MPICC) -c $<

.cpp.o:
	$(CHARMC) -c $< 
