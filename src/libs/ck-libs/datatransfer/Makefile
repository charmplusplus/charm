CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)

# Directory where Magic software headers live:
MGC=mgc
MGCOBJS=$(MGC)/MgcIntr3DTetrTetr.o  $(MGC)/MgcMath.o  $(MGC)/MgcPlane.o \
	$(MGC)/MgcTetrahedron.o  $(MGC)/MgcVector3.o

HEADERS=
# datatransfer.h
HEADDEP=$(HEADERS) 
TOBJS=$(MGCOBJS) transfer.o tetmesh.o tetmesh_noboite.o
OBJS=$(TOBJS) tetmesh_fem.o paralleltransfer.o pgm.o
DEST=pgm
LIBS=-module collide -module fem -language ampi
#$(CDIR)/lib/libmoduledatatransfer.a

test1=test_cg3d
test1o=$(test1).C $(TOBJS) charm_standalone.o
test2=test_noboite
test2o=$(test2).C $(TOBJS) charm_standalone.o
test3=test_transfer
test3o=$(test3).C $(TOBJS) charm_standalone.o
tests=$(test1) $(test2) $(test3)

all: $(DEST) test

$(DEST): $(OBJS) $(COMPAT)
	$(CHARMC) $(OBJS) $(COMPAT) -o $@ $(LIBS)

headers: $(HEADERS)
	cp $(HEADERS) $(CDIR)/include/
	touch headers

$(test1): $(test1o) $(HEADDEP)
	$(CHARMC) $(test1o) -o $(test1) -seq -language c++

$(test2): $(test2o) $(HEADDEP)
	$(CHARMC) $(test2o) -o $(test2) -seq -language c++

$(test3): $(test3o) $(HEADDEP)
	$(CHARMC) $(test3o) -o $(test3) -seq -language c++

test: $(tests)
	./$(test1) 1 10000
	./$(test2) test_noboite_dir/simple.noboite
	./$(test3) test_transfer_dir/src.noboite test_transfer_dir/src.noboite
	./$(test3) test_transfer_dir/dest.noboite test_transfer_dir/dest.noboite
	./$(test3) test_transfer_dir/src.noboite test_transfer_dir/dest.noboite
	./$(test3) test_transfer_dir/dest.noboite test_transfer_dir/src.noboite

clean:
	rm -rf *.a *.def.h *.decl.h *.o $(MGC)/*.o \
		SunWS_cache $(DEST) $(tests) headers

# Trick gmake into compiling .C's into .o's.
%.o: %.C
	$(CHARMC) -c $< -o $@

%.o: %.cpp
	$(CHARMC) -c $< -o $@

# Trick other makes into compiling .C's into .o's.
.SUFFIXES: .C .cpp

.C.o:
	$(CHARMC) -c $< -o $@

.cpp.o:
	$(CHARMC) -c $< -o $@
