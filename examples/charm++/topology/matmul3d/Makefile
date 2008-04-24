CHARMC=../../../../bin/charmc $(OPTS)

MATH_LIB        = -lesslbg -lesslsmpbg -lxlf90_r  \
		  -lmass -lmassv -lxlfmath -lxlomp_ser -lxlsmp -lpthread

LOCAL_LIBS      = -L/soft/apps/LAPACK \
		  -L/opt/ibmcmp/xlf/bg/11.1/bglib \
		  -L/opt/ibmcmp/xlsmp/bg/1.7/bglib \
		  -L/bgsys/ibm_essl/sles10/prod/opt/ibmmath/lib \
		  -L/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib \
		  -L/gpfs/home/bohm/zlib/lib

OBJS = matmul3d.o

all: matmul3d

matmul3d: $(OBJS)
	$(CHARMC) -O3 -language charm++ $(LOCAL_LIBS) $(MATH_LIB) -o matmul3d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -O3 -language charm++ -tracemode projections -lz -o matmul3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -O3 -language charm++ -tracemode summary -lz -o matmul3d.sum $(OBJS)

matmul3d.decl.h: matmul3d.ci
	$(CHARMC)  matmul3d.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o matmul3d matmul3d.prj charmrun *~

matmul3d.o: matmul3d.C matmul3d.h matmul3d.decl.h
	$(CHARMC) -c -O3 -I/bgsys/ibm_essl/sles10/prod/opt/ibmmath/include matmul3d.C
