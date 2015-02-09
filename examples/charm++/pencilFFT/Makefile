-include ../../common.mk
#Set FFTW_HOME home to the fftw-2.1.5 local installation

#FFTW_HOME=/bgl/local/fftw-2.1.5/
FFTW_HOME=/usr
OPTS=-g -O3 

CHARMC=../../../bin/charmc $(OPTS)
INCLUDES=-I$(FFTW_HOME)/include -I$(CHARMBASE)/include/fftlib \
	 -DFFTW_ENABLE_FLOAT=1 -DUSE_FFTW_DECLS=1

LIBS=-module PencilFFT -L$(FFTW_HOME)/lib -language charm++ -lsfftw -lsrfftw

OBJS = testpencil.o 

all:	cifiles testpencil

testpencil: $(OBJS)
	$(CHARMC) -language charm++ -o testpencil $(OBJS) ${LIBS}

cifiles: testpencil.ci
	$(CHARMC)  testpencil.ci

clean:
	rm -f *~ *.decl.h *.def.h conv-host *.o testpencil charmrun

testpencil.o: testpencil.C
	$(CHARMC) -c testpencil.C ${INCLUDES}

test: all
	@echo "Intra-processor Testpencil.."
	$(call run, ./testpencil +p1)
	@echo "Inter-processor Testpencil.."
	$(call run, ./testpencil +p2)

bgtest: all
	@echo "Intra-processor Testpencil.."
	$(call run, ./testpencil +p1 +x1 +y1 +z1)
	@echo "Inter-processor Testpencil.."
	$(call run, ./testpencil +p2 +x2 +y1 +z1)

