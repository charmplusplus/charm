-include ../../common.mk
FFTW_HOME=/bgl/local/fftw-2.1.5/
OPTS=-g -O3 

CHARMC=../../../bin/charmc $(OPTS)
INCLUDES=-I$(FFTW_HOME)/include -I$(CHARMBASE)/include/fftlib \
	 -DFFTW_ENABLE_FLOAT=1

LIBS=-module PencilFFT -L$(FFTW_HOME)/lib -language charm++ -lsfftw -lsrfftw

OBJS = testpme.o patch.o

all:	cifiles testpme

testpme: $(OBJS)
	$(CHARMC) -language charm++ -o testpme $(OBJS) ${LIBS}

cifiles: testpme.ci
	$(CHARMC)  testpme.ci

clean:
	rm -f *~ *.decl.h *.def.h conv-host *.o testpme charmrun

testpme.o: testpme.C
	$(CHARMC) -c testpme.C ${INCLUDES}

patch.o: patch.C
	$(CHARMC) -c patch.C ${INCLUDES}

test: all
	@echo "Intra-processor TestPME.."
	$(call run, ./testpme +p1)
	@echo "Inter-processor TestPME.."
	$(call run, ./testpme +p2)

bgtest: all
	@echo "Intra-processor TestPME.."
	$(call run, ./testpme +p1 +x1 +y1 +z1)
	@echo "Inter-processor TestPME.."
	$(call run, ./testpme +p2 +x2 +y1 +z1)

