-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS = ataTest.o

all: ataTest

ataTest: $(OBJS)
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_COMPILER" = "bgxlc" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	$(CHARMC) -language charm++  -o ataTest -module NDMeshStreamer $(OBJS)

ataTest.decl.h: ataTest.ci
	$(CHARMC)  ataTest.ci

test: all
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_COMPILER" = "bgxlc" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	$(call run, +p4 ./ataTest )

clean:
	rm -f *.decl.h *.def.h conv-host *.o ataTest charmrun

ataTest.o: ataTest.C ataTest.decl.h
	CHARMINC=../../../include . ../../../tmp/conv-config.sh ; \
	if [ "$$CMK_COMPILER" = "bgxlc" ] ; \
	then \
	  echo "Skipping test whose compilation will fail under BG XLC"; exit 0 ; \
	fi ; \
	$(CHARMC) -c ataTest.C
