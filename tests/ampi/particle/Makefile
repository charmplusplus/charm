-include ./../../common.mk
CHARMC = ./../../../bin/ampicxx $(OPTS) $(MOPTS)

OBJS = pCodeAMPI.o

all: pCode

pCode: $(OBJS)
	$(CHARMC) -o pCode $(OBJS)

pCodeAMPI.o:
	$(CHARMC) -c particleCode.cpp -o pCodeAMPI.o

clean:
	-rm -f *.o pCode $(OBJS)

test: pCode
	$(call run, ./pCode +p1 +vp1 )
	$(call run, ./pCode +p2 +vp27 )
