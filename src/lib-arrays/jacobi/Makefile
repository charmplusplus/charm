CHARMC=/home/brunner/projects/charm/net-hp-cc/bin/charmc -O -I$(AHOME)
#AHOME=/home/brunner/projects/Arrays/arrays
AHOME=../arrays

OBJS = 	jacobi.o RRMap.o\
	$(AHOME)/Array1D.o $(AHOME)/ArrayMap.o $(AHOME)/ArrayElement.o

CIFILES = jacobi.ci RRMap.ci 

all:	jacobi

jacobi:	$(OBJS)
	$(CHARMC) -language charm++ -o jacobi $(OBJS)

cifiles:
	$(CHARMC)  $(CIFILES)

clean:
	rm -f *.o ljdyn

jacobi.o: jacobi.C
	$(CHARMC) -c jacobi.C

RRMap.o: RRMap.C RRMap.h
	$(CHARMC) -c RRMap.C


