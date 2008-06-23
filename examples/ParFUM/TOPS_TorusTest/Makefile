
CHARMC=../../../bin/charmc $(OPTS)
FLAGS=-g

all : pgm

.PHONY: pgm
pgm : main.o 
	$(CHARMC) -language ParFUM_TOPS -o pgm main.o $(FLAGS)  -language ParFUM -module netfem

main.o : main.C
	$(CHARMC) -c main.C $(FLAGS)

clean :
	rm -rf *.o pgm a.out charmrun *disp*.dat
