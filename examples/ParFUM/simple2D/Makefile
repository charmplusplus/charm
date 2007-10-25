OPTS += -I..  
CHARMC = charmc 

SRCS = simple2D.C \
       cst_nl.C \
       main.C \
       ../ampi_split.C

OBJS = $(SRCS:.C=.o)

all: simple2D

simple2D: $(OBJS)
	$(CHARMC) -o simple2D $(OBJS) \
	    -language ampi -module ParFUM -module netfem -module MetisLB

%.o: %.C
	$(CHARMC) $(OPTS) -c $< -o $@


test: simple2D
	./charmrun ./simple2D +vp4 +p2

bgtest: simple2D
	./charmrun ./simple2D +vp4 +p2 +x2 +y2 +z1

.PHONY: clean
clean:
	rm -f simple2D fsimple2D *.o conv-host charmrun ../ampi_split.o
