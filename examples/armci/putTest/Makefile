CHARMC=../../../bin/charmc $(OPTS)

all: put

put: put.c
	$(CHARMC) -o put put.c -language armci 

test: put
	./charmrun +p2 ./put

clean:
	rm -f put charmrun
