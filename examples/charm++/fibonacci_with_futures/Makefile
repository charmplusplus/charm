# Variables
CHARMC = /Users/stevenqie/software/charm/netlrts-darwin-arm8/bin/charmc
TARGET = fibonacci                       
SRCS = fib_futures.ci fib_futures.C          

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CHARMC) fib_futures.ci             
	$(CHARMC) -c fib_futures.C    
	$(CHARMC) -o $(TARGET) fib_futures.o 

clean:
	rm -f *.o $(TARGET) fib.decl.h fib.def.h charmrun 

.PHONY: all clean