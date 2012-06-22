
CHARJC=../../bin/charjc
#CJFLAGS=--debug --verbose --AST --stdout --translate-only
#CJFLAGS=--translate-only --AST
CJFLAGS=--translate-only

.PHONY: all clean jacobi chunk
all: jacobi chunk

jacobi:
	$(CHARJC) $(CJFLAGS) jacobi.cj

chunk:
	$(CHARJC) $(CJFLAGS) chunk.cj

clean:
	rm -rf *.gen *.h *.o charmrun jacobi jacobi_pre.cj 
