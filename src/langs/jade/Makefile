# Compiles libjade.a and puts it into place
# Compiles *.g files, which requires ANTLR
#  Puts resulting .java files into place in charm/java/charm/jade/

CDIR=../../..
#Antlr jar files needed for jade user
AUDIR = $(CDIR)/bin/charmjavabin/antlr.jar
#Antlr jar files needed for jade developer
ADDIR = $(CDIR)/bin/charmjavabin/antlralld.jar

CHARMC=$(CDIR)/bin/charmc $(OPTS)
CP=/bin/cp
MV=/bin/mv

LIB = libjade.a
LIBOBJ = JArray.o
HEADERS = JArray.h jade.h
LIBDEST =  $(CDIR)/lib/

JAVADEST = $(CDIR)/java/charm/jade

all: c l

all2: c l p

default:
	@echo "Specify target: l(ib) p(arsers) c(clean)"

clean: cleanlib cleanparsers

l: lib

c: clean

p: parsers

#================================================================
lib: $(LIBDEST)

$(LIBDEST) : cleanlib $(LIB)
	$(MV) $(LIB) $(LIBDEST)
	$(CP) $(HEADERS) $(CDIR)/include

$(LIB): $(LIBOBJ)
	$(CHARMC) -o $(LIB) $(LIBOBJ) 

JArray.o: JArray.h JArray.C
	$(CHARMC) -c JArray.C

cleanlib:
	rm -f *.o $(LIB)

#================================================================

parsers: cleanparsers t0 t1 t2 t3

#Call Antlr to compile a .g file
Antlr = java -classpath $(ADDIR) antlr.Tool

#The Java parser
t0:
	$(Antlr) java.g
	$(CP) JavaRecognizer.java JavaTokenTypes.java JavaLexer.java $(JAVADEST)
#	$(javacA) JavaRecognizer.java JavaTokenTypes.java JavaLexer.java

#The first pass tree parser
t1:
	$(Antlr) java.tree1.g
	$(CP) JavaTreeParser1.java JavaTreeParser1TokenTypes.java $(JAVADEST)
#	$(javacA) JavaTreeParser1.java JavaTreeParser1TokenTypes.java

#The second pass tree parser
t2:
	$(Antlr) java.tree2.g
	$(CP) JavaTreeParser2.java JavaTreeParser2TokenTypes.java $(JAVADEST)
#	$(javacA) JavaTreeParser2.java JavaTreeParser2TokenTypes.java

t3:
	$(Antlr) java.tree.g
	$(CP) JavaTreeParser.java JavaTreeParserTokenTypes.java $(JAVADEST)
#	$(javacA) JavaTreeParser.java JavaTreeParserTokenTypes.java

cleanparsers:
	rm -f JavaLexer.java JavaTokenTypes.java JavaTokenTypes.txt JavaRecognizer.java
	rm -f JavaTreeParser1.java JavaTreeParser1TokenTypes.java JavaTreeParser1TokenTypes.txt
	rm -f JavaTreeParser2.java JavaTreeParser2TokenTypes.java JavaTreeParser2TokenTypes.txt
	rm -f JavaTreeParser.java JavaTreeParserTokenTypes.java JavaTreeParserTokenTypes.txt
