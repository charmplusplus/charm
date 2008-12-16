CHARJC=../../../bin/charjc

objects := $(patsubst %.cj,%.o,$(wildcard *.cj))
all: $(objects)
%.o: %.cj
	$(CHARJC) --debug --verbose --AST --stdout $<
