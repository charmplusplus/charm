ANT=ant -lib lib/ant-antlr3.jar

# Ant does the actual building, just invoke it
.DEFAULT_GOAL:dist
dist:
	$(ANT) dist
compile:
	$(ANT) compile
clean:
	$(ANT) clean
doc:
	$(ANT) doc
all:
	$(ANT) all

test:
	./bin/charjc tests/hello.cj

