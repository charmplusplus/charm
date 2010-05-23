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

# The lousy antlrv3 ant task won't give proper output from antlr errors, so
# use these horrible direct invocations instead
.PHONY: debug debug-charj debug-semantics debug-emitter
debug: debug-charj debug-semantics debug-emitter

CLASSPATH = lib/antlr-3.2.jar:lib/JSAP-2.1.jar
debug-charj:
	cp src/charj/translator/Charj.g build/src/charj/translator/Charj.g
	java -classpath $(CLASSPATH) org.antlr.Tool \
	    -o build/src/charj/translator -lib build/src/charj/translator \
	    build/src/charj/translator/Charj.g
debug-semantics:
	cp src/charj/translator/CharjSemantics.g build/src/charj/translator/CharjSemantics.g	
	java -classpath $(CLASSPATH) org.antlr.Tool \
	    -o build/src/charj/translator -lib build/src/charj/translator \
	    build/src/charj/translator/CharjSemantics.g
debug-emitter:
	cp src/charj/translator/CharjEmitter.g build/src/charj/translator/CharjEmitter.g
	java -classpath $(CLASSPATH) org.antlr.Tool -o build/src/charj/translator \
	    -lib build/src/charj/translator build/src/charj/translator/CharjEmitter.g

# For any unrecognized rules, try to compile a test program with that name
%:
	./bin/charjc --debug --verbose --AST --stdout --translate-only tests/$@/$@.cj
