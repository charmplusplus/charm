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
debug-charj:
	cp src/charj/translator/Charj.g build/src/charj/translator/Charj.g
	java -classpath lib/antlr-2.7.7.jar:lib/antlr-3.1b1.jar:lib/antlr-runtime-3.1b1.jar:lib/stringtemplate-3.1.jar:lib/gunit-1.0.2.jar:lib/JSAP-2.1.jar org.antlr.Tool -o build/src/charj/translator -lib build/src/charj/translator build/src/charj/translator/Charj.g
debug-semantics:
	cp src/charj/translator/CharjSemantics.g build/src/charj/translator/CharjSemantics.g	
	java -classpath lib/antlr-2.7.7.jar:lib/antlr-3.1b1.jar:lib/antlr-runtime-3.1b1.jar:lib/stringtemplate-3.1.jar:lib/gunit-1.0.2.jar:lib/JSAP-2.1.jar org.antlr.Tool -o build/src/charj/translator -lib build/src/charj/translator build/src/charj/translator/CharjSemantics.g
debug-emitter:
	cp src/charj/translator/CharjEmitter.g build/src/charj/translator/CharjEmitter.g
	java -classpath lib/antlr-2.7.7.jar:lib/antlr-3.1b1.jar:lib/antlr-runtime-3.1b1.jar:lib/stringtemplate-3.1.jar:lib/gunit-1.0.2.jar:lib/JSAP-2.1.jar org.antlr.Tool -o build/src/charj/translator -lib build/src/charj/translator build/src/charj/translator/CharjEmitter.g

# For any unrecognized rules, try to compile a test program with that name
%:
	./bin/charjc --debug --verbose --AST --stdout tests/$@.cj
