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
	./bin/charjc --debug --verbose --stdout tests/Hello.cj


# The lousy antlrv3 ant task won't give proper output from antlr errors, so
# use these horrible direct invocations instead
debug-charj:
	java -classpath /Users/abecker/Documents/workspace/charm/src/langs/charj/lib/antlr-2.7.7.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/antlr-3.1b1.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/antlr-runtime-3.1b1.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/stringtemplate-3.1.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/gunit-1.0.2.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/JSAP-2.1.jar org.antlr.Tool -o /Users/abecker/Documents/workspace/charm/src/langs/charj/build/src/charj/translator -lib /Users/abecker/Documents/workspace/charm/src/langs/charj/build/src/charj/translator /Users/abecker/Documents/workspace/charm/src/langs/charj/build/src/charj/translator/Charj.g
debug-emitter:
	java -classpath /Users/abecker/Documents/workspace/charm/src/langs/charj/lib/antlr-2.7.7.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/antlr-3.1b1.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/antlr-runtime-3.1b1.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/stringtemplate-3.1.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/gunit-1.0.2.jar:/Users/abecker/Documents/workspace/charm/src/langs/charj/lib/JSAP-2.1.jar org.antlr.Tool -o /Users/abecker/Documents/workspace/charm/src/langs/charj/build/src/charj/translator -lib /Users/abecker/Documents/workspace/charm/src/langs/charj/build/src/charj/translator /Users/abecker/Documents/workspace/charm/src/langs/charj/build/src/charj/translator/CharjEmitter.g
