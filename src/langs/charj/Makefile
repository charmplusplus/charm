
ANT := ant -lib lib/ant-antlr3.jar
CLASSPATH := lib/antlr-3.2.jar:lib/JSAP-2.1.jar
SRCDIR := src/charj/translator
BUILDDIR := build/src/charj/translator
ANTLR := java -classpath $(CLASSPATH) org.antlr.Tool -o $(BUILDDIR) -lib $(BUILDDIR)
ANTLRFLAGS := -Xmultithreaded

SRC_GRAMMARS := $(wildcard $(SRCDIR)/*.g)
BUILD_GRAMMARS := $(subst $(SRCDIR),$(BUILDDIR),$(SRC_GRAMMARS:.g=.d))

.PHONY: all build compile grammars clean doc
all: build grammars
	$(ANT) dist

build:
	@echo "\ninit:"
	@mkdir -p $(BUILDDIR)
	@cp -p $(SRCDIR)/* $(BUILDDIR)
	@mkdir -p build/classes
	@mkdir -p build/classes/META-INF
	@mkdir -p build/lib
	@mkdir -p docs
	@echo "\nantlr:"

compile: $(wildcard $(BUILDDIR)/*.java)
	$(ANT) compile

clean:
	$(ANT) clean

doc:
	$(ANT) doc

grammars: build $(BUILD_GRAMMARS)

$(BUILDDIR)/%.d: $(SRCDIR)/%.g $(SRCDIR)/Charj.g
	@echo "\n    [antlr] $(<F)"
	$(ANTLR) $(ANTLRFLAGS) $(BUILDDIR)/$(<F)
	@touch $@

