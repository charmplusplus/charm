
ANT := ant -lib lib/ant-antlr3.jar
CLASSPATH := lib/antlr-3.2.jar:lib/JSAP-2.1.jar
SRCDIR := src/charj/translator
BUILDDIR := build/src/charj/translator
ANTLRC := java -classpath $(CLASSPATH) org.antlr.Tool
ANTLRFLAGS := -Xmultithreaded -o $(BUILDDIR) -lib $(BUILDDIR)

SRC_GRAMMARS := $(wildcard $(SRCDIR)/*.g)
BUILD_GRAMMARS := $(subst $(SRCDIR),$(BUILDDIR),$(SRC_GRAMMARS:.g=.d))

.PHONY: all build grammars clean doc
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

clean:
	$(ANT) clean

doc:
	$(ANT) doc

grammars: build $(BUILD_GRAMMARS)

# All grammar files depend on the Charj.g being compiled first,
# so that the token list is available.
$(BUILDDIR)/Charj.d: $(SRCDIR)/Charj.g
	@echo "\n    [antlr] $(<F)"
	$(ANTLRC) $(ANTLRFLAGS) $(BUILDDIR)/$(<F)
	@touch $@

$(BUILDDIR)/%.d: $(SRCDIR)/%.g $(BUILDDIR)/Charj.d
	@echo "\n    [antlr] $(<F)"
	$(ANTLRC) $(ANTLRFLAGS) $(BUILDDIR)/$(<F)
	@touch $@

