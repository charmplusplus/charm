WEBDIR	= /www/manuals

all: html pdf

pdf: check-latex
	@echo "Building PDF documentation"
	sphinx-build -b latex -n . pdf
	make -C pdf
	@echo "PDF documentation built. Open ./pdf/charm.pdf in a PDF viewer."

html: check-sphinx
	@echo "Building HTML documentation"
	sphinx-build -n . html
	@echo "HTML documentation built. Open ./html/index.html in a web browser."

# This rule was used to copy the manual to our local web server
# web: all
#	-rm -rf $(WEBDIR)/backup
#	mkdir -p $(WEBDIR)/backup && \
#	mv $(WEBDIR)/pdf $(WEBDIR)/html $(WEBDIR)/backup
#	cp -r pdf html $(WEBDIR) && \
#	chgrp -R kale $(WEBDIR)/pdf $(WEBDIR)/html && \
#	chmod -R g+w  $(WEBDIR)/pdf $(WEBDIR)/html || \
#	{ rm -rf $(WEBDIR)/pdf $(WEBDIR)/html && \
#	mv $(WEBDIR)/backup/pdf $(WEBDIR)/backup/html $(WEBDIR); }
#	-rm -rf $(WEBDIR)/backup

clean:
	rm -rf pdf html

check-sphinx:
	@which -s sphinx-build || { echo "Sphinx (http://www.sphinx-doc.org) not found. Install it with 'pip install sphinx'."; exit 1; }

check-latex: check-sphinx
	@which -s pdflatex || { echo "Warning: Pdflatex not found. Exiting."; exit 1; }
