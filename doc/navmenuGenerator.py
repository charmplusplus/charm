#!/usr/bin/python

from bs4 import BeautifulSoup
import sys
import os

# Accept filename as user input
argc = len( sys.argv )
if (argc < 2): raise Exception
fileName = sys.argv[1];

# Construct a DOM object
soup = BeautifulSoup(open(fileName), "lxml")

# Get just the table of contents from the index page
toc = soup.find("ul","ChildLinks").extract()

# Retain only part and chapter titles
for sctn in toc.select("li > ul > li > ul"):
    sctn.extract()

# Discard all br tags
for b in toc("br"):
    b.extract()

# Setup classes etc
toc['class'] = "manual-toc"

# Print cleaned up markup to stdout
print( toc.prettify(formatter="html") )


