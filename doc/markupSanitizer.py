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

# Assuming, tt tags are not spewed recklessly by latex2html,
# replace them with code tags
for t in soup('tt'):
    t.wrap( soup.new_tag('code') )
    t.unwrap()

# Rewrap all div class=alltt blocks in pre tags
for d in soup('div','alltt'):
    d.wrap( soup.new_tag('pre') )
    d.unwrap()

# Remove br tags required within pre sections
for p in soup('pre'):
    for b in p('br'):
        b.extract()

# Print cleaned up markup to stdout
print( soup.prettify(formatter="html") )

