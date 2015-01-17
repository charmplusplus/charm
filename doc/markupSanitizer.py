#!/usr/bin/python

from bs4 import BeautifulSoup
import sys
import os

if sys.version < '3':
    import codecs
    def u(x):
        return codecs.unicode_escape_decode(x)[0]
else:
    def u(x):
        return x

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

# Remove br and span tags from within pre sections
for p in soup('pre'):
    for b in p('br'):
        b.extract()
    for s in p('span'):
        s.unwrap()

# Remove all useless class 'arabic' spans
for s in soup('span','arabic'):
    s.unwrap()

# Extract the navigation bar
navmenu = soup.find('div', 'navigation')
if navmenu:
    navmenu.extract()

# Wrap the remaining contents within a div
if not soup.find('div', id='maincontainer'):
    soup.body['id'] = 'maincontainer'
    soup.body.name = 'div'
    soup.find('div', id='maincontainer').wrap( soup.new_tag('body') )

if navmenu:
    # If this navmenu doesn't already have a TOC, insert one
    if not navmenu.find('ul','manual-toc'):
        # Add a toc within the navmenu
        navmenuTOC = BeautifulSoup(open("tmp-navmenu.html"), "lxml")
        navmenuTOC = navmenuTOC.find('ul','manual-toc').extract()
        navmenuTOC.append( BeautifulSoup("".join([
        '<li><a href="http://charm.cs.illinois.edu">PPL Homepage</a></li>',
        '<li><a href="http://charm.cs.illinois.edu/help">Other Manuals</a></li>'])
        ) )
        navmenu.append(navmenuTOC)

    # Insert navigation symbols to prev and next links
    prevsymbol = soup.new_tag('span')
    prevsymbol['class'] = 'navsymbol'
    prevsymbol.string = u('\xab')
    prv = navmenu.find('li',id='nav-prev')
    if prv:
        prv.find('a').insert(0, prevsymbol)

    nextsymbol = soup.new_tag('span')
    nextsymbol['class'] = 'navsymbol'
    nextsymbol.string = u('\xbb')
    nxt = navmenu.find('li',id='nav-next')
    if nxt:
        nxt.find('a').append(nextsymbol)

    # Reinsert the navigation bar at the end
    soup.body.append(navmenu)

# Extract the title
titl = soup.find('title')

# Replace the head section with the user-supplied head markup
soup.find('head').extract()
newhead = BeautifulSoup(open("../assets/head.html"), "lxml")
newhead = newhead.find('head').extract()
newhead.append(titl)
soup.html.body.insert_before(newhead)

# Print cleaned up markup to stdout
print( soup.prettify(formatter="html") )

