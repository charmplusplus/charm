# Stub makefile for LaTeX PPL manual
FILE=manual
TEX=$(FILE).tex ccs.tex cmi.tex conditions.tex cpvmacros.tex isomalloc.tex \
	index.tex queue.tex simulator.tex threads.tex thsync.tex usermain.tex \
	onesided.tex persistent.tex
DEST=converse
LATEX2HTML=$(L2H) -split 5
DOCTITLE='Converse Programming Manual'
PROJECT_LINK='<a href="http://charm.cs.uiuc.edu/research/converse">Converse Homepage</a><br>'

include ../Makefile.common

index.tex:
	touch index.tex
