# Stub makefile for LaTeX PPL manual
FILE=manual
TEX=$(FILE).tex 
DEST=debugger
LATEX2HTML=$(L2H) -split 5
DOCTITLE='Charm++ Debugger Manual'
PROJECT_LINK='<a href="http://charm.cs.uiuc.edu/research/debugger">CharmDebug Homepage</a><br>'

FIG_TARGET=build_eps_fig
FIG_CLEAN=figs/*.eps $(FIG_TARGET)

include ../Makefile.common

build_eps_fig:
	cd figs; ./convert.sh
	touch build_eps_fig
