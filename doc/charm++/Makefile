# Stub makefile for LaTeX PPL manual
FILE=manual
TEX=$(FILE).tex arrays.tex callbacks.tex chares.tex commlib.tex delegation.tex \
	entry.tex groups.tex inhertmplt.tex index.tex intro.tex \
	helloworld.tex utilities.tex loadb.tex advancedlb.tex futures.tex \
	marshalling.tex messages.tex modules.tex nodegroups.tex othercalls.tex \
	overview.tex pup.tex advancedpup.tex quiesce.tex readonly.tex reductions.tex sdag.tex \
	python.tex alltoall.tex history.tex \
	../projections/tracing.tex
DEST=charm++
LATEX2HTML=$(L2H) -split 5
DOCTITLE = 'Charm++ Manual'

include ../Makefile.common

index.tex:
	touch index.tex
