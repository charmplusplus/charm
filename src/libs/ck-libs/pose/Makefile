# Default charmc options 
# - changes to be passed in via simulation makefiles 
# **************************************************
CDIR=../../../..
CHARMC=$(CDIR)/bin/charmc $(OPTS)
POSEBINDIR =  $(CDIR)/bin/
POSEINCDIR =  $(CDIR)/include/pose

# POSE object definitions - DO NOT MODIFY
#***********************************
OBJECTS=	eventID.o stats.o pose.o srtable.o cancel.o evmpool.o \
		pvtobj.o lbObject.o ldbal.o gvt.o event.o eqheap.o \
		evq.o sim.o rep.o strat.o con.o opt.o adapt.o
LIB = libpose.a
LIBDEST =  $(CDIR)/lib/$(LIB)

# POSE LIBRARY GENERATION RULES -
# DO NOT MODIFY
# *******************************
$(LIBDEST):	$(OBJECTS)
	$(CHARMC) -o $(LIBDEST) $(OBJECTS) 
	[ ! -d $(POSEINCDIR) ] && mkdir $(POSEINCDIR) || true
	/bin/cp -f  *.h $(POSEINCDIR)
	/bin/cp -f  etrans.pl $(POSEBINDIR)

# POSE COMPILATION RULES 
# - DO NOT MODIFY
#***********************
chpt.o : chpt.C chpt.h
	$(CHARMC) -c chpt.C

adapt.o : adapt.C adapt.h 
	$(CHARMC) -c adapt.C

opt.o : opt.C opt.h 
	$(CHARMC) -c opt.C

con.o : con.C con.h 
	$(CHARMC) -c con.C

strat.o : strat.C sim.h
	$(CHARMC) -c strat.C

rep.o : rep.C rep.h
	$(CHARMC) -c rep.C

sim.o : sim.C sim.h sim.def.h sim.decl.h
	$(CHARMC) -c sim.C

sim.def.h sim.decl.h : sim.ci evmpool.def.h
	$(CHARMC) sim.ci

evq.o : evq.C sim.h sim.def.h
	$(CHARMC) -c evq.C

eqheap.o : eqheap.C eqheap.h
	$(CHARMC) -c eqheap.C

gvt.o : gvt.C gvt.h gvt.def.h gvt.decl.h stats.def.h
	$(CHARMC) -c gvt.C

gvt.def.h gvt.decl.h : gvt.ci
	$(CHARMC) gvt.ci

ldbal.o : ldbal.C ldbal.h ldbal.def.h ldbal.decl.h stats.def.h
	$(CHARMC) -c ldbal.C

ldbal.def.h ldbal.decl.h : ldbal.ci
	$(CHARMC) ldbal.ci

pose.o : pose.C pose.h sim.def.h sim.decl.h pose.def.h pose.decl.h \
	stats.def.h stats.decl.h ldbal.def.h ldbal.decl.h \
	gvt.def.h gvt.decl.h
	$(CHARMC) -c pose.C

pose.def.h pose.decl.h : pose.ci
	$(CHARMC) pose.ci

cancel.o : cancel.C cancel.h 
	$(CHARMC) -c cancel.C

pvtobj.o : pvtobj.C pvtobj.h 
	$(CHARMC) -c pvtobj.C

lbObject.o : lbObject.C lbObject.h 
	$(CHARMC) -c lbObject.C

srtable.o : srtable.C srtable.h 
	$(CHARMC) -c srtable.C

stats.o : stats.C stats.h stats.def.h stats.decl.h gvt.def.h ldbal.def.h
	$(CHARMC) -c stats.C

stats.def.h stats.decl.h : stats.ci
	$(CHARMC) stats.ci

evmpool.o : evmpool.C evmpool.h evmpool.def.h evmpool.decl.h
	$(CHARMC) -c evmpool.C

evmpool.def.h evmpool.decl.h : evmpool.ci
	$(CHARMC) evmpool.ci

event.o : event.C event.h 
	$(CHARMC) -c event.C

eventID.o : eventID.C eventID.h 
	$(CHARMC) -c eventID.C

# HOUSE-KEEPING RULES
#********************
clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f libpose.a
