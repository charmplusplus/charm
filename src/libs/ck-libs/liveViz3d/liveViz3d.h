/*
  Interface to server portion of the sixty library.
*/
#ifndef __UIUC_CHARM_LIVEVIZ3D_H
#define __UIUC_CHARM_LIVEVIZ3D_H

#include "pup.h"
#include "viewpoint.h"
#include "viewable.h"
#include "ckvector3d.h"
#include "liveViz3d_impl.h"

template <class T>
class pupPtrHolder {
	T *ptr;
public:
	pupPtrHolder(T *p_) :ptr(p_) {}
	pupPtrHolder(void) :ptr(0) {}
	pupPtrHolder(const pupPtrHolder<T> &p_) :ptr(p_.ptr) {}
	
	T *release(void) {
		T *ret=ptr;
		ptr=0;
		return ret;
	}
	
	void pup(PUP::er &p) {
		if (!ptr) ptr=new T;
		p|(*ptr);
	}
	inline friend void operator|(PUP::er &p,pupPtrHolder<T> &v) {v.pup(p);}
};

#include "liveViz3d.decl.h" //For liveViz3dRequestMsg

/*
Register for libsixty redraw requests.  This routine
must be called exactly once on processor 0.
The callback will be executed whenever the user updates
their viewpoint--it will be called with a liveViz3dRequestMsg.
*/
void liveViz3dInit(const CkBbox3d &box,CkCallback incomingRequest);


class liveViz3dRequestMsg : public CMessage_liveViz3dRequestMsg {
public:
	liveViz3dNewViewpoint nv;
};

/*
This object should live on every viewable: its 
"handleRequest" method should be called every time
the user changes their viewpoint.  You must implement
a subclass of this class that implements the "view" 
method to draw yourself.
*/
class liveViz3dViewableImpl;
class liveViz3dViewable : public CkViewable {
	CkViewableID id;
	CkInterestSet univPoints;
	liveViz3dViewableImpl *impl;
public:
	liveViz3dViewable(void);
	~liveViz3dViewable();
	
	void setID(const CkViewableID &id_) {id=id_;}
	void setUnivPoints(const CkInterestSet &univPoints_) {univPoints=univPoints_;}
	virtual const CkViewableID &getViewableID(void) {return id;}
	virtual const CkInterestSet &getInterestPoints(void) {return univPoints;}
	
	virtual void handleRequest(liveViz3dRequestMsg *m);
	
	virtual void view(const CkViewpoint &vp,CkImage &dest) =0;
	
	virtual void pup(PUP::er &p);
};
PUPmarshall(liveViz3dViewable);

#endif
