/*
  Interface to server portion of the liveViz3d library.
*/
#ifndef __UIUC_CHARM_LIVEVIZ3D_H
#define __UIUC_CHARM_LIVEVIZ3D_H

#include "pup.h"
#include "ckvector3d.h"
#include "ckviewpoint.h"
#include "ckviewable.h"

class CkViewHolder {
	typedef CkView T;
	T *ptr;
public:
	CkViewHolder(T *p_) :ptr(p_) {}
	CkViewHolder(void) :ptr(0) {}
	CkViewHolder(const CkViewHolder &p_) :ptr(p_.ptr) {}
	
	T *release(void) {
		T *ret=ptr;
		ptr=0;
		return ret;
	}
	
	void pup(PUP::er &p) {
		if (p.isUnpacking()) 
			ptr=pup_unpack(p);
		else
			pup_pack(p,*ptr);
	}
	inline friend void operator|(PUP::er &p,CkViewHolder &v) {v.pup(p);}
};

#include "liveViz3d.decl.h" //For liveViz3dRequestMsg

/**
 * Register for libsixty redraw requests.  This routine
 * must be called exactly once on processor 0.
 * The callback will be executed whenever the user updates
 * their viewpoint--it will be called with a liveViz3dRequestMsg.
 */
void liveViz3dInit(const CkBbox3d &box,CkCallback incomingRequest);

class liveViz3dRequestMsg : public CMessage_liveViz3dRequestMsg {
public:
	int clientID; //Unique identifier for this client
	CkViewpoint vp; //Viewpoint of request
};


/**
 * Call this routine with each viewable each time the
 * incomingRequest callback is executed.  
 * Can be called on any processor.
 * Be sure to delete the message after all the calls.
 */
void liveViz3dHandleRequest(const liveViz3dRequestMsg *m,CkViewable &v);

#endif
