/**
  Interface to server portion of a more complicated
  interface to the liveViz3d library-- it defines a 
  single array of objects, each of which represent a 
  single CkViewable.

  Orion Sky Lawlor, olawlor@acm.org, 2003
*/
#ifndef __UIUC_CHARM_LV3D_SERVER_1_H
#define __UIUC_CHARM_LV3D_SERVER_1_H

#include "lv3d0_server.h"

#define LV3D_USE_FLAT 1 /* include link to liveViz 2D image assembly */
#if LV3D_USE_FLAT
#include "liveViz.h"
#else
class liveVizRequestMsg;
#endif

#include "lv3d1.decl.h" /* for message superclasses */

/**  Manager for a LV3D_Array. */
class LV3D1_ServerMgr : public LV3D_ServerMgr {
	CProxy_LV3D_Array a;
public:
	
	LV3D1_ServerMgr(const CProxy_LV3D_Array &a_) :a(a_) {}
	virtual void newClient(int clientID) {
		a.LV3D_NewClient(clientID);
	}
	virtual void newViewpoint(LV3D_ViewpointMsg *m) {
		a.LV3D_Viewpoint(m);
	}
	virtual void doBalance(void);
};

/**
 We'll be running lv3d on this array-- do any needed magic.
 Currently changes the map to stay away from PE 0
 (which does CCS interfacing).
*/
void LV3D1_Attach(CkArrayOptions &opts);


/**
 * Register for lv3d redraw requests.  This routine
 * must be called exactly once on processor 0 of the server.
 * The array, which must inherit from LV3D_Array,
 * will be used for all communication.
 */
void LV3D1_Init(CkArrayID LV3D_ArrayID,LV3D_Universe *theUniverse=0);


/**
  This message is used to prioritize rendering requests.
  It's always allocated using its "new_" method, so a 
   priority is always attached.
 FIXME: add ability to thread a single render request,
   so rendering can suspend in the middle.
 FIXME: handle requestID wraparound.
*/
class LV3D_RenderMsg : public CMessage_LV3D_RenderMsg {
public:
	int clientID; //Unique identifier for this client
	int frameID; //0 is first frame, then incrementing
	int viewableID; //Viewable to render
	
	/** Allocate a prioritized Render message for this viewpoint.
	  prioAdj is a double-precision scaling value for priority.
	*/
	static LV3D_RenderMsg *new_(int client,int frame,int viewable,double prioAdj);
	static void delete_(LV3D_RenderMsg *m);
};


class impl_LV3D_Array;

/**
 This array holds all the visible objects in one liveViz3d
 computation.  Users should inherit all their visible classes
 from this kind of array element.
 
Rationale:
 By keeping all viewables in one big array, we can update 
 the viewpoint using a single broadcast.  The sensible alternative
 of keeping several different arrays (e.g., one array of tiles,
 one of buildings, one of trees) would result in multiple broadcasts,
 and make addressing more complicated.
*/
class LV3D_Array : public CBase_LV3D_Array {
	impl_LV3D_Array *impl;
	void init();
public:
	LV3D_Array(void) {init();}
	LV3D_Array(CkMigrateMessage *m) :CBase_LV3D_Array(m) {init();}
	
	/**
	 Add this viewable to our set.  The viewable is still
	 owned by the caller, but must survive until removed or
	 the element is destroyed.
	 
	 This routine is often called exactly one per array 
	 element lifetime, and often from the constructor 
	 or pup routine.
	 
	 For liveViz, the array element basically only exists 
	 to provide network access for this CkViewable.
	 
	 There is a small fixed limit on the number of CkViewables 
	 you can add.  1 is safe.
	*/
	void addViewable(CkViewable *v);
	void removeViewable(CkViewable *v);
	
	virtual void pup(PUP::er &p);
	~LV3D_Array();
	
// Network-called methods:
	/**
	  This request is broadcast every time a client connects.
	 */
	virtual void LV3D_NewClient(int clientID);
	
	/**
	  Prepare to handle a call.  Can be used to add viewables lazily.
	*/
	virtual void LV3D_Prepare(void);
	
	/**
	  Participate in load balancing.
	*/
	virtual void LV3D_DoBalance(void);
	
	/**
	  This request is broadcast every time a client viewpoint changes.
	  Internally, it asks the stored CkViewables if they should redraw,
	  and if so, queues up a LV3DRenderMsg.
	 */
	virtual void LV3D_Viewpoint(LV3D_ViewpointMsg *m);
	
	/**
	  This method is used to prioritize rendering.
	*/
	virtual void LV3D_Render(LV3D_RenderMsg *m);
	
	/**
	  This entry method is only used when rendering to
	  plain old server-assembled liveViz 2d.
	*/
	virtual void LV3D_FlatRender(liveVizRequestMsg *m);
};

#endif
