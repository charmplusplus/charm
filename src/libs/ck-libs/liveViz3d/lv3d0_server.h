/**
  Interface to server portion of the 
  simplest (level 0) liveViz3d library.

  This defines a CCS stream of outgoing CkView's,
  headed to the client machine.

  Orion Sky Lawlor, olawlor@acm.org, 2003
*/
#ifndef __UIUC_CHARM_LV3D0_SERVER_H
#define __UIUC_CHARM_LV3D0_SERVER_H

#include "lv3d0.h" /* shared client and server classes */
#include "lv3d0.decl.h" /* for message superclass */

/// This message is sent every time a client viewpoint changes.
class LV3D_ViewpointMsg : public CMessage_LV3D_ViewpointMsg {
public:
	int clientID; //Unique identifier for this client
	int frameID; //0 is first frame, then incrementing
	CkViewpoint viewpoint; //Viewpoint of request
};

/**
  Represents a set of impostors available for updates.
*/
class LV3D_ServerMgr {
public:
	virtual ~LV3D_ServerMgr();
	virtual void newClient(int clientID) =0;
	virtual void newViewpoint(LV3D_ViewpointMsg *m) =0;
	virtual void doBalance(void) =0;
};

/**
  Set up to accept LiveViz3D requests.  
  This routine must be called exactly once from processor 0
  at startup.
  \param clientUniverse Universe to pass to client.
  \param frameUpdate Callback to call (with a LV3D_ViewpointMsg) when a
     client viewpoint changes.
 */
void LV3D0_Init(LV3D_Universe *clientUniverse,LV3D_ServerMgr *mgr);

/**
  Send this view from the server to this client.
  This routine can be called on any processor.
  You must set the view's id and prio fields.
 */
void LV3D0_Deposit(CkView *v,int clientID);


#endif
