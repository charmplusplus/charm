/*
  Interface to simplest (level 0) liveViz3d library.
  This file is shared between client and server.

  Orion Sky Lawlor, olawlor@acm.org, 2004/4/3
*/
#ifndef __UIUC_CHARM_LV3D0_H
#define __UIUC_CHARM_LV3D0_H

#include "pup.h"
#include "ckvector3d.h"
#include "ckviewpoint.h"
#include "ckviewable.h"
#if CMK_LIVEVIZ3D_CLIENT
/* Client side */
#  include "ogl/main.h"
#else
/* Server side */
#define CMK_LIVEVIZ3D_SERVER 1
#endif

/// Private class that stores object data for the universe client side.
class LV3D_Universe_Table;

/**
 This is the class that represents the entire 
 viewable domain. It performs all necessary drawing
 every frame.  This class is sent from the server 
 to the client in response to a "setup" message.
*/
class LV3D_Universe : public PUP::able {
public:
// CLIENT AND SERVER
	LV3D_Universe() 
		:object_table(0) {}
	LV3D_Universe(CkMigrateMessage *m) 
		:PUP::able(m), object_table(0) {}
	virtual ~LV3D_Universe();
	
	/**
	 Pup routine is called on both client (for unpacking)
	 and server (for packing, and for checkpointing).
	*/
	virtual void pup(PUP::er &p);
	
	/**
	  All subclasses must be PUP::able, so they can be sent to the 
	  client properly.
	*/
	PUPable_decl(LV3D_Universe);
	
// CLIENT ONLY
#ifdef CMK_LIVEVIZ3D_CLIENT
	/**
	  Set up this client.  Should set any needed drawing options.
	  Default does nothing, which gives you no lighting,
	  no depth, with alpha, and clear blue background color.
	*/
	virtual void setupClient(oglOptions &i);
	
	/// Post-OpenGL initialization.
	virtual void setupGL(void);
	
	/**
	  Return the client GUI controller.  Default is a trackball,
	  centered on the middle of the unit cube.
	*/
	virtual oglController *makeController(void);

	/**
	  Add this view to the universe.  This is called
	  once per incoming network view on the client.
	  This call transfers ownership of the view.
	  The default implementation stores the view in the table below.
	*/
	virtual void viewResponse(CkView *v);
	
	/**
	  Draw the world to this camera using OpenGL calls.  
	  This routine is called once per frame on the client.
	  The default implementation just draws all known views.
	*/
	virtual void render(const CkViewpoint &vp);

protected:
	/// Return the last known view for this viewableID.
	///   Returns NULL if no view is available.
	CkView *lookup(const CkViewableID &src);
#endif
private:
	/// Stores cached CkView's on the client side.
	LV3D_Universe_Table *object_table;

};


#endif
