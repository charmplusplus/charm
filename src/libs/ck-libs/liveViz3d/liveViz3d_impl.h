/*
  On-the-wire formats for various ccs requests used by sixty.
*/
#ifndef __UIUC_CHARM_VP_CCS_H
#define __UIUC_CHARM_VP_CCS_H

#include "pup.h"
#include "viewpoint.h"

/*
"lv3d_getViews" handler:
	Sent by the client to request the actual images
of any updated object views.

Outgoing request is a single integer, the client ID
Incoming response is a set of updated CkViews:
    int n; p|n;
    for (int i=0;i<n;i++) p|view[i];
*/


/*
"lv3d_newViewpoint" handler:
	Sent by the client when his viewpoint changes,
to request updated views of the objects.

Outgoing request is a ccsNewViewpoint, below.
There is no response.
*/
class liveViz3dNewViewpoint {
public:
	int clientID; //Unique identifier for this client
	CkViewpoint vp; //Viewpoint of request
	
	liveViz3dNewViewpoint(void) {
	}
	void pup(PUP::er &p) {
		p|clientID;
		p|vp;
	}
};
PUPmarshall(liveViz3dNewViewpoint);

#endif
