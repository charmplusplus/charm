/*
Prototypes for CCS/visualization interaction routines, 
the lowest level of liveViz. (liveViz level 0)

Orion Sky Lawlor, olawlor@acm.org, 6/11/2002
*/
#ifndef __UIUC_CHARM_LIVEVIZ0_H
#define __UIUC_CHARM_LIVEVIZ0_H

#include "conv-ccs.h"

typedef unsigned char byte;
#include "ckvector3d.h"

struct liveVizRequest {
	CcsDelayedReply replyToken;
	int code; //Application-defined request type
	int wid,ht;//Size (pixels) of requested image
};

struct liveVizRequest3d : public liveVizRequest {
	CkVector3d x,y,z,o; //Coordinate axes & origin of screen
	double minZ,maxZ; //Range of Z values to display
};

/*
  The client just requested an image as described above.
  You must implement this routine, which will be called from a single processor.
*/
void liveViz0Get(const liveVizRequest3d &req);

/*
  Each call to liveViz, above, must eventually result in a 
  call to this routine, which delivers an entire, finished image 
  to the client.
*/
void liveViz0Deposit(const liveVizRequest &req,byte * imageData);


/*
  Begin responding to image requests under the given configuration.
  liveVizStartCCS should be called exactly once on processor 0 when
  your program is ready to begin processing image requests.
*/
class liveVizConfig {
	//If true, expect a color (3-byte/pixel) image
	//  instead of a greyscale (1 byte/pixel) image
	bool isColor; 
	
	/*If true, the server produces images regardless of whether the client
	  asks for them; this is true for animating and batch-mode images.
	  If false, the server will respond to client requests; this is true for
	  viz-on-demand services.
	  The client repeatedly asks for an image if this is set; if not set
	  it only asks when the client's window size or viewpoint changes.
	 */
	bool serverPush;
	
	bool is3d; //If true, show 3d camera motion tools
	CkBbox3d box; //If 3d, the viewed object's bounding box, universe coords
	
	int verbose; //If nonzero, printf status info; higher numbers give more prints
public:
	liveVizConfig() {isColor=false; serverPush=false; is3d=false; verbose=1;}
	
	//2D constructor:
	liveVizConfig(bool isColor_,bool serverPush_) {
		isColor=isColor_;
		serverPush=serverPush_;
		is3d=false;
		verbose=0;
	}
	
	//3D constructor:
	liveVizConfig(bool isColor_,bool serverPush_,const CkBbox3d &box_) {
		isColor=isColor_;
		serverPush=serverPush_;
		is3d=true;
		box=box_;
		verbose=0;
	}
	
	//Increment the verbosity level
	void moreVerbose(void) {verbose++;}
	
	//Extract the configuration:
	bool getColor(void) const {return isColor;}
	int getBytesPerPixel(void) const {return isColor?3:1;}
	bool getPush(void) const {return serverPush;}
	bool get3d(void) const {return is3d;}
	const CkBbox3d &getBox(void) const {return box;}
	bool getVerbose(int thanLevel) const {return verbose>=thanLevel;}
};
void liveViz0Init(const liveVizConfig &cfg);

#endif /* def(thisHeader) */

