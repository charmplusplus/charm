/*
  Data types, function prototypes,  etc. exported by liveViz.
  This layer does image assembly, and is the most commonly-used 
  interface to liveViz.
 */
#ifndef __UIUC_CHARM_LIVEVIZ_H
#define __UIUC_CHARM_LIVEVIZ_H

#include "liveViz0.h"
#include "liveViz.decl.h"

/*
  Start liveViz.  This routine should be called once on processor 0
  when you are ready to begin receiving image requests.
  
  The arrayID is the array that will make deposits into liveViz.
  
  The callback is signalled each time a client requests an image.
  The image parameters are passed in as a "liveVizRequestMsg *" message.
*/
void liveVizInit(const liveVizConfig &cfg, CkArrayID a, CkCallback c);

class liveVizRequestMsg : public CMessage_liveVizRequestMsg {
public:
	liveVizRequest3d req;
	liveVizRequestMsg() {}
	liveVizRequestMsg(const liveVizRequest3d &req_) :req(req_) {}
};

/*
  Deposit a (sizex x sizey) pixel portion of the final image,
  starting at pixel (startx,starty) in the final image.
  The "client" pointer is used to perform reductions, it's 
  normally "this".  Each array element must call deposit, even
  if it's just an empty deposit, like:
  	liveVizDeposit(0,0, 0,0, NULL, this);
*/
void liveVizDeposit(const liveVizRequest &req,
		    int startx, int starty, 
		    int sizex, int sizey, const byte * imageData,
		    ArrayElement* client);

//As above, but taking a message instead of a request:
inline void liveVizDeposit(liveVizRequestMsg *reqMsg,
		    int startx, int starty, 
		    int sizex, int sizey, const byte * imageData,
		    ArrayElement* client)
{
	liveVizDeposit(reqMsg->req,startx,starty,sizex,sizey,imageData,client);
	delete reqMsg;
}

#endif /* def(thisHeader) */
