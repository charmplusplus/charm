/*
  liveVizPoll: application-polling image-assembly interface.

Jonathan A. Booth, jbooth@uiuc.edu
*/
#include "liveViz.h"
#include "liveViz_impl.h"

CProxy_liveVizPollArray proxy;
extern CProxy_liveVizPollArray proxy;
void liveVizPoll0Get(liveVizRequestMsg *req);

// A shadow array to handle reductions as well as to deal with image synch.
class liveVizPollArray : public ArrayElementMax {
private:
  CkMsgQ<liveVizPollRequestMsg> requests;
public:
  liveVizPollArray() : requests() { usesAtSync = CmiTrue; };
  liveVizPollArray(CkMigrateMessage *m) : requests() { usesAtSync = CmiTrue; };
  void isAtSync() { AtSync(); };
  void pup(PUP::er &p) {
    ArrayElementMax::pup(p);
    p|requests;
  }
  void request(liveVizPollRequestMsg *msg) {
    requests.enq(msg);
  }
  liveVizPollRequestMsg * poll(double timestep) {
    if ( ! requests.length() ) {
      return NULL;
    } else {
      return requests.deq();
    }
  }
};


// A listener to create a fully shadowed array. I have a feeling that this
// should probably be globalized into a library somewhere eventually rather
// than kept local to liveViz as it seems like a useful tool.
class liveVizArrayListener : public CkArrayListener {
public:
  liveVizArrayListener() : CkArrayListener(0) {};
  void ckBeginInserting() {
  }
  void ckEndInserting() {
    proxy.doneInserting();
  }
  void ckElementCreating(ArrayElement *elt) {
    proxy(elt->thisIndexMax).insert();
  }
};


CkComponent *liveVizGroup::ckLookupComponent(int userIndex) {
	if ( userIndex == 4321 ) {
		return new liveVizArrayListener();
	} else {
		return IrrGroup::ckLookupComponent(userIndex);
	}
}


//Called by clients to start liveViz.
void liveVizPollInit(const liveVizConfig &cfg,
		     CkArrayOptions &opts,
		     liveVizPollMode mode)
{
  if (CkMyPe()!=0) CkAbort("liveVizInit must be called only on processor 0!");

  // Create our proxy array, who will buffer incoming requests and make them
  // available to the user's code. Tell the user's array to bindTo my proxy.
  proxy = CProxy_liveVizPollArray::ckNew();
  opts.bindTo(proxy);

  // Now throw that all down onto the regular old liveViz
  CkCallback myCB((void (*) (void *))liveVizPoll0Get);
  liveVizInit(cfg, proxy, myCB);

  // Create our array listener object who'll manage our proxy array.
  CkComponentID listener(lvG, 4321);
  opts.addListener(listener);
}

// Called by clients to look and see if they need to do image work
liveVizPollRequestMsg *liveVizPoll(ArrayElement *from, double timestep) {
  liveVizPollArray *p = proxy(from->thisIndexMax).ckLocal();
  if ( !p ) {
#ifdef DEBUG
    ckerr << "liveViz warning: liveVizPoll(): proxy at from->thisIndexMax is null!" << endl;
#endif
    return NULL;
  } else {
    return p->poll(timestep);
  }
}

// Called to sync the polling element
void liveVizPollSync(ArrayElement *from) {
  liveVizPollArray *p = proxy(from->thisIndexMax).ckLocal();
  if ( p ) { p->isAtSync(); }
}


//Called by lower layers when an image request comes in on processor 0.
//  Just forwards request on to user.
void liveVizPoll0Get(liveVizRequestMsg *msg) {
  proxy.request(new liveVizPollRequestMsg(msg->req));
  delete msg;
}

//Called by clients to deposit a piece of the final image
void liveVizPollDeposit(ArrayElement *client,
                    double timestep,
		    const liveVizRequest &req,
		    int startx, int starty, 
		    int sizex, int sizey, const byte * src)
{
  if (lv_config.getVerbose(2))
    CkPrintf("liveVizDeposit> Deposited image at (%d,%d), (%d x %d) pixels, on pe %d\n",
    	startx,starty,sizex,sizey,CkMyPe());

//Allocate a reductionMessage:
  CkRect r(startx,starty, startx+sizex,starty+sizey);
  r=r.getIntersect(CkRect(req.wid,req.ht)); //Never copy out-of-bounds regions
  if (r.isEmpty()) r.zero();
  int bpp=lv_config.getBytesPerPixel();
  byte *dest;
  CkReductionMsg *msg=allocateImageMsg(req,r,&dest);
  
//Copy our image into the reductionMessage:
  if (!r.isEmpty()) {
    //We can't just copy image with memcpy, because we may be clipping user data here:
    CkImage srcImage(sizex,sizey,bpp,(byte *)src);
    srcImage.window(r.getShift(-startx,-starty)); //Portion of src overlapping dest
    CkImage destImage(r.wid(),r.ht(),bpp,dest);
    destImage.put(0,0,srcImage);
  }

//Contribute this image to the reduction
  msg->setCallback(CkCallback(vizReductionHandler));

  liveVizPollArray *p = proxy(client->thisIndexMax).ckLocal();
  if ( !p ) {
    CkError("LiveViz error: somehow an element has been created who has"
            " no corresponding proxy member!");
  }
  p->contribute(msg);
}

#include "liveVizPoll.def.h"

