/*
  liveVizPoll: application-polling image-assembly interface.

  Original Author: Jonathan A. Booth, jbooth@uiuc.edu
*/
#include "liveViz.h"
#include "liveViz_impl.h"

static CProxy_liveVizPollArray proxy;
void liveVizPoll0Forward(liveVizRequestMsg *req);

/**
 A shadow array to handle reductions as well as to deal with image synch.
 
 Incoming requests are broadcast, queued up, then given to the user on demand.
*/
class liveVizPollArray : public ArrayElementMax {
private:
  CkMsgQ<liveVizRequestMsg> requests;
  void init(void) {
    proxy=thisArrayID;
  }
public:
  liveVizPollArray() { init(); }
  liveVizPollArray(CkMigrateMessage *m) { init(); }
  void pup(PUP::er &p) {
    ArrayElementMax::pup(p);
    p|requests;
  }
  
  /// Incoming request: queue it up.
  void request(liveVizRequestMsg *msg) {
    requests.enq(msg);
  }
  
  /// Hand any pending requests over to the user.
  liveVizRequestMsg * poll(double timestep) {
    if ( ! requests.length() ) {
      return NULL;
    } else {
      return requests.deq();
    }
  }

  /// Redefined poll(), since it doesn't even use the timestep parameter
  liveVizRequestMsg * poll(void) {
  	return poll(0.0);
  }

};


/// A listener to create a fully shadowed array. I have a feeling that this
/// should probably be globalized into a library somewhere eventually rather
/// than kept local to liveViz as it seems like a useful tool.
class liveVizArrayListener : public CkArrayListener {
  CProxy_liveVizPollArray poll;
public:
  liveVizArrayListener(const CProxy_liveVizPollArray &poll_) 
  	: CkArrayListener(0),poll(poll_) {}
  liveVizArrayListener(CkMigrateMessage *m) : CkArrayListener(m) {}
  void pup(PUP::er &p) {
     CkArrayListener::pup(p);
     p|poll;
  }
  PUPable_decl(liveVizArrayListener)

  void ckBeginInserting() {
  }
  void ckEndInserting() {
    poll.doneInserting();
  }
  void ckElementCreating(ArrayElement *elt) {
    poll(elt->thisIndexMax).insert();
  }
};


//Called by clients to start liveViz.
void liveVizPollInit(const liveVizConfig &cfg,
		     CkArrayOptions &opts,
		     liveVizPollMode mode)
{
  if (CkMyPe()!=0) CkAbort("liveVizInit must be called only on processor 0!");

  // Create our proxy array, who will buffer incoming requests and make them
  // available to the user's code. Tell the user's array to bindTo my proxy.
  CProxy_liveVizPollArray poll = CProxy_liveVizPollArray::ckNew();
  opts.bindTo(poll);

  // Incoming image requests get sent to the poll array
  CkCallback myCB(CkIndex_liveVizPollArray::request(0), poll);
  liveVizInit(cfg, poll, myCB);

  // Create our array listener object who'll manage our proxy array.
  opts.addListener(new liveVizArrayListener(poll));
}

// Called by clients to look and see if they need to do image work
liveVizRequestMsg *liveVizPoll(ArrayElement *from, double timestep) {
  liveVizPollArray *p = proxy(from->thisIndexMax).ckLocal();
  if ( !p ) {
#ifdef DEBUG
    CkError("liveViz warning: liveVizPoll(): proxy at from->thisIndexMax is null!\n");
#endif
    return NULL;
  } else {
    return p->poll(timestep);
  }
}

// Same, but without the unused timestep parameter
liveVizRequestMsg *liveVizPoll(ArrayElement *from) {
  return liveVizPoll(from,0.0);
}


//Called by clients to deposit a piece of the final image
void liveVizPollDeposit(ArrayElement *client,
                    double timestep,
		    const liveVizRequest &req,
		    int startx, int starty, 
		    int sizex, int sizey, const byte * src,
		    liveVizCombine_t combine
		    )
{
  liveVizPollArray *p = proxy(client->thisIndexMax).ckLocal();
  if ( !p ) {
    CkError("LiveViz error: somehow an element has been created who has"
            " no corresponding proxy member!");
  }
  liveVizDeposit(req,startx,starty,sizex,sizey,src,p,combine);
}


#include "liveVizPoll.def.h"

