/*
  liveVizPoll

  Original Author: Jonathan A. Booth, jbooth@uiuc.edu
  Latest   Author: Isaac Dooley 08/2005

  The general flow of data and function calls is this:

  element 1   calls liveVizPollDeposit() ---\
  element 2   calls liveVizPollDeposit() ----\
  element 3   calls liveVizPollDeposit() -----> combined using imagePollCombinerReducer()
  element ... calls liveVizPollDeposit() ----/
  element n   calls liveVizPollDeposit() ---/ 
 
                                             
  The combined image then is given to vizPollReductionHandler() on processor 0.
  It then calls liveVizOPollDeposit() which either queues up the image or 
  sends the image back to the client via a pending request in the request queue.

  We use a group to guarantee that processor zero will hold the queued images or requests

*/


#include "liveViz.h"
#include "liveViz_impl.h"
#include "ImageData.h"


/*readonly*/ CProxy_liveVizPollArray myGroupProxy;
CkReduction::reducerType poll_image_combine_reducer;
extern CkCallback clientGetImageCallback;


// A node group which handles enqueuing images on processor 0
class liveVizPollArray : public CBase_liveVizPollArray {
private:
  CkMsgQ<liveVizRequestMsg> requestQueue; // for unserviced client requests
  CkQ<imageUnit> imageQueue;      // for images not yet sent to client

  void init(){
	CkAssert(requestQueue.length()==0);	
	CkAssert(imageQueue.length()==0);
	CkPrintf("init()\n");
  }

public:

  liveVizPollArray() {init();}

  // Handle incoming request
  void request(liveVizRequestMsg *msg) {
	CkAssert(CkMyPe() == 0);
	if(imageQueue.length()>0){
	  imageUnit image = imageQueue.deq();
	  msg->req.wid = image.wid;
	  msg->req.ht = image.ht;
	  CkPrintf("wid=%d ht=%d\n", image.wid, image.ht);
	  liveViz0Deposit(msg->req, image.imageData);
	  delete msg;
	}
	else {
	  requestQueue.enq(msg);
	}
  }
  
  // Handle server generated image
  void liveVizPoll0Deposit(int wid, int ht, int bpp, int len, byte * imageData) {
	// if there are pending requests, send image to one of them
	if(requestQueue.length()>0){
	  liveVizRequestMsg *msg = requestQueue.deq();
	  msg->req.wid = wid;
	  msg->req.ht = ht;
	  CkPrintf("wid=%d ht=%d\n", wid, ht);
	  liveViz0Deposit(msg->req, imageData);	  
	}
	// else enqueue the server generated image
	else{
	  const imageUnit newimage(bpp,wid,ht,imageData);
	  imageQueue.enq(newimage);
	}
  }
  

};




/**
   Called by the reduction manager to combine all the source images
   received on one processor. This function adds all the image on one
   processor to one list of non-overlapping images. 
   
   Then the result is reduced onto processor zero in function
   vizPollReductionHandler()
*/
CkReductionMsg *imagePollCombineReducer(int nMsg,CkReductionMsg **msgs)
{
  if (nMsg==1) { //Don't bother copying if there's only one source
	CkReductionMsg *ret=msgs[0];
	msgs[0]=NULL; //Prevent reduction manager from double-delete
	return ret;
  }

  CkPrintf("Using hard-coded bpp=3\n");
  int bpp=3;
  ImageData imageData (bpp);
  
  CkReductionMsg* msg = CkReductionMsg::buildNew(imageData.CombineImageDataSize (nMsg,
																				 msgs),
												 NULL, poll_image_combine_reducer);
  
  imageData.CombineImageData(nMsg, msgs, (byte*)(msg->getData()));
  return msg;
}



/*
  Reduction handles an "image" from each processor. The result ends up here.
  Unpacks image, and passes it on to layer 0.
*/
void vizPollReductionHandler(void *r_msg)
{
  CkReductionMsg *msg = (CkReductionMsg*)r_msg;
  imageHeader *hdr=(imageHeader *)msg->getData();
  byte *srcData=sizeof(imageHeader)+(byte *)msg->getData();
  CkPrintf("Using hard-coded bpp=3\n");
  int bpp=3;
  CkRect destRect(0,0,hdr->req.wid,hdr->req.ht);
  int len = hdr->r.wid() * hdr->r.ht() * bpp;
  if (destRect==hdr->r) { //Client contributed entire image-- pass along unmodified
  	myGroupProxy[0].liveVizPoll0Deposit(hdr->r.wid(), hdr->r.ht(), bpp, len, srcData);
  }
  else { //Client didn't quite cover whole image-- have to pad
	CkImage src(hdr->r.wid(),hdr->r.ht(),bpp,srcData);
	CkAllocImage dest(hdr->req.wid,hdr->req.ht,bpp);
	dest.clear();
	dest.put(hdr->r.l,hdr->r.t,src);
	myGroupProxy[0].liveVizPoll0Deposit(hdr->r.wid(), hdr->r.ht(), bpp, len, dest.getData());
  }
  delete msg;
}




//Called by server objects to deposit a piece of the final image
void liveVizPollDeposit(ArrayElement *client,
						int startx, int starty, 
						int sizex, int sizey,             // The dimensions of the piece I'm depositing
						int imagewidth, int imageheight,  // The dimensions of the entire image
						const byte * src,
						liveVizCombine_t combine_reducer_type, // a combiner type like "sum_image_data"
						int bytes_per_pixel
						)
{
  
  CkPrintf("%d: liveVizPollDeposit> Deposited image at (%d,%d), (%d x %d) pixels\n",CkMyPe(),startx,starty,sizex,sizey);
  
  ImageData imageData (bytes_per_pixel);
 
  // reduce all images from this processor
  poll_image_combine_reducer=CkReduction::addReducer(imagePollCombineReducer);
  CkReductionMsg* msg = CkReductionMsg::buildNew(imageData.GetBuffSize (startx,
																		starty,
																		sizex,
																		sizey,
																		imagewidth,
																		imageheight,
																		src),
												 NULL, poll_image_combine_reducer);
  imageData.WriteHeader(combine_reducer_type,NULL,(byte*)(msg->getData()));
  imageData.AddImage (imagewidth, (byte*)(msg->getData()));
  
  //Contribute this processor's images to the global reduction
  msg->setCallback(CkCallback(vizPollReductionHandler));
  client->contribute(msg);
}





//Called by clients to start liveViz.
void liveVizPollInit()
{
  if (CkMyPe()!=0) CkAbort("liveVizInit must be called only on processor 0!");

  myGroupProxy = CProxy_liveVizPollArray::ckNew();
 
  // Incoming image requests get sent to the poll array
  CkCallback myCB(CkIndex_liveVizPollArray::request(0), myGroupProxy[0]);
  clientGetImageCallback=myCB;

  liveViz0PollInit();

#ifdef DEBUG_PRINTS
  CkPrintf("Done with liveVizPollInit\n");
#endif

}


#include "liveVizPoll.def.h"

