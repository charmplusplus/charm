/*
  liveViz: image-assembly interface.

Orion Sky Lawlor, olawlor@acm.org, 6/11/2002
*/
#include "liveViz.h"
#include "image.h"

static liveVizConfig lv_config;
static CkCallback clientGetImageCallback;

//Called by clients to start liveViz.
void liveVizInit(const liveVizConfig &cfg, CkArrayID a, CkCallback c)
{
  if (CkMyPe()!=0) CkAbort("liveVizInit must be called only on processor 0!");
  clientGetImageCallback=c;
  //Broadcast the liveVizConfig object via our group:
  //  lv_config can't be a readonly because we may be called after
  //  main::main (because, e.g., communication is needed to find the
  //  bounding box inside cfg).
  CProxy_liveVizGroup::ckNew(cfg);
}

static void liveVizInitComplete(void *rednMessage);

//The liveVizGroup is only used to set lv_config on every processor.
class liveVizGroup : public Group {
public:
	liveVizGroup(const liveVizConfig &cfg) {
		lv_config=cfg;
		contribute(0,0,CkReduction::sum_int,liveVizInitComplete);
	}
};

//Called by reduction handler once every processor has the lv_config object
static void liveVizInitComplete(void *rednMessage) {
  delete (CkReductionMsg *)rednMessage;
  liveViz0Init(lv_config);
}


//Called by lower layers when an image request comes in on processor 0.
//  Just forwards request on to user.
void liveViz0Get(const liveVizRequest3d &req)
{
  clientGetImageCallback.send(new liveVizRequestMsg(req));
}

//Image combining reduction type: defined below.
static CkReductionMsg *imageCombine(int nMsg,CkReductionMsg **msgs);
static void vizReductionHandler(void *r_msg);
static CkReduction::reducerType imageCombineReducer;

/*This array has 512 entries-- it's used to clip off large values
when summing bytes (like image values) together.  On a machine with
a small cache, it may be better to use an "if" instead of this table.
*/
static byte *overflowArray;

static void liveVizNodeInit(void) {
	imageCombineReducer=CkReduction::addReducer(imageCombine);
	overflowArray=new byte[512];
	for (int i=0;i<512;i++) overflowArray[i]=(byte)((i<256)?i:255);
}

#if 1
/****************** Fairly Slow, but straightforward image combining *****************
The contributed data looks like:
	liveVizRequest
	Rectangle of image, in final image coordinates (fully clipped)
	image data

This turned out to be incredibly hideous, because the simple data 
structure above has to be compressed into a flat run of bytes.
I'm considering changing the reduction interface to use messages,
or some sort of pup'd object.
*/

class imageHeader {
public:
	liveVizRequest req;
	Rect r;
	imageHeader(const liveVizRequest &req_,const Rect &r_)
		:req(req_), r(r_) {}
};

static CkReductionMsg *allocateImageMsg(const liveVizRequest &req,const Rect &r,
	byte **imgDest)
{
  imageHeader hdr(req,r);
  CkReductionMsg *msg=CkReductionMsg::buildNew(
  	sizeof(imageHeader)+r.area()*lv_config.getBytesPerPixel(),
	NULL,imageCombineReducer);
  byte *dest=(byte *)msg->getData();
  *(imageHeader *)dest=hdr;
  *imgDest=dest+sizeof(hdr);
  return msg;
}

//Called by clients to deposit a piece of the final image
void liveVizDeposit(const liveVizRequest &req,
		    int startx, int starty, 
		    int sizex, int sizey, const byte * src,
		    ArrayElement* client)
{
  if (lv_config.getVerbose(2))
    CkPrintf("liveVizDeposit> Deposited image at (%d,%d), (%d x %d) pixels, on pe %d\n",
    	startx,starty,sizex,sizey,CkMyPe());

//Allocate a reductionMessage:
  Rect r(startx,starty, startx+sizex,starty+sizey);
  r=r.getIntersect(Rect(req.wid,req.ht)); //Never copy out-of-bounds regions
  if (r.isEmpty()) r.zero();
  int bpp=lv_config.getBytesPerPixel();
  byte *dest;
  CkReductionMsg *msg=allocateImageMsg(req,r,&dest);
  
//Copy our image into the reductionMessage:
  if (!r.isEmpty()) {
    //We can't just copy image with memcpy, because we may be clipping user data here:
    Image srcImage(sizex,sizey,bpp,(byte *)src);
    srcImage.window(r.getShift(-startx,-starty)); //Portion of src overlapping dest
    Image destImage(r.wid(),r.ht(),bpp,dest);
    destImage.put(0,0,srcImage);
  }

//Contribute this image to the reduction
  msg->setCallback(vizReductionHandler);
  client->contribute(msg);
}

/*Called by the reduction manager to combine all the source images 
received on one processor.
*/
static CkReductionMsg *imageCombine(int nMsg,CkReductionMsg **msgs)
{
  if (nMsg==1) { //Don't bother copying if there's only one source
        if (lv_config.getVerbose(2))
	    CkPrintf("imageCombine> Skipping combine on pe %d\n",CkMyPe());
  	CkReductionMsg *ret=msgs[0]; 
	msgs[0]=NULL; //Prevent reduction manager from double-delete
	return ret;
  }
  int m;
  int bpp=lv_config.getBytesPerPixel();
  imageHeader *firstHdr=(imageHeader *)msgs[0]->getData();

//Determine the size of the output image
  Rect destRect; destRect.makeEmpty();
  for (m=0;m<nMsg;m++) destRect=destRect.getUnion(((imageHeader *)msgs[m]->getData())->r);
  
//Allocate output message of that size
  byte *dest;
  CkReductionMsg *msg=allocateImageMsg(firstHdr->req,destRect,&dest);
  
//Add each source image to the destination
// Everything should be pre-clippped, so no further geometric clipping is needed.
// Brightness clipping, of course, is still necessary.
  Image destImage(destRect.wid(),destRect.ht(),bpp,dest);
  destImage.clear();
  for (m=0;m<nMsg;m++) {
  	byte *src=(byte *)msgs[m]->getData();
  	imageHeader *mHdr=(imageHeader *)src;
	src+=sizeof(imageHeader); 
	if (lv_config.getVerbose(2))
	    CkPrintf("imageCombine>    pe %d  image %d is (%d,%d, %d,%d)\n",
    	          CkMyPe(),m,mHdr->r.l,mHdr->r.t,mHdr->r.r,mHdr->r.b);
	Image srcImage(mHdr->r.wid(),mHdr->r.ht(),bpp,src);
	destImage.addClip(mHdr->r.l-destRect.l,mHdr->r.t-destRect.t,srcImage,overflowArray);
  }

  return msg;
}

/*
Called once final image has been assembled (reduction handler).
Unpacks image, and passes it on to layer 0.
*/
static void vizReductionHandler(void *r_msg)
{
  CkReductionMsg *msg = (CkReductionMsg*)r_msg;
  imageHeader *hdr=(imageHeader *)msg->getData();
  byte *srcData=sizeof(imageHeader)+(byte *)msg->getData();
  int bpp=lv_config.getBytesPerPixel();
  Rect destRect(0,0,hdr->req.wid,hdr->req.ht);
  if (destRect==hdr->r) { //Client contributed entire image-- pass along unmodified
    liveViz0Deposit(hdr->req,srcData);
  }
  else 
  { //Client didn't quite cover whole image-- have to pad
    Image src(hdr->r.wid(),hdr->r.ht(),bpp,srcData);
    AllocImage dest(hdr->req.wid,hdr->req.ht,bpp);
    dest.clear();
    dest.put(hdr->r.l,hdr->r.t,src);
    liveViz0Deposit(hdr->req,dest.getData());
  }
  delete msg;
}

#else
/****************** Fast but complex image combining *****************

Aught to do some sort of run-length encoding, or list-of-rectangles here.

INCOMPLETE
*/

#endif


#include "liveViz.def.h"
