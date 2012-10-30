/*
  liveViz: image-assembly interface.

Orion Sky Lawlor, olawlor@acm.org, 6/11/2002
*/
#include "liveViz.h"
#include "liveViz_impl.h"
#include "ImageData.h"

PUPbytes(liveVizConfig)

liveVizConfig lv_config;
CkReduction::reducerType image_combine_reducer;
CProxy_liveVizGroup lvG;
CProxy_LiveVizBoundElement lvBoundArray;
bool usingBoundArray;
CkCallback clientGetImageCallback;

//Called by clients to start liveViz.
void liveVizInit(const liveVizConfig &cfg, CkArrayID a, CkCallback c)
{
  if (CkMyPe()!=0) CkAbort("liveVizInit must be called only on processor 0!");
  clientGetImageCallback=c;
  //Broadcast the liveVizConfig object via our group:
  //  lv_config can't be a readonly because we may be called after
  //  main::main (because, e.g., communication is needed to find the
  //  bounding box inside cfg).
  usingBoundArray = false;
  lvG = CProxy_liveVizGroup::ckNew(cfg);
}

//Called by clients to start liveViz.
void liveVizInit(const liveVizConfig &cfg, CkArrayID a, CkCallback c, CkArrayOptions &opts)
{
  if (CkMyPe()!=0) CkAbort("liveVizInit must be called only on processor 0!");
  clientGetImageCallback=c;
  //Broadcast the liveVizConfig object via our group:
  //  lv_config can't be a readonly because we may be called after
  //  main::main (because, e.g., communication is needed to find the
  //  bounding box inside cfg).
  usingBoundArray = true;
  CkArrayOptions boundOpts;
  int dimension = opts.getNumInitial().dimension;
  switch(dimension){
  case 1: boundOpts.setNumInitial(opts.getNumInitial().data()[0]); break;
  case 2: boundOpts.setNumInitial(opts.getNumInitial().data()[0],opts.getNumInitial().data()[1]); break;
  case 3: boundOpts.setNumInitial(opts.getNumInitial().data()[0],opts.getNumInitial().data()[1],opts.getNumInitial().data()[2]); break;
  default: CmiAbort("Arrays with more than 3 dimensions are not currently supported by liveViz");
  }
  boundOpts.bindTo(a);
  lvBoundArray = CProxy_LiveVizBoundElement::ckNew(boundOpts);
  lvG = CProxy_liveVizGroup::ckNew(cfg);
}


//Called by reduction handler once every processor has the lv_config object
void liveVizInitComplete(void *rednMessage) {
  delete (CkReductionMsg *)rednMessage;
  liveViz0Init(lv_config);
}

liveVizRequestMsg *liveVizRequestMsg::buildNew(const liveVizRequest &req,const void *data,int dataLen)
{
	liveVizRequestMsg *m=new (dataLen,0) liveVizRequestMsg();
	m->req=req;
	memcpy(m->data,data,dataLen);
	m->dataLen=dataLen;
	return m;
}


//Called by lower layers when an image request comes in on processor 0.
//  Just forwards request on to user.
void liveViz0Get(const liveVizRequest &req,void *data,int dataLen)
{
  clientGetImageCallback.send(liveVizRequestMsg::buildNew(req,data,dataLen));
}

/*This array has 512 entries-- it's used to clip off large values
when summing bytes (like image values) together.  On a machine with
a small cache, it may be better to use an "if" instead of this table.
*/
static byte *overflowArray=CkImage::newClip();

#if 1
/****************** Fast but complex image combining *****************

Here I create a list of non-overlapping images (lines) and contribute 
the list of images rather than the combined images.

The contributed data looks like:
  a list of Images (lines) packed into a run of bytes
**********************************************************************/

/// Called by clients to deposit a piece of the final image.
///  Starts the reduction process.
void liveVizDeposit(const liveVizRequest &req,
                    int startx, int starty,
                    int sizex, int sizey, const byte * src,
                    ArrayElement* client,
                    liveVizCombine_t combine)
{
  if (lv_config.getVerbose(2))
    CkPrintf("liveVizDeposit> Deposited image at (%d,%d), (%d x %d) pixels, on pe %d \n",startx,starty,sizex,sizey,CkMyPe());

  ImageData imageData (lv_config.getBytesPerPixel ());
  CkReductionMsg* msg = CkReductionMsg::buildNew(imageData.GetBuffSize (startx,
                                                                        starty,
                                                                        sizex,
                                                                        sizey,
                                                                        req.wid,
																		req.ht,
                                                                        src),
                                                 NULL, image_combine_reducer);
  imageData.WriteHeader(combine,&req,(byte*)(msg->getData()));
  imageData.AddImage (req.wid, (byte*)(msg->getData()));

  //Contribute this image to the reduction
  msg->setCallback(CkCallback(vizReductionHandler));
 
  if(usingBoundArray){
    lvBoundArray[client->thisIndexMax].deposit(msg);
  }else {
    client->contribute(msg);
  }
}


/**
Called by the reduction manager to combine all the source images
received on one processor. This function adds all the image on one
processor to one list of non-overlapping images.
*/
CkReductionMsg *imageCombineReducer(int nMsg,CkReductionMsg **msgs)
{
  if (nMsg==1) { //Don't bother copying if there's only one source
    if (lv_config.getVerbose(2))
      CkPrintf("imageCombine> Skipping combine on pe %d\n",CkMyPe());
    
    CkReductionMsg *ret=msgs[0];
    msgs[0]=NULL; //Prevent reduction manager from double-delete
    return ret;
  }

  if (lv_config.getVerbose(2))
    CkPrintf("imageCombine> image combine on pe %d\n",CkMyPe());

  ImageData imageData (lv_config.getBytesPerPixel ());

  CkReductionMsg* msg = CkReductionMsg::buildNew(imageData.CombineImageDataSize (nMsg,
                                                                                 msgs),
                                                 NULL, image_combine_reducer);

  imageData.CombineImageData (nMsg, msgs, (byte*)(msg->getData()));
  return msg;
}

/* Called once at end of reduction:
   Unpacks images, combines them to form one image and passes it on to layer 0. */
void vizReductionHandler(void *r_msg)
{
  CkReductionMsg *msg = (CkReductionMsg*)r_msg;
  ImageData imageData (lv_config.getBytesPerPixel ());
  liveVizRequest req;
  byte *image = imageData.ConstructImage ((byte*)(msg->getData ()), req);
  
  if (lv_config.getVerbose(2))
      CkPrintf("vizReductionHandler> Assembled image on PE %d \n", CkMyPe());
  
  if (lv_config.getBytesPerPixel()!=lv_config.getNetworkBytesPerPixel())
  { /* Reformat image for the wire: 
      (only used by floating-point images for now) */
	int row=req.wid*lv_config.getBytesPerPixel();
	int netRow=req.wid*lv_config.getNetworkBytesPerPixel();
  	byte *netImage=new byte[req.ht*netRow];
	liveVizFloatToRGB(req, (float *)image, netImage, req.wid*req.ht);
	delete[] image;
	image=netImage;
  }

  liveViz0Deposit(req,image);

  delete[] image;
  delete msg;
}

static void liveVizNodeInit(void) {
  image_combine_reducer=CkReduction::addReducer(imageCombineReducer);
}

#else

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


CkReductionMsg *allocateImageMsg(const liveVizRequest &req,const CkRect &r,
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

  client->contribute(msg);
}

/*Called by the reduction manager to combine all the source images
received on one processor.
*/
CkReductionMsg *imageCombine(int nMsg,CkReductionMsg **msgs)
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
  CkRect destRect; destRect.makeEmpty();
  for (m=0;m<nMsg;m++) destRect=destRect.getUnion(((imageHeader *)msgs[m]->getData())->r);

//Allocate output message of that size
  byte *dest;
  CkReductionMsg *msg=allocateImageMsg(firstHdr->req,destRect,&dest);

//Add each source image to the destination
// Everything should be pre-clippped, so no further geometric clipping is needed.
// Brightness clipping, of course, is still necessary.
  CkImage destImage(destRect.wid(),destRect.ht(),bpp,dest);
  destImage.clear();
  for (m=0;m<nMsg;m++) {
  	byte *src=(byte *)msgs[m]->getData();
  	imageHeader *mHdr=(imageHeader *)src;
	src+=sizeof(imageHeader);
	if (lv_config.getVerbose(2))
	    CkPrintf("imageCombine>    pe %d  image %d is (%d,%d, %d,%d)\n",
    	          CkMyPe(),m,mHdr->r.l,mHdr->r.t,mHdr->r.r,mHdr->r.b);
	CkImage srcImage(mHdr->r.wid(),mHdr->r.ht(),bpp,src);
	destImage.addClip(mHdr->r.l-destRect.l,mHdr->r.t-destRect.t,srcImage,overflowArray);
  }

  return msg;
}

/*
Called once final image has been assembled (reduction handler).
Unpacks image, and passes it on to layer 0.
*/
void vizReductionHandler(void *r_msg)
{
  CkReductionMsg *msg = (CkReductionMsg*)r_msg;
  imageHeader *hdr=(imageHeader *)msg->getData();
  byte *srcData=sizeof(imageHeader)+(byte *)msg->getData();
  int bpp=lv_config.getBytesPerPixel();
  CkRect destRect(0,0,hdr->req.wid,hdr->req.ht);
  if (destRect==hdr->r) { //Client contributed entire image-- pass along unmodified
    liveViz0Deposit(hdr->req,srcData);
  }
  else
  { //Client didn't quite cover whole image-- have to pad
    CkImage src(hdr->r.wid(),hdr->r.ht(),bpp,srcData);
    CkAllocImage dest(hdr->req.wid,hdr->req.ht,bpp);
    dest.clear();
    dest.put(hdr->r.l,hdr->r.t,src);
    liveViz0Deposit(hdr->req,dest.getData());
  }
  delete msg;
}
#endif


#include "liveViz.def.h"
