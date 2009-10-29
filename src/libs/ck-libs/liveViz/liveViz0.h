/*
Prototypes  for CCS/visualization interaction routines, 
the lowest level of liveViz. (liveViz level 0)

Orion Sky Lawlor, olawlor@acm.org, 6/11/2002
*/
#ifndef __UIUC_CHARM_LIVEVIZ0_H
#define __UIUC_CHARM_LIVEVIZ0_H

#include "conv-ccs.h"
#include "pup.h"
#include "ckvector3d.h"

typedef unsigned char byte;

/// A liveVizRequest describes the basics of a client image request.
struct liveVizRequest {
	CcsDelayedReply replyToken;
	int code; ///< Application-defined request type
	int wid,ht;///< Size (pixels) of requested image
	
	typedef enum {compressionNone=0, compressionJPEG=1, compressionRunLength=2} compressionType_t;
	int compressionType; ///< 0: uncompressed; 1: JPEG; 2: zip?
	int compressionQuality; ///< For jpeg, 0..100 percent quality.
	
	void pupNetwork(PUP::er &p);
};

/// Additional 3D data associated with a 3D LiveViz request.
struct liveVizRequest3d {
	CkVector3d x,y,z,o; ///< Coordinate axes & origin of screen
	double minZ,maxZ; ///< Range of Z values to display
	
	void pup(PUP::er &p);
};



/**
  Called when the client requests an image as described above.
  You must implement this routine, which will be called from a single processor.
  data and dataLen are additional data supplied with the CCS request.
  This routine is called on the processor where the request arrives;
  this is normally processor 0.
*/
void liveViz0Get(const liveVizRequest &req,void *data,int dataLen);

/**
  Each call to liveViz, above, must eventually result in a 
  call to this routine, which delivers an entire, finished image 
  to the client.
*/
void liveViz0Deposit(const liveVizRequest &req,byte * imageData);


/**
  liveVizConfig describes the type of images available from
  this parallel server--images may be color or greyscale,
  be pushed to the client or pulled, and may come from a set
  of 3d axes or be simply 2d.
*/
class liveVizConfig {
public:
	/* This enum determines how images are combined on the parallel machine */
	typedef enum {
		/* Greyscale pixels: 1 byte (grey) per pixel */
		pix_greyscale=0,
		/* Color pixels: 3 bytes (RGB) per pixel */
		pix_color=1,
		/* Floating-point pixels, one float per pixel */
		pix_float=2
	} pixel_t;

	liveVizConfig() { init(pix_greyscale); }
	
	/// 2D constructors:
	liveVizConfig(bool isColor,bool serverPush_) {
		init(isColor?pix_color:pix_greyscale,serverPush_);
	}
	liveVizConfig(pixel_t pix,bool serverPush_) {
		init(pix,serverPush_);
	}
	
	/// 3D constructors:
	liveVizConfig(bool isColor,bool serverPush_,const CkBbox3d &box_) {
		init(isColor?pix_color:pix_greyscale,serverPush_);
		is3d=true; box=box_;
	}
	liveVizConfig(pixel_t pix,bool serverPush_,const CkBbox3d &box_) {
		init(pix,serverPush_);
		is3d=true; box=box_;
	}
	
	/// Increment the verbosity level on the server.
	void moreVerbose(void) {verbose++;}
	
	/// Extract configuration information:
	bool getColor(void) const {return pixels==pix_color;}
	/// Return the number of bytes per pixel used during assembly
	int getBytesPerPixel(void) const {return bytesPerPixel;}
	/// Return the number of bytes per pixel used on the wire
	int getNetworkBytesPerPixel(void) const {return (pixels==pix_greyscale)?1:3;}
	bool getPush(void) const {return serverPush;}
	bool get3d(void) const {return is3d;}
	const CkBbox3d &getBox(void) const {return box;}
	bool getVerbose(int thanLevel) const {return verbose>=thanLevel;}
	
	void pupNetwork(PUP::er &p);
	
private:
	pixel_t pixels;
	int bytesPerPixel;
	
	/**
	 If true, the server produces images regardless of whether the client
	  asks for them; this is true for animating and batch-mode images.
	  If false, the server will respond to client requests; this is true for
	  viz-on-demand services.
	  The client repeatedly asks for an image if this is set; if not set
	  it only asks when the client's window size or viewpoint changes.
	 */
	bool serverPush;
	
	bool is3d; ///< If true, show 3d camera motion tools
	CkBbox3d box; ///< If 3d, the viewed object's bounding box, universe coords
	
	int verbose; ///< If nonzero, printf status info; higher numbers give more prints
	
	void init(pixel_t pix,bool serverPush=false);
};


/**
  Begin responding to image requests under the given configuration.
  liveViz0Init should be called exactly once on processor 0 when
  your program is ready to begin processing image requests.
*/
void liveViz0Init(const liveVizConfig &cfg);
void liveViz0PollInit();


// A wrapper for the queue of images which is built for the Poll mode
class imageUnit{
public:
  int bpp;
  int wid;
  int ht;
  byte * imageData;
  
  int len(){return bpp*wid*ht;}

  imageUnit(int _bpp, int _wid, int _ht, byte*_imageData){
	bpp=_bpp;
	wid=_wid;
	ht=_ht;
	imageData=_imageData;
  }

  imageUnit(){}
};




#endif /* def(thisHeader) */

