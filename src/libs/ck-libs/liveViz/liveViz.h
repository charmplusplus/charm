/*
  Data types, function prototypes,  etc. exported by liveViz.
  This layer does image assembly, and is the most commonly-used 
  interface to liveViz.
 */
#ifndef __UIUC_CHARM_LIVEVIZ_H
#define __UIUC_CHARM_LIVEVIZ_H

#include "liveViz0.h"
#include "ckimage.h"
#include "colorScale.h"
#include "pup_toNetwork.h"

/********************** LiveViz ***********************/
#include "liveViz.decl.h"

typedef enum {
  sum_image_data,
  max_image_data,
  sum_float_image_data,
  max_float_image_data
} liveVizCombine_t;

/*
  Start liveViz.  This routine should be called once on processor 0
  when you are ready to begin receiving image requests.

  The arrayID is the array that will make deposits into liveViz.

  The callback is signalled each time a client requests an image.
  The image parameters are passed in as a "liveVizRequestMsg *" message.
*/
void liveVizInit(const liveVizConfig &cfg, CkArrayID a, CkCallback c);
void liveVizInit(const liveVizConfig &cfg, CkArrayID a, CkCallback c, CkArrayOptions &opts);

class liveVizRequestMsg : public CMessage_liveVizRequestMsg {
public:
	liveVizRequest req;
	
	/// Additional client request data: raw network bytes from client.
	/// Use liveVizRequestUnpack to extract the data from this message.
	char *data;
	int dataLen;
	
	liveVizRequestMsg() {}
	static liveVizRequestMsg *buildNew(const liveVizRequest &req,const void *data,int dataLen);
};

/// Unpack the extra client request data as network-byte-order ints,
///  by calling pup on this class.
template<class T>
inline void liveVizRequestUnpack(const liveVizRequestMsg *src,T &dest)
{
	PUP_toNetwork_unpack p(src->data);
	p|dest;
	if (p.size()!=src->dataLen) {
		CkError("liveVizRequestUnpack: client sent %d bytes, but you wanted %d bytes!\n",
			src->dataLen, p.size());
		CkAbort("liveVizRequestUnpack size mismatch\n");
	}
}

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
                    ArrayElement* client,
                    liveVizCombine_t combine=sum_image_data);


//As above, but taking a message instead of a request:
inline void liveVizDeposit(liveVizRequestMsg *reqMsg,
		    int startx, int starty,
		    int sizex, int sizey, const byte * imageData,
		    ArrayElement* client,
                   liveVizCombine_t combine=sum_image_data)
{
	liveVizDeposit(reqMsg->req,startx,starty,sizex,sizey,imageData,client,
                       combine);
	delete reqMsg;
}

/**
  A user-written routine to convert floating-point pixels
  (when initialized with liveVizConfig::pix_float) to 
  RGB pixels, which are actually sent across the wire.
  This routine will only be called on processor 0, at the
  end of a liveViz image assembly, in pix_float mode.
*/
extern "C"
void liveVizFloatToRGB(liveVizRequest &req, 
	const float *floatSrc, unsigned char *destRgb,
	int nPixels);

/********************** LiveVizPoll **********************
These declarations should probably live in a header named "liveVizPoll.h"
*/
#include "liveVizPoll.decl.h"



/**
Initialize the poll mode of liveViz.  This routine should
be called from main::main.

*/
void liveVizPollInit();


typedef liveVizRequestMsg liveVizPollRequestMsg;


/**
Th Poll Mode has been extensively rewritten, Please read the new description in the manual.

Note the big changes:
   liveVizPoll() no longer exists
   liveVizPollDeposit requires some additional parameters

*/

void liveVizPollDeposit(ArrayElement *from,
						int startx, int starty, 
						int sizex, int sizey,             // The dimensions of the piece I'm depositing
						int imagewidth, int imageheight,  // The dimensions of the entire image
						const byte * imageData,
						liveVizCombine_t _image_combine_reducer=sum_image_data,
						int bytes_per_pixel=3
						);


class LiveVizBoundElement : public CBase_LiveVizBoundElement {
public:
	LiveVizBoundElement(){}
	LiveVizBoundElement(CkMigrateMessage *msg): CBase_LiveVizBoundElement(msg){}
	~LiveVizBoundElement(){}
	
	void deposit(CkReductionMsg *msg){
		contribute(msg);
	}
};



#endif /* def(thisHeader) */
