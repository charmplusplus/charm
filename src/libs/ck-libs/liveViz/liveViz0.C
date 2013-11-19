/*
Lowest level of liveViz: Responds to CCS requests, and formats replies.

Orion Sky Lawlor, olawlor@acm.org, 6/2002
 */
#include <string> /* need std::string below */
#include <stdio.h>
#include "charm++.h"
#include "conv-ccs.h"
#include <sys/types.h>
#include "liveViz0.h"
#include "pup_toNetwork.h"


//Current liveViz application configuration.
//  This is data that never changes during the course of a run.
static liveVizConfig config; 


void liveVizConfig::init(pixel_t pix,bool push)
{
	pixels=pix;
	switch(pixels) {
	case pix_greyscale: bytesPerPixel=1; break;
	case pix_color: bytesPerPixel=3; break;
	case pix_float: bytesPerPixel=4; break;
	default: CmiAbort("Unrecognized liveViz pixel code!\n");
	};
	serverPush=push;
	is3d=false;
	
	verbose=0;
}

/*This pup routine defines the on-the-wire layout of the configuration response*/
void liveVizConfig::pupNetwork(PUP::er &p) {
	int version=1; // Server version number
	p|version; 
	bool isColor=(pixels!=pix_greyscale);
	p|isColor;
	if(isColor)
        {
          pixels = pix_color;
          bytesPerPixel=3;
        }
        p|serverPush;
	p|is3d;
	if (is3d) {
		p|box.min;
		p|box.max;
	}
}

/* This pup routine defines the on-the-wire layout of the image request */
void liveVizRequest::pupNetwork(PUP::er &p) {
	int version=2; // Client version number
	p|version; 
	p|code;
	p|wid;
	p|ht;
	if (version>=2) { /* new version includes network format */
		p|compressionType;
		p|compressionQuality;
	} else { /* old version cannot compress data */
		compressionType=compressionNone;
		compressionQuality=0;
	}
}
	
void liveVizRequest3d::pup(PUP::er &p) {
	p|x; p|y; p|z; p|o;
	p|minZ; p|maxZ;
}

/*
 CCS handler "lvConfig", taking no arguments, returning a liveVizConfig.
 A client requests our configuration.
 */
extern "C" void getImageConfigHandler(char * msg)
{
  PUP_toNetwork_sizer sp;
  config.pupNetwork(sp);
  int len=sp.size();
  char *buf=new char[len];
  PUP_toNetwork_pack pp(buf);
  config.pupNetwork(pp);
  if (len!=pp.size()) CkAbort("liveVizConfig get pup mismatch");
  if (config.getVerbose(1))
    CmiPrintf("CCS getImageConfig> Sending a new client my configuration\n");
  CcsSendReply(len,buf);
  delete[] buf;
  CmiFree(msg); //Throw away the client's request
}

//static double startTime;
/*
 CCS handler "lvImage", taking a liveVizRequest, returning binary image data.
 A client requests an image from us.
 */
extern "C" void getImageHandler(char * msg)
{
  int msgLen=CmiSize(msg);
  char *buf=(char *)(msg+CmiMsgHeaderSizeBytes); msgLen-=CmiMsgHeaderSizeBytes;
  liveVizRequest o;
  PUP_toNetwork_unpack up(buf);
  o.pupNetwork(up);
  buf+=up.size(); msgLen-=up.size();
  int wid=o.wid,ht=o.ht;
  
  if (config.getVerbose(2))
    CmiPrintf("CCS getImage> Request for (%d x %d) or (0x%x x 0x%x) pixel image.\n",
	      wid,ht,wid,ht);
  if (msgLen<0) { 
    CmiError("liveViz0 getImageHandler Rejecting too-short image request\n");
    return;
  }
  
  o.replyToken = CcsDelayReply();
  liveViz0Get(o,buf,msgLen);
  CmiFree(msg); //Throw away the client's request
}


#if CMK_USE_LIBJPEG && !defined(__CYGWIN__)
#include <string> /* STL */
#include "jpeglib.h" /* Independent JPEG Group's "libjpeg", version 6b */


/* Expanded data destination object for stl output */
typedef struct {
  struct jpeg_destination_mgr pub; /* public fields */

  std::string *stl_dest;		/* target stream */
  JOCTET * buffer;		/* start of buffer */
} stl_destination_mgr;

typedef stl_destination_mgr * stl_dest_ptr;

#define OUTPUT_BUF_SIZE  4096	/* choose an efficiently stl cat'able size */

extern "C" void liveViz0_jpeg_stl_dest_init_destination (j_compress_ptr cinfo)
{
  stl_dest_ptr dest = (stl_dest_ptr) cinfo->dest;
  dest->buffer = (JOCTET *)
      (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
				  OUTPUT_BUF_SIZE * sizeof(JOCTET));
  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}
extern "C" boolean liveViz0_jpeg_stl_dest_empty_output_buffer (j_compress_ptr cinfo)
{
  stl_dest_ptr dest = (stl_dest_ptr) cinfo->dest;

  dest->stl_dest->append((char *)dest->buffer,OUTPUT_BUF_SIZE);

  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

  return TRUE;
}
extern "C" void liveViz0_jpeg_stl_dest_term_destination (j_compress_ptr cinfo)
{
  stl_dest_ptr dest = (stl_dest_ptr) cinfo->dest;
  size_t datacount = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;
  dest->stl_dest->append((char *)dest->buffer,datacount);
}

void jpeg_stl_dest(j_compress_ptr cinfo, std::string *stl_dest)
{
  stl_dest_ptr dest;

  /* The destination object is made permanent so that multiple JPEG images
   * can be written to the same file without re-executing jpeg_stl_dest.
   * This makes it dangerous to use this manager and a different destination
   * manager serially with the same JPEG object, because their private object
   * sizes may be different.  Caveat programmer.
   */
  if (cinfo->dest == NULL) {	/* first time for this JPEG object? */
    cinfo->dest = (struct jpeg_destination_mgr *)
      (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
				  sizeof(stl_destination_mgr));
  }

  dest = (stl_dest_ptr) cinfo->dest;
  dest->pub.init_destination = liveViz0_jpeg_stl_dest_init_destination;
  dest->pub.empty_output_buffer = liveViz0_jpeg_stl_dest_empty_output_buffer;
  dest->pub.term_destination = liveViz0_jpeg_stl_dest_term_destination;
  dest->stl_dest = stl_dest;
}

/**
  Return a binary string containing the JPEG-compressed data of 
  this raster image, which is wid x ht pixels, and 
  each pixel bpp bytes (must be either 1, for greyscale; or 3, for RGB).
  
  quality controls the compression rate, from
  	quality == 0, tiny compressed image, very low quality
	quality == 100, huge compressed image, very high quality
  
  This code derived from the "example.c" that ships with libjpeg; 
  see that file for comments.
*/
std::string JPEGcompressImage(int wid,int ht,int bpp, const byte *image_data, int quality) {
  struct jpeg_compress_struct cinfo;
  JSAMPROW row_pointer[1];	/* pointer to JSAMPLE row[s] */
  int row_stride;		/* physical row width in image buffer */

  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  /* Now we can initialize the JPEG compression object. */
  jpeg_create_compress(&cinfo);
  
  std::string ret;
  jpeg_stl_dest(&cinfo,&ret);

  cinfo.image_width = wid; 	/* image width and height, in pixels */
  cinfo.image_height = ht;
  
  while (cinfo.image_height>65000) {
  /* FUNKY HACK: 
  	JPEG library can't support images over 64K pixels.
	But we want tall-skinny images for 3D volume impostors.
	So we just *lie* to the JPEG library, and it'll interpret
	the image data as several side-by-side interleaved scanlines.
	The decompressor doesn't (necessarily!) need to change either...
  */
  	if (cinfo.image_height&1) {
		CkError("liveViz0 JPEGlib WARNING: cannot shrink odd image height %d\n",cinfo.image_height);
	}
  	cinfo.image_height/=2;
	cinfo.image_width*=2;
  }
  
  
  switch (bpp) {
  case 1:
  	cinfo.input_components = 1;		/* # of color components per pixel */
  	cinfo.in_color_space = JCS_GRAYSCALE; 	/* colorspace of input image */
	break;
  case 3:
  	cinfo.input_components = 3;		/* # of color components per pixel */
  	cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */
	break;
  default:
  	CkError("liveViz0's JPEGcompressImage: JPEGlib can only handle 1 or 3 bytes per pixel, not %d bpp\n",bpp);
	CkAbort("liveViz0's JPEGcompressImage: JPEGlib can only handle 1 or 3 bytes per pixel");
	break;
  }
  
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
  jpeg_start_compress(&cinfo, TRUE);
  row_stride = cinfo.image_width * bpp;	/* JSAMPLEs per row in image_buffer */
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = (JSAMPLE *)(& image_data[cinfo.next_scanline * row_stride]);
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  return ret;
}

#endif

void liveViz0Deposit(const liveVizRequest &req,byte * imageData)
{
  int len=req.wid*req.ht*config.getNetworkBytesPerPixel();
  if (config.getVerbose(2))
    CmiPrintf("CCS getImage> Reply for (%d x %d) pixel or %d byte image.\n",
	      req.wid,req.ht,len);
  switch (req.compressionType) {
    case liveVizRequest::compressionNone: /* send uncompressed pixels */
      CcsSendDelayedReply(req.replyToken, len, imageData);
      break;
#if CMK_USE_LIBJPEG && !defined(__CYGWIN__)
    case liveVizRequest::compressionJPEG: { /* JPEG-compress the data */
      std::string data=JPEGcompressImage(req.wid,req.ht,
                config.getNetworkBytesPerPixel(),imageData,
		req.compressionQuality);
      CcsSendDelayedReply(req.replyToken, data.size(), &data[0]);
      break;
    }
#endif
    case liveVizRequest::compressionRunLength:
        {
            std::string data;
            for(int i=0; i<req.ht*req.wid;)
            {
                int j=i;
                while(imageData[j]==imageData[i]&&i-j<255&&i<req.ht*req.wid)
                    i++;
                data.push_back((char)((i-j)&0xff));
                data.push_back(imageData[j]);
            }
            CcsSendDelayedReply(req.replyToken, data.size(), &data[0]);
        }
        break;
    default:
      CkError("liveViz0.C WARNING: Ignoring liveViz client's unrecognized compressionType %d\n",req.compressionType);
      CcsSendDelayedReply(req.replyToken, 0, 0);
  };
}



//Startup routine-- must be called on processor 0
void liveViz0Init(const liveVizConfig &cfg) {
  config=cfg;
  CcsRegisterHandler("lvConfig",(CmiHandler)getImageConfigHandler);
  CcsRegisterHandler("lvImage", (CmiHandler)getImageHandler);
  if (config.getVerbose(1))
    CmiPrintf("CCS getImage handlers registered.  Waiting for clients...\n");
}

void liveViz0PollInit() {
  CcsRegisterHandler("lvImage", (CmiHandler)getImageHandler);
  if (config.getVerbose(1))
    CmiPrintf("CCS getImage handler registered.  Waiting for clients...\n");
}
