/*
Lowest level of liveViz: Responds to CCS requests, and formats replies.

Orion Sky Lawlor, olawlor@acm.org, 6/2002
 */
#include <stdio.h>
#include "charm++.h"
#include "conv-ccs.h"
#include <sys/types.h>
#include "liveViz0.h"
#include "networkVar.h"

//Current liveViz application configuration.
//  This is data that never changes during the course of a run.
static liveVizConfig config; 

void copy(const networkVector3d &src,vector3d &dest)
{
	dest.x=src.x;
	dest.y=src.y;
	dest.z=src.z;
}
void copy(const vector3d &src,networkVector3d &dest)
{
	dest.x=src.x;
	dest.y=src.y;
	dest.z=src.z;
}

/*
  A client requests our configuration:
 */
extern "C" void getImageConfigHandler(char * msg)
{
  /*This class defines the on-the-wire layout of the configuration response*/
  struct configInfo{
    networkInt version; //Server version number (currently 1)
    networkInt isColor;
    networkInt isPush;
    networkInt is3d;
    networkVector3d min,max; //Bounding box, if 3d
  };
  configInfo ret;
  ret.version=1; //Hardcoded server version
  ret.isColor=config.getColor()?1:0;
  ret.isPush=config.getPush()?1:0;
  ret.is3d=config.get3d()?1:0;
  bbox3d box; box.empty(); box.add(vector3d(0,0,0));
  if (config.get3d()) box=config.getBox();
  copy(box.min,ret.min); copy(box.max,ret.max);
  if (config.getVerbose(1))
    CmiPrintf("CCS getImageConfig> Sending a new client my configuration\n");
  CcsSendReply(sizeof(configInfo), &ret);
  CmiFree(msg); //Throw away the client's request
}

/*
  This class defines the on-the-wire layout of an image request:
*/
class networkImageRequest {
public:
  networkInt version; //Client version number (currently 1)
  networkInt code;//Request code-- application-defined
  networkInt wid,ht;//Size (pixels) of requested image
  //The following fields are only valid for 3d servers:
  networkVector3d x,y,z,o;//Coordinate axes of screen (xScreen=(v-o).dot(x))
  networkDouble minZ,maxZ;//range of allowable z values
};

/*
 A client requests an image from us.
 */
extern "C" void getImageHandler(char * msg)
{
  networkImageRequest *r=(networkImageRequest *)(msg+CmiMsgHeaderSizeBytes);
  int wid=r->wid,ht=r->ht;
  
  if (config.getVerbose(2))
    CmiPrintf("CCS getImage> Request for (%d x %d) or (0x%x x 0x%x) pixel image.\n",
	      wid,ht,wid,ht);
  
  liveVizRequest3d o;
  o.replyToken = CcsDelayReply();
  o.code=r->code;
  o.wid=r->wid;
  o.ht=r->ht;
  if (config.get3d()) { /*Grab a 3d request*/
    copy(r->x,o.x); copy(r->y,o.y); copy(r->z,o.z); copy(r->o,o.o);
    o.minZ=r->minZ;
    o.maxZ=r->maxZ;
  }
  liveViz0Get(o);
  CmiFree(msg); //Throw away the client's request
}

void liveViz0Deposit(const liveVizRequest &req,byte * imageData)
{
  int len=req.wid*req.ht*config.getBytesPerPixel();
  if (config.getVerbose(2))
    CmiPrintf("CCS getImage> Reply for (%d x %d) pixel or %d byte image.\n",
	      req.wid,req.ht,len);
  CcsSendDelayedReply(req.replyToken, len, imageData);
}

//Startup routine-- must be called on processor 0
void liveViz0Init(const liveVizConfig &cfg) {
  config=cfg;
  CcsRegisterHandler("lvConfig",(CmiHandler)getImageConfigHandler);
  CcsRegisterHandler("lvImage", (CmiHandler)getImageHandler);
  if (config.getVerbose(1))
    CmiPrintf("CCS getImage handlers registered.  Waiting for clients...\n");
}
