/*
Lowest level of liveViz: Responds to CCS requests, and formats replies.

Orion Sky Lawlor, olawlor@acm.org, 6/2002
 */
#include <stdio.h>
#include "charm++.h"
#include "conv-ccs.h"
#include <sys/types.h>
#include "liveViz0.h"
#include "pup_toNetwork.h"

//Current liveViz application configuration.
//  This is data that never changes during the course of a run.
static liveVizConfig config; 

/*This pup routine defines the on-the-wire layout of the configuration response*/
void liveVizConfig::pupNetwork(PUP::er &p) {
	int version=1; // Server version number
	p|version; 
	p|isColor;
	p|serverPush;
	p|is3d;
	if (is3d) {
		p|box.min;
		p|box.max;
	}
}

/* This pup routine defines the on-the-wire layout of the image request */
void liveVizRequest::pupNetwork(PUP::er &p) {
	int version=1; // Client version number
	p|version; 
	p|code;
	p|wid;
	p|ht;
}
	
void liveVizRequest3d::pup(PUP::er &p) {
	p|x; p|y; p|z; p|o;
	p|minZ; p|maxZ;
}

/*
  A client requests our configuration:
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

void liveViz0Deposit(const liveVizRequest &req,byte * imageData)
{
  // CkPrintf("LiveViz: sending ccs back %.6f s\n",CmiWallTimer()-startTime);
  int len=req.wid*req.ht*config.getBytesPerPixel();
  if (config.getVerbose(2))
    CmiPrintf("CCS getImage> Reply for (%d x %d) pixel or %d byte image.\n",
	      req.wid,req.ht,len);
  CcsSendDelayedReply(req.replyToken, len, imageData);
  // CkPrintf("LiveViz: request took %.6f s\n",CmiWallTimer()-startTime);
}

//Startup routine-- must be called on processor 0
void liveViz0Init(const liveVizConfig &cfg) {
  config=cfg;
  CcsRegisterHandler("lvConfig",(CmiHandler)getImageConfigHandler);
  CcsRegisterHandler("lvImage", (CmiHandler)getImageHandler);
  if (config.getVerbose(1))
    CmiPrintf("CCS getImage handlers registered.  Waiting for clients...\n");
}
