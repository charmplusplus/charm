#include <stdio.h>


#include "headerpad.h"
/*readonly*/ CProxy_main mainProxy;
main::main(CkArgMsg *m) {
    delete m;
    mainProxy=thishandle;
    CmiPrintf("Info: converse header: %d envelope: %d\n", CmiReservedHeaderSize, sizeof(envelope));
    // make a message with 1 payload
    testMsg *msg = new testMsg;
    // get pointer to envelope header
    unsigned char *env= (unsigned char*) UsrToEnv(msg);
    unsigned char *hdr= (unsigned char*) msg;
    hdr-=sizeof(envelope);
    hdr-=CmiReservedHeaderSize;
    // output converse header field by field
    
    // output converse header byte by byte
    CkPrintf("CmiHeader\n");
    for(int i=0;i<CmiReservedHeaderSize;i++)
      CkPrintf("%03u|",hdr[i]);
    CkPrintf("\n");
    CkPrintf("Envelope\n");
    for(int i=0;i<sizeof(envelope);i++)
      CkPrintf("%03u|",env[i]);
    CkPrintf("\n");
    mainProxy.recv(msg);
  }
void main::recv(testMsg *msg)
{
  CkPrintf("message as received\n");
    // get pointer to envelope header
    unsigned char *env= (unsigned char*) UsrToEnv(msg);
    unsigned char *hdr= (unsigned char*) msg;
    hdr-=sizeof(envelope);
    hdr-=CmiReservedHeaderSize;
    // output converse header field by field
    
    // output converse header byte by byte
    CkPrintf("CmiHeader\n");
    for(int i=0;i<CmiReservedHeaderSize;i++)
      CkPrintf("%03u|",hdr[i]);
    CkPrintf("\n");
    CkPrintf("Envelope\n");
    for(int i=0;i<sizeof(envelope);i++)
      CkPrintf("%03u|",env[i]);
    CkPrintf("\n");
    CkExit();
}

#include "headerpad.def.h"
