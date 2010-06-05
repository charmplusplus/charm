
#include <stdio.h>

#include "blue.h"
#include "bigsim_record.h"
#include "blue_impl.h"
#include "charm++.h"
#include "envelope.h"

int converseheader = 0;

int _heter = 1;

BgMessageRecorder::BgMessageRecorder(FILE * f_, int node) :f(f_) {
  nodelevel = node;
  fwrite(&nodelevel, sizeof(int),1,f);

  converseheader = CmiReservedHeaderSize;
  fwrite(&converseheader, sizeof(int),1,f);

  // write processor data
  int d = BgGetGlobalWorkerThreadID();
  int mype = d;
  fwrite(&d, sizeof(int),1,f);

  nodeInfo *myNode = cta(threadinfo)->myNode;
  d = myNode->x;
  fwrite(&d, sizeof(int), 1, f);
  d = myNode->y;
  fwrite(&d, sizeof(int), 1, f);
  d = myNode->z;
  fwrite(&d, sizeof(int), 1, f);

  d = BgGetThreadID();
  fwrite(&d, sizeof(int), 1, f);

  d = BgNumNodes();
  fwrite(&d, sizeof(int), 1, f);

  d = BgNumNodes()*BgGetNumWorkThread();
  fwrite(&d, sizeof(int), 1, f);
 
  d = cva(bgMach).x;
  fwrite(&d, sizeof(int), 1, f);
  d = cva(bgMach).y;
  fwrite(&d, sizeof(int), 1, f);
  d = cva(bgMach).z;
  fwrite(&d, sizeof(int), 1, f);
 
  if (nodelevel == 1 && mype % BgGetNumWorkThread() == 0) {
    write_nodeinfo();
  }
}

void BgMessageRecorder::write_nodeinfo()
{
  int mype = BgGetGlobalWorkerThreadID();
  int mynode = mype / BgGetNumWorkThread();
  char fName[128];
  sprintf(fName,"bgnode_%06d.log",mynode);
  FILE *fp = fopen(fName, "w");
  fprintf(fp, "%d %d\n", mype, mype + BgGetNumWorkThread()-1);
  fclose(fp);
}

CmiBool BgMessageRecorder::record(char *msg) {
		CmiAssert(msg != NULL);
//                if (BgGetGlobalWorkerThreadID()==0) printf("srcpe: %d size: %d handle: %d\n",CmiBgMsgSrcPe(msg),CmiBgMsgLength(msg),CmiBgMsgHandle(msg));
                int d = CmiBgMsgSrcPe(msg);
		pos = ftell(f);
                fwrite(&d, sizeof(int), 1, f);
//CmiAssert(CmiBgMsgThreadID(msg) != -1);
                if ( (nodelevel == 0 && d == BgGetGlobalWorkerThreadID()) ||
                     (nodelevel == 1 && d/BgGetNumWorkThread() == BgGetGlobalWorkerThreadID()/BgGetNumWorkThread()) ) {
                    //CmiPrintf("[%d] local message.\n", BgGetGlobalWorkerThreadID());
                    return CmiTrue; // don't record local msg
                }
/*
if (BgGetGlobalWorkerThreadID()==1 && CmiBgMsgHandle(msg) == 21) {
int *m = (int *) ((char *)msg+CmiReservedHeaderSize);
printf("replay: %d %d\n", m[0], m[1]);
}
*/
		if (_heter) {                     // heter
		int isCharm = (CmiGetHandler(msg)==_charmHandlerIdx) || (CmiGetXHandler(msg)==_charmHandlerIdx);
                fwrite(&isCharm, sizeof(int), 1, f);
		if (isCharm) {
		PUP::toDisk p(f);
 		envelope *env = (envelope*)msg;
		char *m = (char *)EnvToUsr(env);
		CkPupMessage(p, (void **)&m, 2);
		}
		else {    // converse
                d = CmiBgMsgLength(msg) - CmiReservedHeaderSize;
                fwrite(&d, sizeof(int), 1, f);
                fwrite(msg+CmiReservedHeaderSize, sizeof(char), d, f);
		}
		d = CmiGetHandler(msg);
                fwrite(&d, sizeof(int), 1, f);
                d = CmiGetXHandler(msg);
                fwrite(&d, sizeof(int), 1, f);
                d = CmiGetInfo(msg);
                fwrite(&d, sizeof(int), 1, f);
		  // bigsim header
		d = CmiBgMsgType(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgLength(msg) - sizeof(envelope);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgNodeID(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgThreadID(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgHandle(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgID(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgSrcPe(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgFlag(msg);
                fwrite(&d, sizeof(int), 1, f);
		d = CmiBgMsgRefCount(msg);
                fwrite(&d, sizeof(int), 1, f);
		}
		else {      // store by bytes
                d = CmiBgMsgLength(msg);
                fwrite(&d, sizeof(int), 1, f);
                fwrite(msg, sizeof(char), d, f);
		}
                //CmiPrintf("[%d] BgMessageRecord>  PE: %d size: %d msg: %p\n", BgGetGlobalWorkerThreadID(), CmiBgMsgSrcPe(msg),CmiBgMsgLength(msg), msg);
                return CmiTrue;
}

BgMessageReplay::BgMessageReplay(FILE * f_, int node) :f(f_) {
  nodelevel = node;
  lcount = rcount = 0;

  int d;
  fread(&d, sizeof(int), 1, f);
  if (nodelevel != d) {
    CmiPrintf("BgReplay> Fatal error: can not replay %s logs.\n", d?"node level":"processor level");
    CmiAbort("BgReplay error");
  }

  fread(&converseheader, sizeof(int),1,f);

  fread(&d, sizeof(int), 1, f);
  BgSetGlobalWorkerThreadID(d);

  nodeInfo *myNode = cta(threadinfo)->myNode;
  fread(&d, sizeof(int), 1, f);
  myNode->x = d;
  fread(&d, sizeof(int), 1, f);
  myNode->y = d;
  fread(&d, sizeof(int), 1, f);
  myNode->z = d;

  fread(&d, sizeof(int), 1, f);
  BgSetThreadID(d);
  if (nodelevel == 0 && d != 0)
    CmiAbort("BgReplay> Fatal error: can not replay processor level log of rank other than 0.");
  fread(&d, sizeof(int), 1, f);
  BgSetNumNodes(d);
  fread(&d, sizeof(int), 1, f);
  if (nodelevel == 1 && d/BgNumNodes() != BgGetNumWorkThread())
    CmiAbort("BgReplay> Fatal error: the number of worker threads is not the same as in the logs.\n");
  BgSetNumWorkThread(d/BgNumNodes());

  fread(&d, sizeof(int), 1, f);
  cva(bgMach).x = d;
  fread(&d, sizeof(int), 1, f);
  cva(bgMach).y = d;
  fread(&d, sizeof(int), 1, f);
  cva(bgMach).z = d;

  //myNode->id = nodeInfo::XYZ2Local(myNode->x,myNode->y,myNode->z);

  CmiPrintf("BgMessageReplay: PE => %d NumPes => %d wth:%d\n", BgGetGlobalWorkerThreadID(), BgNumNodes()*BgGetNumWorkThread(), BgGetNumWorkThread());

  replay();
}

void BgRead_nodeinfo(int node, int &startpe, int &endpe)
{
    char fName[128];
    sprintf(fName,"bgnode_%06d.log",node);
    FILE *fp = fopen(fName, "r");
    if (fp==NULL) {
      CmiPrintf("BgReplayNode> metadata file for node %d does not exist!\n", node);
      CmiAbort("BgRead_nodeinfo");
    }
    fscanf(fp, "%d %d\n", &startpe, &endpe);
    fclose(fp);
}

int BgMessageReplay::replay(void) {
                int nextPE;
                int ret =  fread(&nextPE, sizeof(int), 1, f);
                if (-1 == ret || ret == 0) {
//			done();
 			callAllUserTracingFunction();   // flush logs
                        ConverseExit();
                        return 0;
                }
		int mype = BgGetGlobalWorkerThreadID();
                if ( (nodelevel == 0 && nextPE == mype) ||
                     (nodelevel == 1 && nextPE/BgGetNumWorkThread() == mype/BgGetNumWorkThread()) ) {
//printf("BgMessageReplay> local message\n");
                  lcount ++;
                  return 0;
                }

		char *msg;
                int nextSize;
		if (_heter) {                     // heter
		int isCharm;
		int d;
                fread(&isCharm, sizeof(int), 1, f);
		if (isCharm) {
printf("Charm msg\n");
		PUP::fromDisk p(f);
		char *m;
		CkPupMessage(p, (void **)&m, 2);
		msg = (char*)UsrToEnv(m);
		}
		else {   // Converse
                  fread(&d, sizeof(int), 1, f);
		  int len = d + CmiReservedHeaderSize;
printf("Converse msg: %d\n", len);
		  msg = (char*)CmiAlloc(len);
		  fread(msg+CmiReservedHeaderSize, sizeof(char), d, f);
		}
		fread(&d, sizeof(int), 1, f);
                CmiSetHandler(msg, d);
                fread(&d, sizeof(int), 1, f);
                CmiSetXHandler(msg, d);
		fread(&d, sizeof(int), 1, f);
                CmiSetInfo(msg, d);
		  // bigsim header
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgType(msg) = d;
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgLength(msg) = d + sizeof(envelope);
		nextSize = CmiBgMsgLength(msg);
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgNodeID(msg) = d;
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgThreadID(msg) = d;
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgHandle(msg) = d;
		if(d<0 || d>1000) CmiPrintf("Suspicious BigSim handler %d\n", d);
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgID(msg) = d;
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgSrcPe(msg) = d;
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgFlag(msg) = d;
                fread(&d, sizeof(int), 1, f);
		CmiBgMsgRefCount(msg) = d;
		}
		else {      // read by bytes
                ret = fread(&nextSize, sizeof(int), 1, f);
                CmiAssert(ret ==1);
                CmiAssert(nextSize > 0);
                msg = (char*)CmiAlloc(nextSize);
                ret = fread(msg, sizeof(char), nextSize, f);
                if (ret != nextSize) {
                  CmiPrintf("Bigsim replay> fread returns only %d when asked %d bytes!\n", ret, nextSize);
                CmiAssert(ret == nextSize);
                }
		}
                CmiAssert(CmiBgMsgLength(msg) == nextSize);
                // CmiPrintf("BgMessageReplay>  pe:%d size: %d handle: %d msg: %p\n", nextPE, nextSize, CmiBgMsgHandle(msg), msg);
                BgSendLocalPacket(mype%BgGetNumWorkThread(), CmiBgMsgHandle(msg), LARGE_WORK, nextSize, msg);
                rcount ++;
                return 1;
}
