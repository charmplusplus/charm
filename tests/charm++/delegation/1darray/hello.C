#include <stdio.h>
#include "hello.decl.h"
#include "envelope.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Hello arr;
/*readonly*/ int nElements;
/*readonly*/ CProxy_DelegateMgr delMgr;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thishandle;
 
    delMgr = CProxy_DelegateMgr::ckNew();

    arr = CProxy_Hello::ckNew(nElements);

    arr[0].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*group*/
class DelegateMgr : public CkDelegateMgr {
public:
	DelegateMgr(void) {}
	virtual void ArraySend(CkDelegateData *pd,int ep,void *m,const CkArrayIndex &idx,CkArrayID a)
	{
		CkArray *arrMgr=CProxy_CkArray(a).ckLocalBranch();
		int onPE=arrMgr->lastKnown(idx);
		if (onPE==CkMyPe()) 
		{ //Send to local element
			arrMgr->deliver((CkMessage *)m, CkDeliver_queue);
		} else 
		{ //Forward to remote element
			ckout<<"DelegateMgr> Sending message for "<<idx.data()[0]<<" to "<<onPE<<endl;
			envelope *env=UsrToEnv(m);
			CkPackMessage(&env);
			forwardMsg(ep,idx,a,env->getTotalsize(),(char *)env);
			CkFreeMsg(m);
		}
	}
	void forwardMsg(int ep,const CkArrayIndex &idx,
			const CkArrayID &a,
			int nBytes,char *env)
	{
		ckout<<"DelegateMgr> Recv message for "<<idx.data()[0]<<endl;
		//Have to allocate a new message because of Charm++'s 
		// weird allocate rules:
		envelope *msg=(envelope *)CmiAlloc(nBytes);
		memcpy(msg,env,nBytes);
		CkUnpackMessage(&msg);
		CProxy_CkArray ap(msg->getsetArrayMgr());
		ap.ckLocalBranch()->deliver((CkMessage *)EnvToUsr(msg),CkDeliver_inline);
	}
};

/*array [1D]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex);
    CProxy_Hello delArr=arr;
    delArr.ckDelegate(delMgr.ckLocalBranch());
    if (thisIndex < nElements-1)
      //Pass the hello on:
      delArr[thisIndex+1].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
