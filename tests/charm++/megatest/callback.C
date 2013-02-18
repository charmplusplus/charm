#include "callback.h"
#include <stdlib.h>

void callback_moduleinit(void)
{
	/*empty*/
}
void callback_init(void) 
{
	CProxy_callbackCoord::ckNew(0);
}

class callbackMsg : public CMessage_callbackMsg {
public:
	int val;
	callbackMsg(int v) :val(v) {}
};

/*This integer lets us tell which method we called,
so if it's not the method that executes we'll be able to tell.
*/
enum {
  typeChare=1000,
  typeGroup=2000,
  typeArray=4000,
  typeCfn=5000,
  msgGood=1,
  msgNull=2
};

int msg2val(callbackMsg *m) {
	if (m==NULL) return 0;
	int ret=m->val;
	delete m;
	return ret;
}

extern "C" void acceptCFnCall(void *param,void *msg) 
{
	CProxy_callbackCoord coord=*(CProxy_callbackCoord *)param;
	coord.done(typeCfn,msg2val((callbackMsg *)msg));
}


class callbackChare : public CBase_callbackChare {
	CProxy_callbackCoord coord;
public:
	callbackChare(CProxy_callbackCoord coord_) :coord(coord_) {}
	void accept(callbackMsg *m) {
		coord.done(typeChare,msg2val(m));
	}
};
class callbackArray : public CBase_callbackArray {
	CProxy_callbackCoord coord;
public:
	callbackArray(CProxy_callbackCoord coord_) :coord(coord_) {}
	callbackArray(CkMigrateMessage *) {}
	void accept(callbackMsg *m) {
		coord.done(typeArray,msg2val(m));
	}
};
class callbackGroup : public CBase_callbackGroup {
	CProxy_callbackCoord coord;
public:
	callbackGroup(CProxy_callbackCoord coord_) :coord(coord_) {}
	void accept(callbackMsg *m) {
		coord.done(typeGroup,msg2val(m));
	}
	void reflect(CkCallback cb,int val) {
		cb.send(new callbackMsg(val));
	}
};

class callbackCoord : public CBase_callbackCoord {
	CProxy_callbackChare cp;
	int nArr;
	CProxy_callbackArray ap;
	CProxy_callbackGroup gp;
	int state;
	int expectType,expectParam;
	int expectCount;

	//Send this callback off:
	void send(const CkCallback &c) {
		if (rand()&0x100) { //Half the time, send the callback ourselves
			c.send(new callbackMsg(expectParam));
		} else //Otherwise, reflect the callback off a random processor:
			gp[rand()%CkNumPes()].reflect(c,expectParam);
	}
	
	void next(void) {
		state++;
		expectParam=rand();
		expectCount=1;
		switch(state) {
		case 0: //Send to chare
			expectType=typeChare;
			send(CkCallback(CkIndex_callbackChare::idx_accept(&callbackChare::accept),
					cp));
			break;
		case 1: //Send to array element
			expectType=typeArray;
			send(CkCallback(CkIndex_callbackArray::accept(NULL),
					CkArrayIndex1D(nArr-1),ap));
			break;
		case 2: //Send to group member 0
			expectType=typeGroup;
			send(CkCallback(CkIndex_callbackGroup::accept(NULL),
					CkNumPes()-1,gp));
			break;
		case 3: //Send to C function
			expectType=typeCfn;
			send(CkCallback(acceptCFnCall,&thisProxy));
			break;
		case 4: //Broadcast to array
			expectCount=nArr;
			expectType=typeArray;
			send(CkCallback(CkIndex_callbackArray::accept(NULL),ap));
			break;
		case 5: //Broadcast to group
			expectCount=CkNumPes();
			expectType=typeGroup;
			send(CkCallback(CkIndex_callbackGroup::accept(NULL),gp));
			break;
		case 6: //That's it
			expectType=-1;
			expectParam=-1;
			thisProxy.threadedTest();
			break;
		};
	}
public:
	callbackCoord(void) {
		cp=CProxy_callbackChare::ckNew(thisProxy);
		nArr=CkNumPes()*2+1;
		ap=CProxy_callbackArray::ckNew(thisProxy,nArr);
		gp=CProxy_callbackGroup::ckNew(thisProxy);
		state=-1;
		next();
	}
	void done(int objectType,int paramVal) {
		if (objectType!=expectType || paramVal!=expectParam) {
			CkError("Callback step %d: expected object %d, got %d; expected param %d, got %d\n",
				state,expectType,objectType, expectParam,paramVal);
			CkAbort("Unexpected callback object or parameter");
		}
		expectCount--;
		if (expectCount==0)
			next();
	}
	void threadedTest(void) {
#if 1
		//Reflect a value off each processor:
		for (int pe=0;pe<CkNumPes();pe+=2) {
			CkCallback cb(CkCallback::resumeThread);
			int expectedVal=237+13*pe;
			gp[pe].reflect(cb,expectedVal);
			callbackMsg *m=(callbackMsg *)(cb.thread_delay());
			int gotVal=msg2val(m);
			if (gotVal!=expectedVal) CkAbort("Threaded callback returned wrong value");
		}
#else
		//Reflect a value off each processor:
		for (int pe=0;pe<CkNumPes();pe+=2) {
			callbackMsg *m;
			int expectedVal=237+13*pe;
			gp[pe].reflect(CkCallbackResumeThread((void*&)m),expectedVal);
			int gotVal=msg2val(m);
                	if (gotVal!=expectedVal) CkAbort("Threaded callback returned wrong value");
		}
#endif
		megatest_finish();
	}
};

MEGATEST_REGISTER_TEST(callback,"olawlor",1)
#include "callback.def.h"


