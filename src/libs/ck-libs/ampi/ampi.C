/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#define exit exit /*Supress definition of exit in ampi.h*/
#include "ampiimpl.h"
// for strlen
#include <string.h>

//------------- startup -------------
int isRestart;
char *restartDir;
static mpi_comm_worlds mpi_worlds;

int mpi_nworlds; /*Accessed by ampif*/
int MPI_COMM_UNIVERSE[MPI_MAX_COMM_WORLDS]; /*Accessed by user code*/

int _ampi_fallback_setup_count;
CDECL void MPI_Setup(void);
FDECL void FTN_NAME(MPI_SETUP,mpi_setup)(void);

int MPI_Main_cpp(int argc,char **argv);
CDECL int MPI_Main(int argc,char **argv);
FDECL void FTN_NAME(MPI_MAIN,mpi_main)(void);

/*Main routine used when missing MPI_Setup routine*/
CDECL void MPI_Fallback_Main(int argc,char **argv)
{
  MPI_Main_cpp(argc,argv);
  MPI_Main(argc,argv);
  FTN_NAME(MPI_MAIN,mpi_main)();
}

/*Startup routine used if user *doesn't* write
  a TCHARM_User_setup routine.
 */
CDECL void MPI_Setup_Switch(void) {
  _ampi_fallback_setup_count=0;
  FTN_NAME(MPI_SETUP,mpi_setup)();
  MPI_Setup();
  if (_ampi_fallback_setup_count==2)
  { //Missing MPI_Setup in both C and Fortran:
    MPI_Register_main(MPI_Fallback_Main,"default");
  }
}


static int nodeinit_has_been_called=0;
CtvDeclare(ampiParent*, ampiPtr);
static void ampiNodeInit(void)
{
  CtvInitialize(ampiParent*, ampiPtr);
  mpi_nworlds=0;
  for(int i=0;i<MPI_MAX_COMM_WORLDS; i++)
  {
    MPI_COMM_UNIVERSE[i] = MPI_COMM_WORLD+1+i;
  }
  TCHARM_Set_fallback_setup(MPI_Setup_Switch);
  nodeinit_has_been_called=1;
}

class MPI_threadstart_t {
public:
	MPI_MainFn fn;
	MPI_threadstart_t() {}
	MPI_threadstart_t(MPI_MainFn fn_)
		:fn(fn_) {}
	void start(void) {
		char **argv=CkGetArgv();
		int argc=CkGetArgc();
		(fn)(argc,argv);
	}
	void pup(PUP::er &p) {
		p|fn;
	}
};
PUPmarshall(MPI_threadstart_t);

extern "C" void MPI_threadstart(void *data)
{
	MPI_threadstart_t t;
	pupFromBuf(data,t);
	t.start();
}

void ampiCreateMain(MPI_MainFn mainFn)
{
	int _nchunks=TCHARM_Get_num_chunks();
	//isRestart = CmiGetArgString(CkGetArgv(), "+restart", &restartDir);
	//Make a new threads array
	MPI_threadstart_t s(mainFn);
	memBuf b; pupIntoBuf(b,s);
	TCHARM_Create_data( _nchunks,MPI_threadstart,
			  b.getData(), b.getSize());
}

static const char *copyCountedStr(const char *src,int len)
{
	char *ret=new char[len+1];
	memcpy(ret,src,len);
	ret[len] = '\0';
	return ret;
}

static void ampiAttach(const char *name,int namelen)
{
	TCharmSetupCookie *tc=TCharmSetupCookie::get();
	if (!tc->hasThreads())
		CkAbort("You must create a thread array with TCharmCreate before calling MPI_Attach!\n");
	int _nchunks=tc->getNumElements();
	CkArrayID threads=tc->getThreads();

	//Allocate the next communicator
	if(mpi_nworlds == MPI_MAX_COMM_WORLDS)
	{
		CkAbort("AMPI> Number of registered comm_worlds exceeded limit.\n");
	}
	int new_idx=mpi_nworlds++;
	MPI_Comm new_world=MPI_COMM_WORLD+1+new_idx;

	//Create and attach the ampiParent array
        CkArrayOptions opts(_nchunks);
        opts.bindTo(threads);
	CProxy_ampiParent parent;
	parent=CProxy_ampiParent::ckNew(new_world,threads,opts);
	tc->addClient(parent);

	//Make a new ampi array
	CkArrayID empty;
	ampiCommStruct emptyComm(new_world,empty,_nchunks);
	CProxy_ampi arr;
	arr=CProxy_ampi::ckNew(parent,emptyComm,opts);

	//Record info. in the mpi_worlds array
	mpi_worlds[new_idx].comm=ampiCommStruct(new_world,arr,_nchunks);
	mpi_worlds[new_idx].name = copyCountedStr(name,namelen);
}

//-------------------- ampiParent -------------------------
ampiParent::ampiParent(MPI_Comm worldNo_,CProxy_TCharm threads_,int isRestart) {
  worldNo=worldNo_;
  //threads=threads_;
  //thread=threads[thisIndex].ckLocal();
  worldPtr=NULL;
  myDDT=&myDDTsto;
  //if (thread==NULL) CkAbort("AMPIParent cannot find its thread!\n");
}
ampiParent::ampiParent(MPI_Comm worldNo_,CProxy_TCharm threads_)
{
  worldNo=worldNo_;
  threads=threads_;
  thread=NULL;
  worldPtr=NULL;
  myDDT=&myDDTsto;
  prepareCtv();
}
ampiParent::ampiParent(CkMigrateMessage *msg) {
  thread=NULL;
  worldPtr=NULL;
  myDDT=&myDDTsto;
}
void ampiParent::pup(PUP::er &p) {
  ArrayElement1D::pup(p);
  p|threads;
  p|worldStruct;
  myDDT->pup(p);
  p|splitComm;
  p|pers;
}
void ampiParent::prepareCtv(void) {
  thread=threads[thisIndex].ckLocal();
  if (thread==NULL) CkAbort("AMPIParent cannot find its thread!\n");
  CtvAccessOther(thread->getThread(),ampiPtr) = this;
}

void ampiParent::ckJustMigrated(void) {
  ArrayElement1D::ckJustMigrated();
  prepareCtv();
}

ampiParent::~ampiParent() {
}


//Children call this when they are first created or just migrated
TCharm *ampiParent::registerAmpi(ampi *ptr,ampiCommStruct s,bool forMigration)
{
  if (thread==NULL) prepareCtv(); //Prevents CkJustMigrated race condition

  if (s.getComm()>=MPI_COMM_WORLD)
  { //We now have our COMM_WORLD-- register it
    //Note that split communicators don't keep a raw pointer, so
    //they don't need to re-register on migration.
     if (worldPtr!=NULL) CkAbort("One ampiParent has two MPI_COMM_WORLDs");
     worldPtr=ptr;
     worldStruct=s;
  }

  if (!forMigration)
  { //Register the new communicator:
     if (s.getComm()>=MPI_COMM_WORLD)
     { //We finally have our COMM_WORLD--start the thread
       thread->ready();
     }
     else if (isSplit(s.getComm())) {
       splitChildRegister(s);
     }
     else
       CkAbort("ampiParent recieved child with bad communicator");
  }

  return thread;
}

void ampiParent::checkpoint(int len, char dname[]){
  char str[256];
  sprintf(str, "%s/%d.cpt",dname, thisIndex);
  ckCheckpoint(str);
}

void ampiParent::restart(int len, char dname[]){
  char str[256];
  sprintf(str, "%s/%d.cpt",dname, thisIndex);
  ckRestart(str);
CkPrintf("[%d]ampiParent::restart this=%p\n",thisIndex,this);
}

//----------------------- ampi -------------------------
ampi::ampi()
{
  parent=NULL;
  thread=NULL;
  msgs=NULL;
  waitingForGeneric=0;
  seqEntries=-1;
}

ampi::ampi(CkArrayID parent_,const ampiCommStruct &s)
   :parentProxy(parent_)
{
  parent=NULL;
  thread=NULL;

  myComm=s; myComm.setArrayID(thisArrayID);
  myRank=myComm.getRankForIndex(thisIndex);

  findParent(false);

  msgs = CmmNew();
  nbcasts = 0;
  waitingForGeneric=0;

  seqEntries=parent->numElements;
  oorder = new AmpiSeqQ[seqEntries];
  nextseq = new int[seqEntries];
  for(int i=0;i<seqEntries;i++) {
    nextseq[i] = 0;
    oorder[i].init();
  }
}

ampi::ampi(CkMigrateMessage *msg)
{
  parent=NULL;
  thread=NULL;
  msgs=NULL;
  waitingForGeneric=0;
  seqEntries=-1;
}

void ampi::ckJustMigrated(void)
{
	ArrayElement1D::ckJustMigrated();
	findParent(true);
}
void ampi::findParent(bool forMigration) {
	parent=parentProxy[thisIndex].ckLocal();
	if (parent==NULL) CkAbort("AMPI can't find its parent!");
	thread=parent->registerAmpi(this,myComm,forMigration);
	if (thread==NULL) CkAbort("AMPI can't find its thread!");
}

void ampi::pup(PUP::er &p)
{
  if(!p.isUserlevel())
    ArrayElement1D::pup(p);//Pack superclass
  p|parentProxy;
  p|myComm;
  p|myRank;
  p|parentProxy;
  p|nbcasts;

  msgs=CmmPup((pup_er)&p,msgs);

  p|seqEntries;
  if(p.isUnpacking())
  {
    oorder = new AmpiSeqQ[seqEntries];
    nextseq = new int[seqEntries];
  }
  for(int i=0; i<seqEntries; i++) p | oorder[i];
  p(nextseq, seqEntries);
}

ampi::~ampi()
{
  delete[] oorder;
  delete[] nextseq;
  CmmFree(msgs);
}

//------------------- maintainance -----------------
#if 1
//Need to figure out how to support checkpoint/restart properly
void ampi::stopthread(){
  thread->stop();
}

void ampi::checkpoint(int len, char dname[])
{
  char str[256];
  sprintf(str, "%s/%d.cpt",dname,thisIndex);
  ckCheckpoint(str);
}

void ampi::checkpointthread(int len, char dname[]){
  char str[256];
  sprintf(str, "%s/thread%d.cpt", dname, thisIndex);
  thread->ckCheckpoint(str);
  thread->resume();
}

void ampi::restart(int len, char dname[]){
  char str[256];
  sprintf(str, "%s/%d.cpt",dname,thisIndex);
  ckRestart(str);
CkPrintf("[%d]ampi::restart this=%p\n",thisIndex,this);
}

void ampi::restartthread(int len, char dname[]){
//CkPrintf("[%d]ampi::restartthread\n",thisIndex);
  char str[256];
  sprintf(str, "%s/thread%d.cpt", dname, thisIndex);
//CkPrintf("[%d] me: %p\n", thisIndex, CthSelf());
  thread->clear();
  thread->ckRestart(str);
  thread->start();
CkPrintf("[%d]ampi::restartthread end\n",thisIndex);
}

/*
CDECL void MPI_Checkpoint(char *dirname)
{
  mkdir(dirname, 0777);
  ampiParent *ptr = getAmpiParent();
  ptr->cthread_id = CthSelf();
  int idx = ptr->thisIndex;
  CProxy_ampi aproxy(ampimain::mpi_comms[ptr->commidx].aid);
  aproxy[idx].checkpoint(new DirMsg(dirname));
  ptr->stop_running();
  CthSuspend();
  ptr = getAmpiParent();
  if(ptr->cthread_id != 0)
    CkAbort("cthread_id not 0 upon return !!\n");
  ptr->start_running();
}
*/
#endif

//------------------------ Communicator Splitting ---------------------

class ampiSplitKey {
public:
	int color; //New class of processes we'll belong to
	int key; //To determine rank in new ordering
	int rank; //Rank in old ordering
	ampiSplitKey() {}
	ampiSplitKey(int color_,int key_,int rank_)
		:color(color_), key(key_), rank(rank_) {}
};

void ampi::split(int color,int key,MPI_Comm *dest)
{
	ampiSplitKey splitKey(color,key,getRank());
	int rootIdx=myComm.getIndexForRank(0);
	CkCallback cb(CkIndex_ampi::splitPhase1(0),CkArrayIndex1D(rootIdx),myComm.getProxy());
	contribute(sizeof(splitKey),&splitKey,CkReduction::concat,cb);

	//FIXME: assumes all the new communicators will have the same MPI_Comm
	// value.  This need not be true anytime after the first split!
	MPI_Comm newComm=parent->getNextSplit();
	*dest=newComm;
	//CkPrintf("[%d (%d)] Split (%d,%d) %d suspend\n",thisIndex,getRank(),color,key,newComm);
	thread->suspend(); //Resumed by ampiParent::splitChildRegister
	//CkPrintf("[%d (%d)] Split (%d,%d) %d resume\n",thisIndex,getRank(),color,key,newComm);
}

extern "C" int compareAmpiSplitKey(const void *a_, const void *b_) {
	const ampiSplitKey *a=(const ampiSplitKey *)a_;
	const ampiSplitKey *b=(const ampiSplitKey *)b_;
	if (a->color!=b->color) return a->color-b->color;
	if (a->key!=b->key) return a->key-b->key;
	return a->rank-b->rank;
}

void ampi::splitPhase1(CkReductionMsg *msg) 
{
	//Order the keys, which orders the ranks properly:
	int nKeys=msg->getSize()/sizeof(ampiSplitKey);
	ampiSplitKey *keys=(ampiSplitKey *)msg->getData();
	if (nKeys!=getSize()) CkAbort("ampi::splitReduce expected a split contribution from every rank!");
	qsort(keys,nKeys,sizeof(ampiSplitKey),compareAmpiSplitKey);
	
	//FIXME: assumes all the new communicators will have the same MPI_Comm
	// value.  This need not be true anytime after the first split! 
	MPI_Comm newComm=parent->getNextSplit();

	//Loop over the sorted keys, which gives us the new arrays:
	int lastColor=keys[0].color-1; //The color we're building an array for
	CProxy_ampi lastAmpi; //The array for lastColor
	int lastRoot=0; //C value for new rank 0 process for latest color
	ampiCommStruct lastComm; //Communicator info. for latest color
	for (int c=0;c<nKeys;c++) {
		if (keys[c].color!=lastColor) 
		{ //Hit a new color-- need to build a new communicator and array
			lastColor=keys[c].color;
			lastRoot=c;
			CkArrayOptions opts;
        		opts.bindTo(parentProxy);
			opts.setNumInitial(0);
			CkArrayID unusedAID; ampiCommStruct unusedComm;
			lastAmpi=CProxy_ampi::ckNew(unusedAID,unusedComm,opts);
			lastAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution
			
			CkPupBasicVec<int> indices; //Maps rank to array indices for new arrau
			for (int i=c;i<nKeys;i++) {
				if (keys[i].color!=lastColor) break; //Done with this color
				int idx=myComm.getIndexForRank(keys[i].rank);
				indices.push_back(idx);
			}
			
			//FIXME: create a new communicator for each color, instead of 
			// (confusingly) re-using the same MPI_Comm number for each.
			lastComm=ampiCommStruct(newComm,lastAmpi,indices.size(),indices);
		}
		int oldRank=keys[c].rank;
		int newRank=c-lastRoot;
		int newIdx=lastComm.getIndexForRank(newRank);
		
		//CkPrintf("[%d (%d)] Split (%d,%d) %d insert\n",newIdx,newRank,keys[c].color,keys[c].key,newComm);
		lastAmpi[newIdx].insert(parentProxy,lastComm);
	}
	
	delete msg;
}

//...newly created array elements register with the parent, which calls:
void ampiParent::splitChildRegister(const ampiCommStruct &s) {
	int idx=s.getComm()-MPI_COMM_FIRST_SPLIT;
	if (splitComm.size()>=idx) {
		splitComm.setSize(idx+1); 
		splitComm.length()=idx+1;
	}
	splitComm[idx]=new ampiCommStruct(s);
	thread->resume(); //Matches suspend at end of ampi::split
}


//------------------------ communication -----------------------

const ampiCommStruct &universeComm2proxy(MPI_Comm universeNo)
{
  if (universeNo>MPI_COMM_WORLD) {
    int worldDex=universeNo-MPI_COMM_WORLD-1;
    if (worldDex>=mpi_nworlds) 
      CkAbort("Bad world communicator passed to universeComm2proxy");
    return mpi_worlds[worldDex].comm;
  }
  CkAbort("Bad communicator passed to universeComm2proxy");
}



void
ampi::generic(AmpiMsg* msg)
{
  if(msg->seq != -1) {
    int src = msg->src;
    oorder[src].put(msg->seq, msg);
    while((msg=oorder[src].get())!=0) {
      inorder(msg);
    }
  } else { //Cross-world or system messages are unordered
    inorder(msg);
  }
  if(waitingForGeneric)
	  thread->resume();
}

void
ampi::inorder(AmpiMsg* msg)
{
  int tags[3];
  tags[0] = msg->tag; tags[1] = msg->src; tags[2] = msg->comm;
  CmmPut(msgs, 3, tags, msg);
}

AmpiMsg *ampi::makeAmpiMsg(int t,int s,const void *buf,int count,int type,MPI_Comm destcomm,int seqNo)
{
  CkDDT_DataType *ddt = getDDT()->getType(type);
  int len = ddt->getSize(count);
  AmpiMsg *msg = new (&len, 0) AmpiMsg(seqNo, t, s, len, destcomm);
  ddt->serialize((char*)buf, (char*)msg->data, count, 1);
  return msg;
}

void
ampi::send(int t, int s, const void* buf, int count, int type,  int rank, MPI_Comm destcomm)
{
  const ampiCommStruct &dest=comm2proxy(destcomm);
  int idx = dest.getIndexForRank(rank);
  int seq = -1;
  if (destcomm<=MPI_COMM_WORLD && t<=MPI_TAG_UB)
  { //Not cross-module: set seqno
     seq = nextseq[idx]++;
  }

  dest.getProxy()[idx].generic(makeAmpiMsg(t,s,buf,count,type,destcomm,seq));
}

void
ampi::sendraw(int t, int s, void* buf, int len, CkArrayID aid, int idx)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(-1, t, s, len, MPI_COMM_WORLD);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void
ampi::recv(int t, int s, void* buf, int count, int type, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  CkDDT_DataType *ddt = getDDT()->getType(type);
  int len = ddt->getSize(count);
  waitingForGeneric=1;
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
    if (msg) break;
    thread->suspend();
  }
  waitingForGeneric=0;
  if(sts)
    ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
  if (msg->length > len) {
    CkError("AMPI: (type=%d, count=%d) Expecting msg of len %d, received %d\n",
            type, count, len, msg->length);
    CkAbort("Exiting.\n");
  }
  ddt->serialize((char*)buf, (char*)msg->data, msg->length/(ddt->getSize(1)), (-1));
  delete msg;
}

void
ampi::probe(int t, int s, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
    if (msg) break;
    thread->schedule();
  }
  if(sts)
    ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
}

int 
ampi::iprobe(int t, int s, int comm, int *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  tags[0] = t; tags[1] = s; tags[2] = comm;
  msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
  if (msg) {
    if(sts)
      ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
    return 1;
  }
  thread->schedule();
  return 0;
}


void 
ampi::bcast(int root, void* buf, int count, int type,MPI_Comm destcomm)
{
  const ampiCommStruct &dest=comm2proxy(destcomm);
  int rootDex=dest.getIndexForRank(root);
  if(rootDex==thisIndex) {
    /* Broadcast my message to the array proxy */
    dest.getProxy().generic(makeAmpiMsg(MPI_BCAST_TAG,nbcasts,buf,count,type,destcomm,-1));
  }
  recv(MPI_BCAST_TAG, nbcasts, buf, count, type, destcomm);
  nbcasts++;
}

void
ampi::bcastraw(void* buf, int len, CkArrayID aid)
{
  AmpiMsg *msg = new (&len, 0) AmpiMsg(-1, MPI_BCAST_TAG, 0, len, 0);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa.generic(msg);
}


//------------------ External Interface -----------------

static ampiParent *getAmpiParent(void) {
  ampiParent *p = CtvAccess(ampiPtr);
  if (p==NULL) CkAbort("Cannot call MPI routines before AMPI is initialized.\n");
  return p;
}

static ampi *getAmpiInstance(MPI_Comm comm) {
  return getAmpiParent()->comm2ampi(comm);
}

CDECL void MPI_Migrate(void)
{
  AMPIAPI("MPI_Migrate");
  TCHARM_Migrate();
}

CDECL int MPI_Init(int *argc, char*** argv)
{
  if (nodeinit_has_been_called) {
    AMPIAPI("MPI_Init");
    return 0;
  }
  else
  { /* Charm hasn't been started yet! */
    CkAbort("Charm Uninitialized!");
  }
}

CDECL int MPI_Initialized(int *isInit)
{
  AMPIAPI("MPI_Initialized");
  *isInit=nodeinit_has_been_called;
  return 0;
}

CDECL int MPI_Comm_rank(MPI_Comm comm, int *rank)
{
  AMPIAPI("MPI_Comm_rank");
  *rank = getAmpiInstance(comm)->getRank();
  return 0;
}

CDECL
int MPI_Comm_size(MPI_Comm comm, int *size)
{
  AMPIAPI("MPI_Comm_size");
  *size = getAmpiInstance(comm)->getSize();
  return 0;
}

CDECL void MPI_Exit(int /*exitCode*/)
{
	AMPIAPI("MPI_Exit");
	TCHARM_Done();
}
FDECL void FTN_NAME(MPI_EXIT,mpi_exit)(int *exitCode)
{
	MPI_Exit(*exitCode);
}

CDECL
int MPI_Finalize(void)
{
  AMPIAPI("MPI_Finalize");
  MPI_Exit(0);
  return 0;
}

CDECL
int MPI_Send(void *msg, int count, MPI_Datatype type, int dest,
                        int tag, MPI_Comm comm)
{
  AMPIAPI("MPI_Send");
  ampi *ptr = getAmpiInstance(comm);
  ptr->send(tag, ptr->getRank(), msg, count, type, dest, comm);
  return 0;
}

//FIXME: This doesn't give the semantics of SSEND:
CDECL
int MPI_Ssend(void *msg, int count, MPI_Datatype type, int dest,
                        int tag, MPI_Comm comm)
{
  AMPIAPI("MPI_Ssend");
  ampi *ptr = getAmpiInstance(comm);
  ptr->send(tag, ptr->getRank(), msg, count, type, dest, comm);
  return 0;
}

CDECL
int MPI_Recv(void *msg, int count, MPI_Datatype type, int src, int tag,
              MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("MPI_Recv");
  ampi *ptr = getAmpiInstance(comm);
  ptr->recv(tag,src,msg,count,type, comm, (int*) status);
  return 0;
}

CDECL
int MPI_Probe(int src, int tag, MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("MPI_Probe");
  ampi *ptr = getAmpiInstance(comm);
  ptr->probe(tag,src, comm, (int*) status);
  return 0;
}

CDECL
int MPI_Iprobe(int src,int tag,MPI_Comm comm,int *flag,MPI_Status *status)
{
  AMPIAPI("MPI_Iprobe");
  ampi *ptr = getAmpiInstance(comm);
  *flag = ptr->iprobe(tag,src,comm,(int*) status);
  return 0;
}

CDECL
int MPI_Sendrecv(void *sbuf, int scount, int stype, int dest,
                  int stag, void *rbuf, int rcount, int rtype,
                  int src, int rtag, MPI_Comm comm, MPI_Status *sts)
{
  AMPIAPI("MPI_Sendrecv");
  int se=MPI_Send(sbuf,scount,stype,dest,stag,comm);
  int re=MPI_Recv(rbuf,rcount,rtype,src,rtag,comm,sts);
  if (se) return se;
  else return re;
}


CDECL
int MPI_Barrier(MPI_Comm comm)
{
  AMPIAPI("MPI_Barrier");
  //HACK: Use collective operation as a barrier.
  MPI_Allreduce(NULL,NULL,0,MPI_INT,MPI_SUM,comm);
  return 0;
}

CDECL
int MPI_Bcast(void *buf, int count, MPI_Datatype type, int root, 
                         MPI_Comm comm)
{
  AMPIAPI("MPI_Bcast");
  ampi *ptr = getAmpiInstance(comm);
  ptr->bcast(root, buf, count, type,comm);
  return 0;
}

static CkReduction::reducerType
getReductionType(int type, int op)
{
  CkReduction::reducerType mytype;
  switch(op) {
    case MPI_MAX :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::max_float; break;
        case MPI_INT : mytype = CkReduction::max_int; break;
        case MPI_DOUBLE : mytype = CkReduction::max_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case MPI_MIN :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::min_float; break;
        case MPI_INT : mytype = CkReduction::min_int; break;
        case MPI_DOUBLE : mytype = CkReduction::min_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case MPI_SUM :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::sum_float; break;
        case MPI_INT : mytype = CkReduction::sum_int; break;
        case MPI_DOUBLE : mytype = CkReduction::sum_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    case MPI_PROD :
      switch(type) {
        case MPI_FLOAT : mytype = CkReduction::product_float; break;
        case MPI_INT : mytype = CkReduction::product_int; break;
        case MPI_DOUBLE : mytype = CkReduction::product_double; break;
        default:
          ckerr << "Type " << type << " not supported." << endl;
          CmiAbort("exiting");
      }
      break;
    default:
      ckerr << "Op " << op << " not supported." << endl;
      CmiAbort("exiting");
  }
  return mytype;
}

void ampi::reduceResult(CkReductionMsg *msg)
{
  ampi::sendraw(MPI_REDUCE_TAG, 0, msg->getData(), msg->getSize(),
             thisArrayID,thisIndex);
  delete msg;
}

static CkReductionMsg *makeRednMsg(CkDDT_DataType *ddt,const void *inbuf,int count,int type,MPI_Op op)
{
  CkReduction::reducerType redtype = getReductionType(type,op);
  int size = ddt->getSize(count);
  CkReductionMsg *msg=CkReductionMsg::buildNew(size,NULL,redtype);
  ddt->serialize((char*)inbuf, (char*)msg->getData(), count, 1);
  return msg;
}

CDECL
int MPI_Reduce(void *inbuf, void *outbuf, int count, int type, MPI_Op op,
                int root, MPI_Comm comm)
{
  AMPIAPI("MPI_Reduce");
  ampi *ptr = getAmpiInstance(comm);
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,op);
  int rootIdx=ptr->comm2proxy(comm).getIndexForRank(root);
  CkCallback reduceCB(CkIndex_ampi::reduceResult(0),CkArrayIndex1D(rootIdx),ptr->getProxy(),true);
  msg->setCallback(reduceCB);
  ptr->contribute(msg);

  if (ptr->thisIndex == rootIdx)
  /*HACK: Use recv() to block until reduction data comes back*/
    ptr->recv(MPI_REDUCE_TAG, 0, outbuf, count, type, MPI_COMM_WORLD);
  return 0;
}

CDECL
int MPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                   MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("MPI_Allreduce");
  ampi *ptr = getAmpiInstance(comm);
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,op);
  CkCallback allreduceCB(CkIndex_ampi::reduceResult(0),ptr->getProxy());
  msg->setCallback(allreduceCB);
  ptr->contribute(msg);
  
  /*HACK: Use recv() to block until the reduction data comes back*/
  ptr->recv(MPI_REDUCE_TAG, 0, outbuf, count, type, MPI_COMM_WORLD);
  return 0;
}

CDECL
double MPI_Wtime(void)
{
  return TCHARM_Wall_timer();
}


/********************************
  Persistent requests are handled in a hideous fashion.
  They need to be rewritten in an object-oriented way to cut
  down on the ridiculous code duplication here.
*/
ampiPersRequests::ampiPersRequests() {
  nrequests = 0;
  nirequests = 0;
  firstfree = 0;
  int i;
  for(i=0;i<100;i++) {
    irequests[i].nextfree = (i+1)%100;
    irequests[i].prevfree = ((i-1)+100)%100;
  }
}

void ampiPersRequests::pup(PUP::er &p) {
//This was Milind's explanation for not pup'ing persistent requests (it's a lie):
  // persistent comm requests will have to be re-registered after
  // migration anyway, so no need to pup them
  // migrate is called only when all irequests are complete, so no need
  // to pup them as well.
}

static ampiPersRequests *getPers(void) {
  return &getAmpiParent()->pers;
}

CDECL
int MPI_Start(MPI_Request *reqnum)
{
  AMPIAPI("MPI_Start");
  ampiPersRequests *ptr = getPers();
  if(*reqnum >= ptr->nrequests) {
    CkAbort("Invalid persistent Request..\n");
  }
  PersReq *req = &(ptr->requests[*reqnum]);
  &(ptr->requests[*reqnum]);
  if(req->sndrcv == 1) { // send request
    ampi *aptr=getAmpiInstance(req->comm);
    aptr->send(req->tag, aptr->getRank(), req->buf, req->count, req->type,
              req->proc, req->comm);
  }
  return 0;
}

CDECL
int MPI_Waitall(int count, MPI_Request *request, MPI_Status *sts)
{
  AMPIAPI("MPI_Waitall");
  ampiPersRequests *ptr = getPers();
  int i;
  for(i=0;i<count;i++) {
    if(request[i] == (-1))
      continue;
    if(request[i] < 100) { // persistent request
      PersReq *req = &(ptr->requests[request[i]]);
      if(req->sndrcv == 2) { // recv request
        getAmpiInstance(req->comm)->recv(req->tag, req->proc, req->buf, req->count,
                  req->type, req->comm, (int*)(sts+i));
      }
    } else { // irecv request
      int index = request[i] - 100;
      PersReq *req = &(ptr->irequests[index]);
      getAmpiInstance(req->comm)->recv(req->tag, req->proc, req->buf, req->count,
                req->type, req->comm, (int*) (sts+i));
      // now free the request
      ptr->nirequests--;
      PersReq *ireq = &(ptr->irequests[0]);
      req->nextfree = ptr->firstfree;
      req->prevfree = ireq[ptr->firstfree].prevfree;
      ireq[req->prevfree].nextfree = index;
      ireq[req->nextfree].prevfree = index;
      ptr->firstfree = index;
    }
  }
  return 0;
}

CDECL
int MPI_Waitany(int count, MPI_Request *request, int *idx, MPI_Status *sts)
{
  AMPIAPI("MPI_Waitany");
  ampiPersRequests *ptr = getPers();
  while(1) {
    for(*idx=0;(*idx)<count;(*idx)++) {
      if(request[*idx] == (-1))
        return 0;
      if(request[*idx] < 100) { // persistent request
        PersReq *req = &(ptr->requests[request[*idx]]);
        if(req->sndrcv == 2) { // recv request
	  ampi *aptr=getAmpiInstance(req->comm);
          if(aptr->iprobe(req->tag, req->proc, req->comm, (int*) sts)) {
           aptr->recv(req->tag, req->proc, req->buf, req->count,
                      req->type, req->comm, (int*)sts);
            return 0;
          }
        }
      } else { // irecv request
        int index = request[*idx] - 100;
        PersReq *req = &(ptr->irequests[index]);
	ampi *aptr=getAmpiInstance(req->comm);
        if(aptr->iprobe(req->tag, req->proc, req->comm, (int*) sts)) {
          aptr->recv(req->tag, req->proc, req->buf, req->count,
                    req->type, req->comm, (int*)sts);
          // now free the request
          ptr->nirequests--;
          PersReq *ireq = &(ptr->irequests[0]);
          req->nextfree = ptr->firstfree;
          req->prevfree = ireq[ptr->firstfree].prevfree;
          ireq[req->prevfree].nextfree = index;
          ireq[req->nextfree].prevfree = index;
          ptr->firstfree = index;
          return 0;
        }
      }
    }
  }
  // should never come here
  return 0;
}

CDECL
int MPI_Wait(MPI_Request *request, MPI_Status *sts)
{
  AMPIAPI("MPI_Wait");
  ampiPersRequests *ptr = getPers();
  if(*request == (-1))
      return 0;
  if(*request < 100) { // persistent request
    PersReq *req = &(ptr->requests[*request]);
    if(req->sndrcv == 2) { // recv request
      getAmpiInstance(req->comm)->recv(req->tag, req->proc, req->buf, req->count,
                req->type, req->comm, (int*)sts);
    }
  } else { // irecv request
    int index = *request - 100;
    PersReq *req = &(ptr->irequests[index]);
    getAmpiInstance(req->comm)->recv(req->tag, req->proc, req->buf, req->count,
              req->type, req->comm, (int*) sts);
    // now free the request
    ptr->nirequests--;
    PersReq *ireq = &(ptr->irequests[0]);
    req->nextfree = ptr->firstfree;
    req->prevfree = ireq[ptr->firstfree].prevfree;
    ireq[req->prevfree].nextfree = index;
    ireq[req->nextfree].prevfree = index;
    ptr->firstfree = index;
  }
  return 0;
}

CDECL
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("MPI_Test");
  ampiPersRequests *ptr = getPers();
  if(*request==(-1)) {
    *flag = 1;
    return 0;
  }
  if(*request < 100) { // persistent request
    PersReq *req = &(ptr->requests[*request]);
    if(req->sndrcv == 2) // recv request
      *flag = getAmpiInstance(req->comm)->iprobe(req->tag, req->proc, req->comm, (int*)sts);
    else
      *flag = 1; // send request
  } else { // irecv request
    int index = *request - 100;
    PersReq *req = &(ptr->irequests[index]);
    *flag = getAmpiInstance(req->comm)->iprobe(req->tag, req->proc, req->comm, (int*) sts);
  }
  return 0;
}

CDECL
int MPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("MPI_Testall");
  int i;
  int tmpflag;
  *flag = 1;
  for(i=0;i<count;i++)
  {
    MPI_Test(&request[i], &tmpflag, sts+i);
    *flag = *flag && tmpflag;
  }
  return 0;
}

CDECL
int MPI_Recv_init(void *buf, int count, int type, int src, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("MPI_Recv_init");

  ampiPersRequests *ptr = getPers();

  if(ptr->nrequests == 100) {
    CmiAbort("Too many persistent commrequests.\n");
  }
  ptr->requests[ptr->nrequests].sndrcv = 2;
  ptr->requests[ptr->nrequests].buf = buf;
  ptr->requests[ptr->nrequests].count = count;
  ptr->requests[ptr->nrequests].type = type;
  ptr->requests[ptr->nrequests].proc = src;
  ptr->requests[ptr->nrequests].tag = tag;
  ptr->requests[ptr->nrequests].comm = comm;
  *req = ptr->nrequests;
  ptr->nrequests ++;
  return 0;
}

CDECL
int MPI_Send_init(void *buf, int count, int type, int dest, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("MPI_Send_init");
  ampiPersRequests *ptr = getPers();
  if(ptr->nrequests == 100) {
    CmiAbort("Too many persistent commrequests.\n");
  }
  ptr->requests[ptr->nrequests].sndrcv = 1;
  ptr->requests[ptr->nrequests].buf = buf;
  ptr->requests[ptr->nrequests].count = count;
  ptr->requests[ptr->nrequests].type = type;
  ptr->requests[ptr->nrequests].proc = dest;
  ptr->requests[ptr->nrequests].tag = tag;
  ptr->requests[ptr->nrequests].comm = comm;
  *req = ptr->nrequests;
  ptr->nrequests ++;
  return 0;
}

static CkDDT *getDDT(void) {
  return getAmpiParent()->myDDT;
}

CDECL
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, 
                         MPI_Datatype *newtype)
{
  AMPIAPI("MPI_Type_contiguous");
  getDDT()->newContiguous(count, oldtype, newtype); 
  return 0;
}

extern  "C"  
int MPI_Type_vector(int count, int blocklength, int stride, 
                     MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_vector");
  getDDT()->newVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, 
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_hvector");
  getDDT()->newHVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_indexed(int count, int* arrBlength, int* arrDisp, 
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_indexed");
  getDDT()->newIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp, 
                       MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_hindexed");
  getDDT()->newHIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

extern  "C"  
int MPI_Type_struct(int count, int* arrBlength, int* arrDisp, 
                     MPI_Datatype* oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("MPI_Type_struct");
  getDDT()->newStruct(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

CDECL
int MPI_Type_commit(MPI_Datatype *datatype)
{
  AMPIAPI("MPI_Type_commit");
  return 0;
}

CDECL
int MPI_Type_free(MPI_Datatype *datatype)
{
  AMPIAPI("MPI_Type_free");
  getDDT()->freeType(datatype);
  return 0;
}


CDECL
int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent)
{
  AMPIAPI("MPI_Type_extent");
  getDDT()->getExtent(datatype);
  return 0;
}

CDECL
int MPI_Type_size(MPI_Datatype datatype, int *size)
{
  AMPIAPI("MPI_Type_size");
  getDDT()->getSize(datatype);
  return 0;
}

CDECL
int MPI_Isend(void *buf, int count, MPI_Datatype type, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Isend");
  ampi *ptr = getAmpiInstance(comm);

  ptr->send(tag, ptr->getRank(), buf, count, type, dest, comm);
  *request = (-1);
  return 0;
}

CDECL
int MPI_Issend(void *buf, int count, MPI_Datatype type, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Issend");
  ampi *ptr = getAmpiInstance(comm);

  ptr->send(tag, ptr->getRank(), buf, count, type, dest, comm);
  *request = (-1);
  return 0;
}

CDECL
int MPI_Irecv(void *buf, int count, MPI_Datatype type, int src,
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Irecv");

  ampiPersRequests *ptr = getPers();
  if(ptr->nirequests == 100) {
    CmiAbort("Too many Irecv requests.\n");
  }

  PersReq *req = &(ptr->irequests[ptr->firstfree]);
  req->sndrcv = 2;
  req->buf = buf;
  req->count = count;
  req->type = type;
  req->proc = src;
  req->tag = tag;
  req->comm = comm;
  *request = ptr->firstfree + 100;
  ptr->nirequests ++;
  // remove this request from the free list
  PersReq *ireq = &(ptr->irequests[0]);
  ptr->firstfree = ireq[ptr->firstfree].nextfree;
  ireq[req->nextfree].prevfree = req->prevfree;
  ireq[req->prevfree].nextfree = req->nextfree;
  return 0;
}

CDECL
int MPI_Ireduce(void *sendbuf, void *recvbuf, int count, int type, MPI_Op op, int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("MPI_Ireduce");
  ampi *ptr = getAmpiInstance(comm);
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),sendbuf,count,type,op);
  int rootIdx=ptr->comm2proxy(comm).getIndexForRank(root);
  CkCallback reduceCB(CkIndex_ampi::reduceResult(0),CkArrayIndex1D(rootIdx),ptr->getProxy(),true);
  msg->setCallback(reduceCB);
  ptr->contribute(msg);

  if (ptr->thisIndex == rootIdx){
  ampiPersRequests *ptr = getPers();
  if(ptr->nirequests == 100) {
    CmiAbort("Too many Irecv requests in MPI_Ireduce.\n");
  }

  // using irecv instead recv to non-block the call and get request pointer
  PersReq *req = &(ptr->irequests[ptr->firstfree]);
    req->sndrcv = 2;
    req->buf = recvbuf;
    req->count = count;
    req->type = type;
    req->proc = 0;
    req->tag = MPI_REDUCE_TAG;
    req->comm = MPI_COMM_WORLD;
    *request = ptr->firstfree + 100;
    ptr->nirequests ++;
    // remove this request from the free list
    PersReq *ireq = &(ptr->irequests[0]);
    ptr->firstfree = ireq[ptr->firstfree].nextfree;
    ireq[req->nextfree].prevfree = req->prevfree;
    ireq[req->prevfree].nextfree = req->nextfree;
  }
  return 0;
}

CDECL
int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int *recvcounts, int *displs, 
                   MPI_Datatype recvtype, MPI_Comm comm) 
{
  AMPIAPI("MPI_Allgatherv");
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize();
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(sendbuf, sendcount, sendtype, i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
  int itemsize = dttype->getSize() ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
  AMPIAPI("MPI_Allgather");
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize();
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(sendbuf, sendcount, sendtype, i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
  int itemsize = dttype->getSize(recvcount) ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  AMPIAPI("MPI_Gatherv");
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize();
  int i;

  MPI_Send(sendbuf, sendcount, sendtype, root, MPI_GATHER_TAG, comm);

  if(ptr->getRank() == root) {
    MPI_Status status;
    CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
    int itemsize = dttype->getSize() ;
  
    for(i=0;i<size;i++) {
      MPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
               i, MPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

CDECL
int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype, 
               int root, MPI_Comm comm)
{
  AMPIAPI("MPI_Gather");
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize();
  int i;
  MPI_Send(sendbuf, sendcount, sendtype, root, MPI_GATHER_TAG, comm);

  if(ptr->getRank()==root) {
    MPI_Status status;
    CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
    int itemsize = dttype->getSize(recvcount) ;
  
    for(i=0;i<size;i++) {
      MPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
               i, MPI_GATHER_TAG, comm, &status);
    }
  }
  return 0;
}

CDECL
int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("MPI_Alltoallv");
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize();
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
  int itemsize = dttype->getSize() ;
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(((char*)sendbuf)+(itemsize*sdispls[i]), sendcounts[i], sendtype,
             i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize() ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*rdispls[i]), recvcounts[i], recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                 MPI_Comm comm)
{
  AMPIAPI("MPI_Alltoall");
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize();
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
  int itemsize = dttype->getSize(sendcount) ;
  int i;
  for(i=0;i<size;i++) {
    MPI_Send(((char*)sendbuf)+(itemsize*i), sendcount, sendtype,
             i, MPI_GATHER_TAG, comm);
  }

  MPI_Status status;
  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  
  for(i=0;i<size;i++) {
    MPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
  return 0;
}

CDECL
int MPI_Comm_dup(int comm, int *newcomm)
{
  AMPIAPI("MPI_Comm_dup");
  *newcomm = comm;
  return 0;
}

CDECL
int MPI_Comm_split(int src,int color,int key,int *dest)
{
  AMPIAPI("MPI_Comm_split");
  getAmpiInstance(src)->split(color,key,dest);
  return 0;
}

CDECL
int MPI_Comm_free(int *comm)
{
  AMPIAPI("MPI_Comm_free");
  return 0;
}

CDECL
int MPI_Abort(int comm, int errorcode)
{
  AMPIAPI("MPI_Abort");
  CkAbort("AMPI: User called MPI_Abort!\n");
  return errorcode;
}

CDECL
int MPI_Get_count(MPI_Status *sts, MPI_Datatype dtype, int *count)
{
  AMPIAPI("MPI_Get_count");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int itemsize = dttype->getSize() ;
  *count = sts->MPI_LENGTH/itemsize;
  return 0;
}

CDECL
int MPI_Pack(void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm)
{
  AMPIAPI("MPI_Pack");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize((char*)inbuf, ((char*)outbuf)+(*position), incount, 1);
  *position += (itemsize*incount);
  return 0;
}

CDECL
int MPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
              int outcount, MPI_Datatype dtype, MPI_Comm comm)
{
  AMPIAPI("MPI_Unpack");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize(((char*)inbuf+(*position)), (char*)outbuf, outcount, 1);
  *position += (itemsize*outcount);
  return 0;
}

CDECL
int MPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz)
{
  AMPIAPI("MPI_Pack_size");
  CkDDT_DataType* dttype = getDDT()->getType(datatype) ;
  return incount*dttype->getSize() ;
}

/* Error handling */
CDECL
int MPI_Error_string(int errorcode, char *string, int *resultlen)
{
  AMPIAPI("MPI_Error_string");
  const char *ret="";
  switch(errorcode) {
  case MPI_SUCCESS:
	   ret="Success";
	   break;
  default:
	   return 1;/*LIE: should be MPI_ERR_something */
  };
  *resultlen=strlen(ret);
  strcpy(string,ret);
  return MPI_SUCCESS;
}


/* Charm++ Extentions to MPI standard: */
CDECL
void MPI_Restart(char *dname)
{
  int len;
  char str[256];
  AMPIAPI("MPI_Restart");
  MPI_Barrier(MPI_COMM_WORLD);
  mkdir(dname,0777);

  ampiParent *parentptr = getAmpiParent();
  sprintf(str, "%s/ampiParent", dname);
  len = strlen(str)+1;
  parentptr->restart(len,str);

  ampi *ampiptr = getAmpiInstance(MPI_COMM_WORLD);
  sprintf(str, "%s/ampi%d", dname, ampiptr->getComm());
  len = strlen(str)+1;
  ampiptr->restart(len,str); //getProxy()[ampiptr->thisIndex].
//CkPrintf("before me: %p\n", CthSelf());
  ampiptr->getProxy()[ampiptr->thisIndex].restartthread(len,str);
  ampiptr->stopthread();
}

CDECL
void MPI_Checkpoint(char *dname)
{
  int len;
  char str[256];
  AMPIAPI("MPI_Checkpoint");
  MPI_Barrier(MPI_COMM_WORLD);
  mkdir(dname,0777);

  ampiParent *parentptr = getAmpiParent();
  sprintf(str, "%s/ampiParent", dname);
  mkdir(str, 0777);
  len = strlen(str);
  parentptr->checkpoint(len,str);

  ampi *ampiptr = getAmpiInstance(MPI_COMM_WORLD);
  sprintf(str, "%s/ampi%d", dname, ampiptr->getComm());
  mkdir(str, 0777);
  len = strlen(str)+1;
  ampiptr->checkpoint(len,str);

  ampiptr->getProxy()[ampiptr->thisIndex].checkpointthread(len,str);
  ampiptr->stopthread();
}

CDECL
void MPI_Print(char *str)
{
  AMPIAPI("MPI_Print");
  ampiParent *ptr = getAmpiParent();
  CkPrintf("[%d] %s\n", ptr->thisIndex, str);
}

CDECL
int MPI_Register(void *d, MPI_PupFn f)
{
	AMPIAPI("MPI_Register");
	return TCHARM_Register(d,f);
}

CDECL
void *MPI_Get_userdata(int idx)
{
	AMPIAPI("MPI_Get_userdata");
	return TCHARM_Get_userdata(idx);
}

CDECL void MPI_Register_main(MPI_MainFn mainFn,const char *name)
{
	AMPIAPI("MPI_Register_main");
	ampiCreateMain(mainFn);
	ampiAttach(name,strlen(name));
}
FDECL void FTN_NAME(MPI_REGISTER_MAIN,mpi_register_main)
	(MPI_MainFn mainFn,const char *name,int nameLen)
{
	AMPIAPI("MPI_register_main");
	ampiCreateMain(mainFn);
	ampiAttach(name,nameLen);
}

CDECL void MPI_Attach(const char *name)
{
	AMPIAPI("MPI_Attach");
	ampiAttach(name,strlen(name));
}
FDECL void FTN_NAME(MPI_ATTACH,mpi_attach)(const char *name,int nameLen)
{
	AMPIAPI("MPI_attach");
	ampiAttach(name,nameLen);
}

void _registerampif(void)
{
  _registerampi();
}
#include "ampi.def.h"
