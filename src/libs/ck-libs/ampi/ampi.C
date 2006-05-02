/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#define exit exit /*Supress definition of exit in ampi.h*/
#include "ampiimpl.h"
#include "tcharm.h"
#include "ampiEvents.h" /*** for trace generation for projector *****/
#include "ampiProjections.h"

#define CART_TOPOL 1
#define AMPI_PRINT_IDLE 0

/* change this define to "x" to trace all send/recv's */
#define MSG_ORDER_DEBUG(x) // x /* empty */
/* change this define to "x" to trace user calls */
#define USER_CALL_DEBUG(x) // ckout<<"vp "<<TCHARM_Element()<<": "<<x<<endl; 
#define STARTUP_DEBUG(x)  // ckout<<"ampi[pe "<<CkMyPe()<<"] "<< x <<endl; 



//------------- startup -------------
static mpi_comm_worlds mpi_worlds;

int mpi_nworlds; /*Accessed by ampif*/
int MPI_COMM_UNIVERSE[MPI_MAX_COMM_WORLDS]; /*Accessed by user code*/

/* ampiReducer: AMPI's generic reducer type 
   MPI_Op is function pointer to MPI_User_function
   so that it can be packed into AmpiOpHeader, shipped 
   with the reduction message, and then plugged into 
   the ampiReducer. 
   One little trick is the ampi::recv which receives
   the final reduction message will see additional
   sizeof(AmpiOpHeader) bytes in the buffer before
   any user data.                             */
class AmpiComplex { 
public: 
	double re, im; 
	void operator+=(const AmpiComplex &a) {
		re+=a.re;
		im+=a.im;
	}
	void operator*=(const AmpiComplex &a) {
		double nu_re=re*a.re-im*a.im;
		im=re*a.im+im*a.re;
		re=nu_re;
	}
	int operator>(const AmpiComplex &a) {
		CkAbort("Cannot compare complex numbers with MPI_MAX");
		return 0;
	}
	int operator<(const AmpiComplex &a) {
		CkAbort("Cannot compare complex numbers with MPI_MIN");
		return 0;
	}
};
typedef struct { float val; int idx; } FloatInt;
typedef struct { double val; int idx; } DoubleInt;
typedef struct { long val; int idx; } LongInt;
typedef struct { int val; int idx; } IntInt;
typedef struct { short val; int idx; } ShortInt;
typedef struct { long double val; int idx; } LongdoubleInt;
typedef struct { float val; float idx; } FloatFloat;
typedef struct { double val; double idx; } DoubleDouble;


#define MPI_OP_SWITCH(OPNAME) \
  int i; \
  switch (*datatype) { \
  case MPI_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(char); } break; \
  case MPI_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(short); } break; \
  case MPI_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int); } break; \
  case MPI_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(long); } break; \
  case MPI_UNSIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned char); } break; \
  case MPI_UNSIGNED_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned short); } break; \
  case MPI_UNSIGNED: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned int); } break; \
  case MPI_UNSIGNED_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(CmiUInt8); } break; \
  case MPI_FLOAT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(float); } break; \
  case MPI_DOUBLE: for(i=0;i<(*len);i++) { MPI_OP_IMPL(double); } break; \
  case MPI_COMPLEX: for(i=0;i<(*len);i++) { MPI_OP_IMPL(AmpiComplex); } break; \
  case MPI_DOUBLE_COMPLEX: for(i=0;i<(*len);i++) { MPI_OP_IMPL(AmpiComplex); } break; \
  case MPI_LONG_LONG_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(CmiInt8); } break; \
  default: \
    ckerr << "Type " << *datatype << " with Op "#OPNAME" not supported." << endl; \
    CmiAbort("Unsupported MPI datatype for MPI Op"); \
  };\

void MPI_MAX( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
	if(((type *)invec)[i] > ((type *)inoutvec)[i]) ((type *)inoutvec)[i] = ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_MAX)
#undef MPI_OP_IMPL
}

void MPI_MIN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
	if(((type *)invec)[i] < ((type *)inoutvec)[i]) ((type *)inoutvec)[i] = ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_MIN)
#undef MPI_OP_IMPL
}

void MPI_SUM( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
	((type *)inoutvec)[i] += ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_SUM)
#undef MPI_OP_IMPL
}

void MPI_PROD( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
	((type *)inoutvec)[i] *= ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_PROD)
#undef MPI_OP_IMPL
}

void MPI_LAND( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_INT:
  case MPI_LOGICAL:
    for(i=0;i<(*len);i++)
      ((int *)inoutvec)[i] = ((int *)inoutvec)[i] && ((int *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_LAND not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_BAND( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i; 
  switch (*datatype) {
  case MPI_INT:
    for(i=0;i<(*len);i++)
      ((int *)inoutvec)[i] = ((int *)inoutvec)[i] & ((int *)invec)[i];
    break;
  case MPI_BYTE:
    for(i=0;i<(*len);i++)
      ((char *)inoutvec)[i] = ((char *)inoutvec)[i] & ((char *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_BAND not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_LOR( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_INT:
  case MPI_LOGICAL:
    for(i=0;i<(*len);i++)
      ((int *)inoutvec)[i] = ((int *)inoutvec)[i] || ((int *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_LOR not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_BOR( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_INT:
    for(i=0;i<(*len);i++)
      ((int *)inoutvec)[i] = ((int *)inoutvec)[i] | ((int *)invec)[i];
    break;
  case MPI_BYTE:
    for(i=0;i<(*len);i++)
      ((char *)inoutvec)[i] = ((char *)inoutvec)[i] | ((char *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_BOR not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_LXOR( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_INT:
  case MPI_LOGICAL:
    for(i=0;i<(*len);i++)
      ((int *)inoutvec)[i] = (((int *)inoutvec)[i]&&(!((int *)invec)[i]))||(!(((int *)inoutvec)[i])&&((int *)invec)[i]); //emulate ^^
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_LXOR not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_BXOR( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_INT:
    for(i=0;i<(*len);i++)
      ((int *)inoutvec)[i] = ((int *)inoutvec)[i] ^ ((int *)invec)[i];
    break;
  case MPI_BYTE:
    for(i=0;i<(*len);i++)
      ((char *)inoutvec)[i] = ((char *)inoutvec)[i] ^ ((char *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_BXOR not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_MAXLOC( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_FLOAT_INT:
    for(i=0;i<(*len);i++)
      if(((FloatInt *)invec)[i].val > ((FloatInt *)inoutvec)[i].idx)
        ((FloatInt *)inoutvec)[i] = ((FloatInt *)invec)[i];
    break;
  case MPI_DOUBLE_INT:
    for(i=0;i<(*len);i++)
      if(((DoubleInt *)invec)[i].val > ((DoubleInt *)inoutvec)[i].idx)
        ((DoubleInt *)inoutvec)[i] = ((DoubleInt *)invec)[i];
    break;
  case MPI_LONG_INT:
    for(i=0;i<(*len);i++)
      if(((LongInt *)invec)[i].val > ((LongInt *)inoutvec)[i].idx)
        ((LongInt *)inoutvec)[i] = ((LongInt *)invec)[i];
    break;
  case MPI_2INT:
    for(i=0;i<(*len);i++)
      if(((IntInt *)invec)[i].val > ((IntInt *)inoutvec)[i].idx)
        ((IntInt *)inoutvec)[i] = ((IntInt *)invec)[i];
    break;
  case MPI_SHORT_INT:
    for(i=0;i<(*len);i++)
      if(((ShortInt *)invec)[i].val > ((ShortInt *)inoutvec)[i].idx)
        ((ShortInt *)inoutvec)[i] = ((ShortInt *)invec)[i];
    break;
  case MPI_LONG_DOUBLE_INT:
    for(i=0;i<(*len);i++)
      if(((LongdoubleInt *)invec)[i].val > ((LongdoubleInt *)inoutvec)[i].idx)
        ((LongdoubleInt *)inoutvec)[i] = ((LongdoubleInt *)invec)[i];
    break;
  case MPI_2FLOAT:
    for(i=0;i<(*len);i++)
      if(((FloatFloat *)invec)[i].val > ((FloatFloat *)inoutvec)[i].idx)
        ((FloatFloat *)inoutvec)[i] = ((FloatFloat *)invec)[i];
    break;
  case MPI_2DOUBLE:
    for(i=0;i<(*len);i++)
      if(((DoubleDouble *)invec)[i].val > ((DoubleDouble *)inoutvec)[i].idx)
        ((DoubleDouble *)inoutvec)[i] = ((DoubleDouble *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_MAXLOC not supported." << endl;
    CmiAbort("exiting");
  }
}
void MPI_MINLOC( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;  
  switch (*datatype) {
  case MPI_FLOAT_INT:
    for(i=0;i<(*len);i++)
      if(((FloatInt *)invec)[i].val < ((FloatInt *)inoutvec)[i].idx)
        ((FloatInt *)inoutvec)[i] = ((FloatInt *)invec)[i];
    break;
  case MPI_DOUBLE_INT:
    for(i=0;i<(*len);i++)
      if(((DoubleInt *)invec)[i].val < ((DoubleInt *)inoutvec)[i].idx)
        ((DoubleInt *)inoutvec)[i] = ((DoubleInt *)invec)[i];
    break;
  case MPI_LONG_INT:
    for(i=0;i<(*len);i++)
      if(((LongInt *)invec)[i].val < ((LongInt *)inoutvec)[i].idx)
        ((LongInt *)inoutvec)[i] = ((LongInt *)invec)[i];
    break;
  case MPI_2INT:
    for(i=0;i<(*len);i++)
      if(((IntInt *)invec)[i].val < ((IntInt *)inoutvec)[i].idx)
        ((IntInt *)inoutvec)[i] = ((IntInt *)invec)[i];
    break;
  case MPI_SHORT_INT:
    for(i=0;i<(*len);i++)
      if(((ShortInt *)invec)[i].val < ((ShortInt *)inoutvec)[i].idx)
        ((ShortInt *)inoutvec)[i] = ((ShortInt *)invec)[i];
    break;
  case MPI_LONG_DOUBLE_INT:
    for(i=0;i<(*len);i++)
      if(((LongdoubleInt *)invec)[i].val < ((LongdoubleInt *)inoutvec)[i].idx)
        ((LongdoubleInt *)inoutvec)[i] = ((LongdoubleInt *)invec)[i];
    break;
  case MPI_2FLOAT:
    for(i=0;i<(*len);i++)
      if(((FloatFloat *)invec)[i].val < ((FloatFloat *)inoutvec)[i].idx)
        ((FloatFloat *)inoutvec)[i] = ((FloatFloat *)invec)[i];
    break;
  case MPI_2DOUBLE:
    for(i=0;i<(*len);i++)
      if(((DoubleDouble *)invec)[i].val < ((DoubleDouble *)inoutvec)[i].idx)
        ((DoubleDouble *)inoutvec)[i] = ((DoubleDouble *)invec)[i];
    break;
  default:
    ckerr << "Type " << *datatype << " with Op MPI_MINLOC not supported." << endl;
    CmiAbort("exiting");
  }
}

// every msg contains a AmpiOpHeader structure before user data
// FIXME: non-commutative operations require messages be ordered by rank
CkReductionMsg *AmpiReducerFunc(int nMsg, CkReductionMsg **msgs){
  AmpiOpHeader *hdr = (AmpiOpHeader *)msgs[0]->getData();
  MPI_Datatype dtype;
  int szhdr, szdata, len;
  MPI_User_function* func;
  func = hdr->func;
  dtype = hdr->dtype;  
  szdata = hdr->szdata;
  len = hdr->len;  
  szhdr = sizeof(AmpiOpHeader);

  //Assuming extent == size
  void *ret = malloc(szhdr+szdata);
  memcpy(ret,msgs[0]->getData(),szhdr+szdata);
  for(int i=1;i<nMsg;i++){
    (*func)((void *)((char *)msgs[i]->getData()+szhdr),(void *)((char *)ret+szhdr),&len,&dtype);
  }
  CkReductionMsg *retmsg = CkReductionMsg::buildNew(szhdr+szdata,ret);
  free(ret);
  return retmsg;
}

CkReduction::reducerType AmpiReducer;

// ------------ startup support -----------
int _ampi_fallback_setup_count;
CDECL void AMPI_Setup(void);
FDECL void FTN_NAME(AMPI_SETUP,ampi_setup)(void);

FDECL void FTN_NAME(MPI_MAIN,mpi_main)(void);

/*Main routine used when missing MPI_Setup routine*/
CDECL void AMPI_Fallback_Main(int argc,char **argv)
{
  AMPI_Main_cpp(argc,argv);
  AMPI_Main(argc,argv);
  FTN_NAME(MPI_MAIN,mpi_main)();
}

void ampiCreateMain(MPI_MainFn mainFn, const char *name,int nameLen);
/*Startup routine used if user *doesn't* write
  a TCHARM_User_setup routine.
 */
CDECL void AMPI_Setup_Switch(void) {
  _ampi_fallback_setup_count=0;
  FTN_NAME(AMPI_SETUP,ampi_setup)();
  AMPI_Setup();
  if (_ampi_fallback_setup_count==2)
  { //Missing AMPI_Setup in both C and Fortran:
    ampiCreateMain(AMPI_Fallback_Main,"default",strlen("default"));
  }
}

static int nodeinit_has_been_called=0;
CtvDeclare(ampiParent*, ampiPtr);
CtvDeclare(int, ampiInitDone);
CtvDeclare(int, ampiFinalized);
CkpvDeclare(int,argvExtracted);
static int enableStreaming = 0;

static void ampiNodeInit(void)
{
  mpi_nworlds=0;
  for(int i=0;i<MPI_MAX_COMM_WORLDS; i++)
  {
    MPI_COMM_UNIVERSE[i] = MPI_COMM_WORLD+1+i;
  }
  TCHARM_Set_fallback_setup(AMPI_Setup_Switch);

  AmpiReducer = CkReduction::addReducer(AmpiReducerFunc);
  
  nodeinit_has_been_called=1;
}

#if PRINT_IDLE
static double totalidle=0.0, startT=0.0;
static int beginHandle, endHandle;
static void BeginIdle(void *dummy,double curWallTime)
{
  startT = curWallTime;
}
static void EndIdle(void *dummy,double curWallTime)
{
  totalidle += curWallTime - startT;
}
#endif

/* for fortran reduction operation table to handle mapping */
typedef MPI_Op  MPI_Op_Array[128];
CtvDeclare(int, mpi_opc);
CtvDeclare(MPI_Op_Array, mpi_ops);

static void ampiProcInit(void){
  CtvInitialize(ampiParent*, ampiPtr);
  CtvInitialize(int,ampiInitDone);
  CtvInitialize(int,ampiFinalized);

  CtvInitialize(MPI_Op_Array, mpi_ops);
  CtvInitialize(int, mpi_opc);

  CkpvInitialize(int, argvExtracted);
  CkpvAccess(argvExtracted) = 0;
  REGISTER_AMPI
  initAmpiProjections();
#if AMPI_COMLIB  
  char **argv=CkGetArgv();
  if(CkpvAccess(argvExtracted)==0){
    enableStreaming=CmiGetArgFlagDesc(argv,"+ampi_streaming","Enable streaming comlib for ampi send/recv.");
  }
#endif
}

void AMPI_Install_Idle_Timer(){
#if AMPI_PRINT_IDLE
  beginHandle = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)BeginIdle,NULL);
  endHandle = CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE,(CcdVoidFn)EndIdle,NULL);
#endif
}

void AMPI_Uninstall_Idle_Timer(){
#if AMPI_PRINT_IDLE
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,beginHandle);
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,endHandle);
#endif
}

PUPfunctionpointer(MPI_MainFn);

class MPI_threadstart_t {
public:
	MPI_MainFn fn;
	MPI_threadstart_t() {}
	MPI_threadstart_t(MPI_MainFn fn_)
		:fn(fn_) {}
	void start(void) {
		char **argv=CmiCopyArgs(CkGetArgv());
		int argc=CkGetArgc();
		(fn)(argc,argv);
	}
	void pup(PUP::er &p) {
		p|fn;
	}
};
PUPmarshall(MPI_threadstart_t);

CDECL void AMPI_threadstart(void *data)
{
	STARTUP_DEBUG("MPI_threadstart")
	MPI_threadstart_t t;
	pupFromBuf(data,t);
	t.start();
}

void ampiCreateMain(MPI_MainFn mainFn, const char *name,int nameLen)
{
	STARTUP_DEBUG("ampiCreateMain")
	int _nchunks=TCHARM_Get_num_chunks();
	//Make a new threads array:
	MPI_threadstart_t s(mainFn);
	memBuf b; pupIntoBuf(b,s);
	TCHARM_Create_data( _nchunks,AMPI_threadstart,
			  b.getData(), b.getSize());
}

/* TCharm Semaphore ID's for AMPI startup */
#define AMPI_TCHARM_SEMAID 0x00A34100 /* __AMPI__ */
#define AMPI_BARRIER_SEMAID 0x00A34200 /* __AMPI__ */

static CProxy_ampiWorlds ampiWorldsGroup;

static void init_operations()
{
  CtvInitialize(MPI_Op_Array, mpi_ops);
  int i = 0;
  MPI_Op *tab = CtvAccess(mpi_ops);
  tab[i++] = MPI_MAX;
  tab[i++] = MPI_MIN;
  tab[i++] = MPI_SUM;
  tab[i++] = MPI_PROD;
  tab[i++] = MPI_LAND;
  tab[i++] = MPI_BAND;
  tab[i++] = MPI_LOR;
  tab[i++] = MPI_BOR;
  tab[i++] = MPI_LXOR;
  tab[i++] = MPI_BXOR;
  tab[i++] = MPI_MAXLOC;
  tab[i++] = MPI_MINLOC;

  CtvInitialize(int, mpi_opc);
  CtvAccess(mpi_opc) = i;
}

/*
Called from MPI_Init, a collective initialization call:
 creates a new AMPI array and attaches it to the current
 set of TCHARM threads.
*/
static ampi *ampiInit(char **argv)
{
  if (CtvAccess(ampiInitDone)) return NULL; /* Already called ampiInit */
  STARTUP_DEBUG("ampiInit> begin")

  MPI_Comm new_world;
  int _nchunks;
  CkArrayOptions opts;
  CProxy_ampiParent parent;
  if (TCHARM_Element()==0)
  { /* I'm responsible for building the arrays: */
	STARTUP_DEBUG("ampiInit> creating arrays")

// FIXME: Need to serialize global communicator allocation in one place.
	//Allocate the next communicator
	if(mpi_nworlds == MPI_MAX_COMM_WORLDS)
	{
		CkAbort("AMPI> Number of registered comm_worlds exceeded limit.\n");
	}
	int new_idx=mpi_nworlds;
	new_world=MPI_COMM_WORLD+1+new_idx;

        //Create and attach the ampiParent array
        CkArrayID threads;
        opts=TCHARM_Attach_start(&threads,&_nchunks);
	parent=CProxy_ampiParent::ckNew(new_world,threads,opts);
	STARTUP_DEBUG("ampiInit> array size "<<_nchunks);
  }
  int *barrier = (int *)TCharm::get()->semaGet(AMPI_BARRIER_SEMAID);
  if (TCHARM_Element()==0)
  {
	//Make a new ampi array
	CkArrayID empty;
	
#if AMPI_COMLIB
	ComlibInstanceHandle ciStreaming;
	ComlibInstanceHandle ciBcast;
	ComlibInstanceHandle ciAllgather;
	ComlibInstanceHandle ciAlltoall;
	ciStreaming=CkGetComlibInstance();
	ciBcast=CkGetComlibInstance();
	ciAllgather=CkGetComlibInstance();
	ciAlltoall=CkGetComlibInstance();
#endif

        CkVec<int> _indices;
        for(int i=0;i<_nchunks;i++) _indices.push_back(i);
        ampiCommStruct worldComm(new_world,empty,_nchunks,_indices);
        CProxy_ampi arr;
#if AMPI_COMLIB
        arr=CProxy_ampi::ckNew(parent,worldComm,ciStreaming,ciBcast,ciAllgather,ciAlltoall,opts);
#else
	//opts.setAnytimeMigration(0);
        arr=CProxy_ampi::ckNew(parent,worldComm,opts);
#endif

	//Broadcast info. to the mpi_worlds array
	// FIXME: remove race condition from MPI_COMM_UNIVERSE broadcast
	ampiCommStruct newComm(new_world,arr,_nchunks);
	if (ampiWorldsGroup.ckGetGroupID().isZero())
		ampiWorldsGroup=CProxy_ampiWorlds::ckNew(newComm);
	else
		ampiWorldsGroup.add(newComm);
	STARTUP_DEBUG("ampiInit> arrays created")

#if AMPI_COMLIB
	StreamingStrategy *sStreaming = new StreamingStrategy(1,10);
	ciStreaming.setStrategy(sStreaming);
	
	BroadcastStrategy *sBcast = new BroadcastStrategy(arr.ckGetArrayID(),USE_HYPERCUBE);
        ciBcast.setStrategy(sBcast);
	
	EachToManyMulticastStrategy *sAllgather = new EachToManyMulticastStrategy(USE_HYPERCUBE,arr.ckGetArrayID(),arr.ckGetArrayID());
        ciAllgather.setStrategy(sAllgather);
	
	EachToManyMulticastStrategy *sAlltoall = new EachToManyMulticastStrategy(USE_PREFIX, arr.ckGetArrayID(),arr.ckGetArrayID());
        ciAlltoall.setStrategy(sAlltoall);
#endif        
  }

  // Find our ampi object:
  ampi *ptr=(ampi *)TCharm::get()->semaGet(AMPI_TCHARM_SEMAID);
  CtvAccess(ampiInitDone)=1;
  CtvAccess(ampiFinalized)=0;
  STARTUP_DEBUG("ampiInit> complete")
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_START(ptr->getThread(), "AMPI_START");
#endif

  init_operations();     // initialize fortran reduction operation table

  return ptr;
}

/// This group is used to broadcast the MPI_COMM_UNIVERSE communicators.
class ampiWorlds : public CBase_ampiWorlds {
public:
    ampiWorlds(const ampiCommStruct &nextWorld) {
        ampiWorldsGroup=thisgroup;
        add(nextWorld);
    }
    ampiWorlds(CkMigrateMessage *m): CBase_ampiWorlds(m) {}
    void pup(PUP::er &p)  { CBase_ampiWorlds::pup(p); }
    void add(const ampiCommStruct &nextWorld) {
        int new_idx=nextWorld.getComm()-(MPI_COMM_WORLD+1);
        mpi_worlds[new_idx].comm=nextWorld;
	if (mpi_nworlds<=new_idx) mpi_nworlds=new_idx+1;
	STARTUP_DEBUG("ampiInit> listed MPI_COMM_UNIVERSE "<<new_idx)
    }
};

//-------------------- ampiParent -------------------------
ampiParent::ampiParent(MPI_Comm worldNo_,CProxy_TCharm threads_)
    		:threads(threads_), worldNo(worldNo_), RProxyCnt(0)
{
  int barrier = 0x1234;
  STARTUP_DEBUG("ampiParent> starting up")
  thread=NULL;
  worldPtr=NULL;
  myDDT=&myDDTsto;
  prepareCtv();

  thread->semaPut(AMPI_BARRIER_SEMAID,&barrier);
	AsyncEvacuate(CmiFalse);
}

ampiParent::ampiParent(CkMigrateMessage *msg):CBase_ampiParent(msg) {
  thread=NULL;
  worldPtr=NULL;
  myDDT=&myDDTsto;
	AsyncEvacuate(CmiFalse);
}

void ampiParent::pup(PUP::er &p) {

  ArrayElement1D::pup(p);
//	printf("[%d] Ampiparent being pupped \n",thisIndex);
  p|threads;

  p|worldStruct;
  myDDT->pup(p);
  p|splitComm;
  p|groupComm;
  p|groups;
  p|ampiReqs;
  p|RProxyCnt;
  p|tmpRProxy;
  p|winStructList;
  p|infos;
}
void ampiParent::prepareCtv(void) {
  thread=threads[thisIndex].ckLocal();
  if (thread==NULL) CkAbort("AMPIParent cannot find its thread!\n");
  CtvAccessOther(thread->getThread(),ampiPtr) = this;
  STARTUP_DEBUG("ampiParent> found TCharm")
}

void ampiParent::ckJustMigrated(void) {
  ArrayElement1D::ckJustMigrated();
  prepareCtv();
}

ampiParent::~ampiParent() {
	STARTUP_DEBUG("ampiParent> destructor called");
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

    //MPI_COMM_SELF has the same member as MPI_COMM_WORLD, but it's alone:
     CkVec<int> _indices;
     _indices.push_back(thisIndex);
     selfStruct = ampiCommStruct(MPI_COMM_SELF,s.getProxy(),1,_indices);
  }
  
  if (!forMigration)
  { //Register the new communicator:
     MPI_Comm comm = s.getComm();
     STARTUP_DEBUG("ampiParent> registering new communicator "<<comm)
     if (comm>=MPI_COMM_WORLD) { 
       // Pass the new ampi to the waiting ampiInit
       thread->semaPut(AMPI_TCHARM_SEMAID, ptr);
     } else if (isSplit(comm)) {
       splitChildRegister(s);
     } else if (isGroup(comm)) {
       groupChildRegister(s);
     } else if (isCart(comm)) {
       cartChildRegister(s);
     } else if (isGraph(comm)) {
       graphChildRegister(s);
     } else if (isInter(comm)) {
       interChildRegister(s);
     } else if (isIntra(comm)) {
       intraChildRegister(s);
     }else
       CkAbort("ampiParent recieved child with bad communicator");
  }

  return thread;
}

// reduction client data - preparation for checkpointing
class ckptClientStruct {
public:
  char *dname;
  ampiParent *ampiPtr;
  ckptClientStruct(char *s, ampiParent *a): dname(s), ampiPtr(a) {}
};

static void checkpointClient(void *param,void *msg)
{
  ckptClientStruct *client = (ckptClientStruct*)param;
  char *dname = client->dname;
  ampiParent *ampiPtr = client->ampiPtr;
  ampiPtr->Checkpoint(strlen(dname), dname);
  delete client;
}

void ampiParent::startCheckpoint(char* dname){
  //if(thisIndex==0) thisProxy[thisIndex].Checkpoint(strlen(dname),dname);
  if (thisIndex==0) {
	ckptClientStruct *clientData = new ckptClientStruct(dname, this);
	CkCallback cb(checkpointClient, clientData);
  	contribute(0, NULL, CkReduction::sum_int, cb);
  }
  else
  	contribute(0, NULL, CkReduction::sum_int);
  thread->stop();
}
void ampiParent::Checkpoint(int len, char* dname){
  if (len == 0) {
    // memory checkpoint
    CkCallback cb(CkIndex_ampiParent::ResumeThread(),thisArrayID);
    CkStartMemCheckpoint(cb);
  }
  else {
    char dirname[256];
    strncpy(dirname,dname,len);
    dirname[len]='\0';
    CkCallback cb(CkIndex_ampiParent::ResumeThread(),thisArrayID);
    CkStartCheckpoint(dirname,cb);
  }
}
void ampiParent::ResumeThread(void){
  thread->resume();
}

int ampiParent::createKeyval(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                             int *keyval, void* extra_state){
	KeyvalNode* newnode = new KeyvalNode(copy_fn, delete_fn, extra_state);
	int idx = kvlist.size();
	kvlist.resize(idx+1);
	kvlist[idx] = newnode;
	*keyval = idx;
	return 0;
}
int ampiParent::freeKeyval(int *keyval){
	if(*keyval <0 || *keyval >= kvlist.size() || !kvlist[*keyval])
		return -1;
	delete kvlist[*keyval];
	kvlist[*keyval] = NULL;
	*keyval = MPI_KEYVAL_INVALID;
	return 0;
}

int ampiParent::putAttr(MPI_Comm comm, int keyval, void* attribute_val){
	if(keyval<0 || keyval >= kvlist.size() || (kvlist[keyval]==NULL))
		return -1;
	ampiCommStruct &cs=*(ampiCommStruct *)&comm2CommStruct(comm);
	// Enlarge the keyval list:
	while (cs.getKeyvals().size()<=keyval) cs.getKeyvals().push_back(0);
	cs.getKeyvals()[keyval]=attribute_val;
	return 0;
}
int ampiParent::kv_is_builtin(int keyval) {
	switch(keyval) {
	case MPI_TAG_UB: kv_builtin_storage=MPI_TAG_UB_VALUE; return 1;
	case MPI_HOST: kv_builtin_storage=MPI_PROC_NULL; return 1;
	case MPI_IO: kv_builtin_storage=0; return 1;
	case MPI_WTIME_IS_GLOBAL: kv_builtin_storage=0; return 1;
	case AMPI_KEYVAL_MYPE: kv_builtin_storage=CkMyPe(); return 1;
	case AMPI_KEYVAL_NUMPES: kv_builtin_storage=CkNumPes(); return 1;
	case AMPI_KEYVAL_MYNODE: kv_builtin_storage=CkMyNode(); return 1;
	case AMPI_KEYVAL_NUMNODES: kv_builtin_storage=CkNumNodes(); return 1;
	default: return 0;
	};
}
int ampiParent::getAttr(MPI_Comm comm, int keyval, void *attribute_val, int *flag){
	*flag = false;
	if (kv_is_builtin(keyval)) { /* Allow access to special builtin flags */
	  *flag=true;
	  *(int *)attribute_val = kv_builtin_storage;
	  return 0;
	}
	if(keyval<0 || keyval >= kvlist.size() || (kvlist[keyval]==NULL))
		return -1; /* invalid keyval */
	
	ampiCommStruct &cs=*(ampiCommStruct *)&comm2CommStruct(comm);
	if (keyval>=cs.getKeyvals().size())  
		return 0; /* we don't have a value yet */
	if (cs.getKeyvals()[keyval]==0)
		return 0; /* we had a value, but now it's zero */
	/* Otherwise, we have a good value */
	*flag = true;
	*(void **)attribute_val = cs.getKeyvals()[keyval];
	return 0;
}
int ampiParent::deleteAttr(MPI_Comm comm, int keyval){
	/* no way to delete an attribute: just overwrite it with 0 */
	return putAttr(comm,keyval,0);
}

//----------------------- ampi -------------------------
void ampi::init(void) {
  parent=NULL;
  thread=NULL;
  msgs=NULL;
  posted_ireqs=NULL;
  resumeOnRecv=false;
  AsyncEvacuate(CmiFalse);
}

ampi::ampi()
{
  /* this constructor only exists so we can create an empty array during split */
  CkAbort("Default ampi constructor should never be called");
}

ampi::ampi(CkArrayID parent_,const ampiCommStruct &s)
   :parentProxy(parent_)
{
  init();

  myComm=s; myComm.setArrayID(thisArrayID);
  myRank=myComm.getRankForIndex(thisIndex);

  findParent(false);

  msgs = CmmNew();
  posted_ireqs = CmmNew();
  nbcasts = 0;

  comlibProxy = thisProxy;
  ComlibDelegateProxy(&comlibProxy);
  
  seqEntries=parent->numElements;
  oorder.init (seqEntries);
}

ampi::ampi(CkArrayID parent_,const ampiCommStruct &s, ComlibInstanceHandle ciStreaming_,
    		ComlibInstanceHandle ciBcast_,ComlibInstanceHandle ciAllgather_,ComlibInstanceHandle ciAlltoall_)
   :parentProxy(parent_),ciStreaming(ciStreaming_),ciBcast(ciBcast_),ciAllgather(ciAllgather_),ciAlltoall(ciAlltoall_)
{
  init();

  myComm=s; myComm.setArrayID(thisArrayID);
  myRank=myComm.getRankForIndex(thisIndex);

  findParent(false);

  msgs = CmmNew();
  posted_ireqs = CmmNew();
  nbcasts = 0;

  comlibProxy = thisProxy;
  ComlibDelegateProxy(&comlibProxy);
  ciStreaming.setSourcePe();
  ciBcast.setSourcePe();
  ciAllgather.setSourcePe();
  ciAlltoall.setSourcePe();

  seqEntries=parent->numElements;
  oorder.init (seqEntries);
}

ampi::ampi(CkMigrateMessage *msg):CBase_ampi(msg)
{
  init();
  
  seqEntries=-1;
}

void ampi::ckJustMigrated(void)
{
	findParent(true);
	ArrayElement1D::ckJustMigrated();
}

void ampi::findParent(bool forMigration) {
        STARTUP_DEBUG("ampi> finding my parent")
	parent=parentProxy[thisIndex].ckLocal();
	if (parent==NULL) CkAbort("AMPI can't find its parent!");
	thread=parent->registerAmpi(this,myComm,forMigration);
	if (thread==NULL) CkAbort("AMPI can't find its thread!");
//	printf("[%d] ampi index %d TCharm thread pointer %p \n",CkMyPe(),thisIndex,thread);
}

static void cmm_pup_ampi_message(pup_er p,void **msg) {
	CkPupMessage(*(PUP::er *)p,msg,1);
	if (pup_isDeleting(p)) delete (AmpiMsg *)*msg;
//	printf("[%d] pupping ampi message %p \n",CkMyPe(),*msg);
}

void ampi::pup(PUP::er &p)
{
  if(!p.isUserlevel())
    ArrayElement1D::pup(p);//Pack superclass
  p|parentProxy;
  p|myComm;
  p|myRank;
  p|nbcasts;
  p|tmpVec;
  p|remoteProxy;
  p|resumeOnRecv;
  p|comlibProxy;
  p|ciStreaming;
  p|ciBcast;
  p|ciAllgather;
  p|ciAlltoall;

  if(p.isUnpacking()){
    ciStreaming.setSourcePe();
    ciBcast.setSourcePe();
    ciAllgather.setSourcePe();
    ciAlltoall.setSourcePe();
  }

  msgs=CmmPup((pup_er)&p,msgs,cmm_pup_ampi_message);
  posted_ireqs = CmmNew();		// FIXME 
//	printf("[%d] ampi index %d msgs table pointer %p\n",CkMyPe(),thisIndex,msgs);

  p|seqEntries;
  p|oorder;
}

ampi::~ampi()
{
  if (CkInRestarting()) {
    // in restarting, we need to flush messages
  int tags[3], sts[3];
  tags[0] = tags[1] = tags[2] = CmmWildCard;
  AmpiMsg *msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
  while (msg) {
    delete msg;
    msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
  }
  }
  CmmFree(msgs);
  CmmFree(posted_ireqs);         // FIXME
}

//------------------------ Communicator Splitting ---------------------
class ampiSplitKey {
public:
	int nextSplitComm;
	int color; //New class of processes we'll belong to
	int key; //To determine rank in new ordering
	int rank; //Rank in old ordering
	ampiSplitKey() {}
	ampiSplitKey(int nextSplitComm_,int color_,int key_,int rank_)
		:nextSplitComm(nextSplitComm_), color(color_), key(key_), rank(rank_) {}
};

/* "type" may indicate whether call is for a cartesian topology etc. */

void ampi::split(int color,int key,MPI_Comm *dest, int type)
{
  if (type == CART_TOPOL) {
	ampiSplitKey splitKey(parent->getNextCart(),color,key,myRank);
	int rootIdx=myComm.getIndexForRank(0);
	CkCallback cb(CkIndex_ampi::splitPhase1(0),CkArrayIndex1D(rootIdx),myComm.getProxy());
	contribute(sizeof(splitKey),&splitKey,CkReduction::concat,cb);

	thread->suspend(); //Resumed by ampiParent::cartChildRegister
	MPI_Comm newComm=parent->getNextCart()-1;
	*dest=newComm;
  } else {
	ampiSplitKey splitKey(parent->getNextSplit(),color,key,myRank);
	int rootIdx=myComm.getIndexForRank(0);
	CkCallback cb(CkIndex_ampi::splitPhase1(0),CkArrayIndex1D(rootIdx),myComm.getProxy());
	contribute(sizeof(splitKey),&splitKey,CkReduction::concat,cb);

	thread->suspend(); //Resumed by ampiParent::splitChildRegister
	MPI_Comm newComm=parent->getNextSplit()-1;
	*dest=newComm;
  }
}

CDECL int compareAmpiSplitKey(const void *a_, const void *b_) {
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
	if (nKeys!=myComm.getSize()) CkAbort("ampi::splitReduce expected a split contribution from every rank!");
	qsort(keys,nKeys,sizeof(ampiSplitKey),compareAmpiSplitKey);

	MPI_Comm newComm = -1;
	for(int i=0;i<nKeys;i++)
		if(keys[i].nextSplitComm>newComm)
			newComm = keys[i].nextSplitComm;

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

			CkVec<int> indices; //Maps rank to array indices for new arrau
			for (int i=c;i<nKeys;i++) {
				if (keys[i].color!=lastColor) break; //Done with this color
				int idx=myComm.getIndexForRank(keys[i].rank);
				indices.push_back(idx);
			}

			//FIXME: create a new communicator for each color, instead of
			// (confusingly) re-using the same MPI_Comm number for each.
			lastComm=ampiCommStruct(newComm,lastAmpi,indices.size(),indices);
		}
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
	if (splitComm.size()<=idx) splitComm.resize(idx+1);
	splitComm[idx]=new ampiCommStruct(s);
	thread->resume(); //Matches suspend at end of ampi::split
}

//-----------------create communicator from group--------------
// The procedure is like that of comm_split very much,
// so the code is shamelessly copied from above
//   1. reduction to make sure all members have called
//   2. the root in the old communicator create the new array
//   3. ampiParent::register is called to register new array as new comm
class vecStruct {
public:
  int nextgroup;
  groupStruct vec;
  vecStruct():nextgroup(-1){}
  vecStruct(int nextgroup_, groupStruct vec_)
    : nextgroup(nextgroup_), vec(vec_) { }
};

void ampi::commCreate(const groupStruct vec,MPI_Comm* newcomm){
  int rootIdx=vec[0];
  tmpVec = vec;
  CkCallback cb(CkIndex_ampi::commCreatePhase1(NULL),CkArrayIndex1D(rootIdx),myComm.getProxy());
  MPI_Comm nextgroup = parent->getNextGroup();
  contribute(sizeof(nextgroup), &nextgroup,CkReduction::max_int,cb);

  if(getPosOp(thisIndex,vec)>=0){
    thread->suspend(); //Resumed by ampiParent::groupChildRegister
    MPI_Comm retcomm = parent->getNextGroup()-1;
    *newcomm = retcomm;
  }else{
    *newcomm = MPI_COMM_NULL;
  }
}

void ampi::commCreatePhase1(CkReductionMsg *msg){
  MPI_Comm *nextGroupComm = (int *)msg->getData();

  CkArrayOptions opts;
  opts.bindTo(parentProxy);
  opts.setNumInitial(0);
  CkArrayID unusedAID;
  ampiCommStruct unusedComm;
  CProxy_ampi newAmpi=CProxy_ampi::ckNew(unusedAID,unusedComm,opts);
  newAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution

  groupStruct indices = tmpVec;
  ampiCommStruct newCommstruct = ampiCommStruct(*nextGroupComm,newAmpi,indices.size(),indices);
  for(int i=0;i<indices.size();i++){
    int newIdx=indices[i];
    newAmpi[newIdx].insert(parentProxy,newCommstruct);
  }
  delete msg;
}

void ampiParent::groupChildRegister(const ampiCommStruct &s) {
  int idx=s.getComm()-MPI_COMM_FIRST_GROUP;
  if (groupComm.size()<=idx) groupComm.resize(idx+1);
  groupComm[idx]=new ampiCommStruct(s);
  thread->resume(); //Matches suspend at end of ampi::split
}

/* Virtual topology communicator creation */
void ampi::cartCreate(const groupStruct vec,MPI_Comm* newcomm){
  int rootIdx=vec[0];
  tmpVec = vec;
  CkCallback cb(CkIndex_ampi::cartCreatePhase1(NULL),CkArrayIndex1D(rootIdx),myComm.getProxy());

  MPI_Comm nextcart = parent->getNextCart();
  contribute(sizeof(nextcart), &nextcart,CkReduction::max_int,cb);

  if(getPosOp(thisIndex,vec)>=0){
    thread->suspend(); //Resumed by ampiParent::cartChildRegister
     MPI_Comm retcomm = parent->getNextCart()-1;
     *newcomm = retcomm;
  }else
    *newcomm = MPI_COMM_NULL;
}

void ampi::cartCreatePhase1(CkReductionMsg *msg){
  MPI_Comm *nextCartComm = (int *)msg->getData();

  CkArrayOptions opts;
  opts.bindTo(parentProxy);
  opts.setNumInitial(0);
  CkArrayID unusedAID;
  ampiCommStruct unusedComm;
  CProxy_ampi newAmpi=CProxy_ampi::ckNew(unusedAID,unusedComm,opts);
  newAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution

  groupStruct indices = tmpVec;
  ampiCommStruct newCommstruct = ampiCommStruct(*nextCartComm,newAmpi,indices.
						size(),indices);
  for(int i=0;i<indices.size();i++){
    int newIdx=indices[i];
    newAmpi[newIdx].insert(parentProxy,newCommstruct);
  }
  delete msg;
}

void ampiParent::cartChildRegister(const ampiCommStruct &s) {
  int idx=s.getComm()-MPI_COMM_FIRST_CART;
  if (cartComm.size()<=idx) {
    cartComm.resize(idx+1);
    cartComm.length()=idx+1;
  }
  cartComm[idx]=new ampiCommStruct(s);
  thread->resume(); //Matches suspend at end of ampi::cartCreate
}

void ampi::graphCreate(const groupStruct vec,MPI_Comm* newcomm){
  int rootIdx=vec[0];
  tmpVec = vec;
  CkCallback cb(CkIndex_ampi::graphCreatePhase1(NULL),CkArrayIndex1D(rootIdx),
		myComm.getProxy());
  MPI_Comm nextgraph = parent->getNextGraph();
  contribute(sizeof(nextgraph), &nextgraph,CkReduction::max_int,cb);

  if(getPosOp(thisIndex,vec)>=0){
    thread->suspend(); //Resumed by ampiParent::graphChildRegister
    MPI_Comm retcomm = parent->getNextGraph()-1;
    *newcomm = retcomm;
  }else
    *newcomm = MPI_COMM_NULL;
}

void ampi::graphCreatePhase1(CkReductionMsg *msg){
  MPI_Comm *nextGraphComm = (int *)msg->getData();

  CkArrayOptions opts;
  opts.bindTo(parentProxy);
  opts.setNumInitial(0);
  CkArrayID unusedAID;
  ampiCommStruct unusedComm;
  CProxy_ampi newAmpi=CProxy_ampi::ckNew(unusedAID,unusedComm,opts);
  newAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution

  groupStruct indices = tmpVec;
  ampiCommStruct newCommstruct = ampiCommStruct(*nextGraphComm,newAmpi,indices
						.size(),indices);
  for(int i=0;i<indices.size();i++){
    int newIdx=indices[i];
    newAmpi[newIdx].insert(parentProxy,newCommstruct);
  }
  delete msg;
}

void ampiParent::graphChildRegister(const ampiCommStruct &s) {
  int idx=s.getComm()-MPI_COMM_FIRST_GRAPH;
  if (graphComm.size()<=idx) {
    graphComm.resize(idx+1);
    graphComm.length()=idx+1;
  }
  graphComm[idx]=new ampiCommStruct(s);
  thread->resume(); //Matches suspend at end of ampi::graphCreate
}

void ampi::intercommCreate(const groupStruct rvec, const int root, MPI_Comm *ncomm){
  if(thisIndex==root) { // not everybody gets the valid rvec
    tmpVec = rvec;
  }
  CkCallback cb(CkIndex_ampi::intercommCreatePhase1(NULL),CkArrayIndex1D(root),myComm.getProxy());
  MPI_Comm nextinter = parent->getNextInter();
  contribute(sizeof(nextinter), &nextinter,CkReduction::max_int,cb);

  thread->suspend(); //Resumed by ampiParent::interChildRegister
  MPI_Comm newcomm=parent->getNextInter()-1;
  *ncomm=newcomm;
}

void ampi::intercommCreatePhase1(CkReductionMsg *msg){
  MPI_Comm *nextInterComm = (int *)msg->getData();

  groupStruct lgroup = myComm.getIndices();
  CkArrayOptions opts;
  opts.bindTo(parentProxy);
  opts.setNumInitial(0);
  CkArrayID unusedAID;
  ampiCommStruct unusedComm;
  CProxy_ampi newAmpi=CProxy_ampi::ckNew(unusedAID,unusedComm,opts);
  newAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution

  ampiCommStruct newCommstruct = ampiCommStruct(*nextInterComm,newAmpi,lgroup.size(),lgroup,tmpVec);
  for(int i=0;i<lgroup.size();i++){
    int newIdx=lgroup[i];
    newAmpi[newIdx].insert(parentProxy,newCommstruct);
  }

  parentProxy[0].ExchangeProxy(newAmpi);
  delete msg;
}

void ampiParent::interChildRegister(const ampiCommStruct &s) {
  int idx=s.getComm()-MPI_COMM_FIRST_INTER;
  if (interComm.size()<=idx) interComm.resize(idx+1);
  interComm[idx]=new ampiCommStruct(s);
  //thread->resume(); // don't resume it yet, till parent set remote proxy
}

void ampi::intercommMerge(int first, MPI_Comm *ncomm){ // first valid only at local root
  if(myRank == 0 && first == 1){ // first (lower) group creates the intracommunicator for the higher group
    groupStruct lvec = myComm.getIndices();
    groupStruct rvec = myComm.getRemoteIndices();
    int rsize = rvec.size();
    tmpVec = lvec;
    for(int i=0;i<rsize;i++)
      tmpVec.push_back(rvec[i]);
    if(tmpVec.size()==0) CkAbort("Error in ampi::intercommMerge: merging empty comms!\n");
  }else{
    tmpVec.resize(0);
  }

  int rootIdx=myComm.getIndexForRank(0);
  CkCallback cb(CkIndex_ampi::intercommMergePhase1(NULL),CkArrayIndex1D(rootIdx),myComm.getProxy());
  MPI_Comm nextintra = parent->getNextIntra();
  contribute(sizeof(nextintra), &nextintra,CkReduction::max_int,cb);

  thread->suspend(); //Resumed by ampiParent::interChildRegister
  MPI_Comm newcomm=parent->getNextIntra()-1;
  *ncomm=newcomm;
}

void ampi::intercommMergePhase1(CkReductionMsg *msg){  // gets called on two roots, first root creates the comm
  if(tmpVec.size()==0) { delete msg; return; }
  MPI_Comm *nextIntraComm = (int *)msg->getData();
  CkArrayOptions opts;
  opts.bindTo(parentProxy);
  opts.setNumInitial(0);
  CkArrayID unusedAID;
  ampiCommStruct unusedComm;
  CProxy_ampi newAmpi=CProxy_ampi::ckNew(unusedAID,unusedComm,opts);
  newAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution

  ampiCommStruct newCommstruct = ampiCommStruct(*nextIntraComm,newAmpi,tmpVec.size(),tmpVec);
  for(int i=0;i<tmpVec.size();i++){
    int newIdx=tmpVec[i];
    newAmpi[newIdx].insert(parentProxy,newCommstruct);
  }
  delete msg;
}

void ampiParent::intraChildRegister(const ampiCommStruct &s) {
  int idx=s.getComm()-MPI_COMM_FIRST_INTRA;
  if (intraComm.size()<=idx) intraComm.resize(idx+1);
  intraComm[idx]=new ampiCommStruct(s);
  thread->resume(); //Matches suspend at end of ampi::split
}

//------------------------ communication -----------------------
const ampiCommStruct &universeComm2CommStruct(MPI_Comm universeNo)
{
  if (universeNo>MPI_COMM_WORLD) {
    int worldDex=universeNo-MPI_COMM_WORLD-1;
    if (worldDex>=mpi_nworlds)
      CkAbort("Bad world communicator passed to universeComm2CommStruct");
    return mpi_worlds[worldDex].comm;
  }
  CkAbort("Bad communicator passed to universeComm2CommStruct");
  return mpi_worlds[0].comm; // meaningless return
}

void ampi::block(void){
	thread->suspend();
}

void ampi::yield(void){
	thread->schedule();
}

void ampi::unblock(void){
	thread->resume();
}

void
ampi::generic(AmpiMsg* msg)
{
MSG_ORDER_DEBUG(
  CkPrintf("AMPI vp %d arrival: tag=%d, src=%d, comm=%d  (from %d, seq %d) resumeOnRecv %d\n",
  	thisIndex,msg->tag,msg->srcRank,msg->comm, msg->srcIdx, msg->seq,resumeOnRecv);
)

//	AmpiMsg *msgcopy = msg;
  if(msg->seq != -1) {
    int srcIdx=msg->srcIdx;
    int n=oorder.put(srcIdx,msg);
    if (n>0) { // This message was in-order
      inorder(msg);
      if (n>1) { // It enables other, previously out-of-order messages
        while((msg=oorder.getOutOfOrder(srcIdx))!=0) {
          inorder(msg);
        }
      }
    }
  } else { //Cross-world or system messages are unordered
    inorder(msg);
  }
  
  if(resumeOnRecv){
    thread->resume();
  }
}

void
ampi::inorder(AmpiMsg* msg)
{
MSG_ORDER_DEBUG(
  CkPrintf("AMPI vp %d inorder: tag=%d, src=%d, comm=%d  (from %d, seq %d)\n",
  	thisIndex,msg->tag,msg->srcRank,msg->comm, msg->srcIdx, msg->seq);
)
  // check posted recvs
  int tags[3], sts[3];
  tags[0] = msg->tag; tags[1] = msg->srcRank; tags[2] = msg->comm;
  IReq *ireq = (IReq *)CmmGet(posted_ireqs, 3, tags, sts);
  if (ireq) {	// receive posted
    ireq->receive(this, msg);
    ireq->tag = sts[0];
    ireq->src = sts[1];
    ireq->comm = sts[2];
  }
  else
    CmmPut(msgs, 3, tags, msg);
}

AmpiMsg *ampi::getMessage(int t, int s, int comm, int *sts)
{
    int tags[3];
    tags[0] = t; tags[1] = s; tags[2] = comm;
    AmpiMsg *msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
    return msg;
}

AmpiMsg *ampi::makeAmpiMsg(int destIdx,
	int t,int sRank,const void *buf,int count,int type,MPI_Comm destcomm)
{
  CkDDT_DataType *ddt = getDDT()->getType(type);
  int len = ddt->getSize(count);
  int sIdx=thisIndex;
  int seq = -1;
  if (destIdx>=0 && destcomm<=MPI_COMM_WORLD && t<=MPI_TAG_UB_VALUE) //Not cross-module: set seqno
     seq = oorder.nextOutgoing(destIdx);
  AmpiMsg *msg = new (len, 0) AmpiMsg(seq, t, sIdx, sRank, len, destcomm);
  ddt->serialize((char*)buf, (char*)msg->data, count, 1);
  return msg;
}

void
ampi::comlibsend(int t, int sRank, const void* buf, int count, int type,  int rank, MPI_Comm destcomm)
{
  delesend(t,sRank,buf,count,type,rank,destcomm,comlibProxy);
}

void
ampi::send(int t, int sRank, const void* buf, int count, int type,  int rank, MPI_Comm destcomm)
{
  const ampiCommStruct &dest=comm2CommStruct(destcomm);
  delesend(t,sRank,buf,count,type,rank,destcomm,dest.getProxy());
}

void
ampi::sendraw(int t, int sRank, void* buf, int len, CkArrayID aid, int idx)
{
  AmpiMsg *msg = new (len, 0) AmpiMsg(-1, t, -1, sRank, len, MPI_COMM_WORLD);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void
ampi::delesend(int t, int sRank, const void* buf, int count, int type,  int rank, MPI_Comm destcomm, CProxy_ampi arrproxy)
{
  if(rank==MPI_PROC_NULL) return;
  const ampiCommStruct &dest=comm2CommStruct(destcomm);
  int destIdx = dest.getIndexForRank(rank);
  if(isInter()){
    sRank = parent->thisIndex;
    destcomm = MPI_COMM_FIRST_INTER;
    destIdx = dest.getIndexForRemoteRank(rank);
    arrproxy = remoteProxy;
  }
 MSG_ORDER_DEBUG(
  CkPrintf("AMPI vp %d send: tag=%d, src=%d, comm=%d (to %d)\n",thisIndex,t,sRank,destcomm,destIdx);
 )

  arrproxy[destIdx].generic(makeAmpiMsg(destIdx,t,sRank,buf,count,type,destcomm));

#ifndef CMK_OPTIMIZE
  int size=0;
  MPI_Type_size(type,&size);
  _LOG_E_AMPI_MSG_SEND(t,destIdx,count,size)
#endif
}

int
ampi::processMessage(AmpiMsg *msg, int t, int s, void* buf, int count, int type)
{
  CkDDT_DataType *ddt = getDDT()->getType(type);
  int len = ddt->getSize(count);
  
  if (msg->length > len && msg->length-len!=sizeof(AmpiOpHeader))
  { /* Received more data than we were expecting */
    char einfo[1024];
    sprintf(einfo, "FATAL ERROR in rank %d MPI_Recv (tag=%d, source=%d)\n"
    	"  Expecting only %d bytes (%d items of type %d), \n"
	"  but received %d bytes from rank %d\nAMPI> MPI_Send was longer than matching MPI_Recv.",
            thisIndex,t,s,
	    len, count, type,
	    msg->length, msg->srcRank);
    CkAbort(einfo);
    return -1;
  }else if(msg->length < len){ // only at rare case shall we reset count by using divide
    count = msg->length/(ddt->getSize(1));
  }
  if (msg->length-len==sizeof(AmpiOpHeader)) {	// reduction msg
    ddt->serialize((char*)buf, (char*)msg->data+sizeof(AmpiOpHeader), count, (-1));
  } else {
    ddt->serialize((char*)buf, (char*)msg->data, count, (-1));
  }
  return 0;
}

int
ampi::recv(int t, int s, void* buf, int count, int type, int comm, int *sts)
{
  MPI_Comm disComm = myComm.getComm();
  if(s==MPI_PROC_NULL) {
    ((MPI_Status *)sts)->MPI_SOURCE = MPI_PROC_NULL;
    ((MPI_Status *)sts)->MPI_TAG = MPI_ANY_TAG;
    ((MPI_Status *)sts)->MPI_LENGTH = 0;
    return 0;
  }
  _LOG_E_END_AMPI_PROCESSING(thisIndex)
#if CMK_BLUEGENE_CHARM
  void *curLog;		// store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
  TRACE_BG_AMPI_SUSPEND();
#endif

  if(isInter()){
    s = myComm.getIndexForRemoteRank(s);
    comm = MPI_COMM_FIRST_INTER;
  }

  int tags[3];
  AmpiMsg *msg = 0;
  
 MSG_ORDER_DEBUG(
  CkPrintf("AMPI vp %d blocking recv: tag=%d, src=%d, comm=%d\n",thisIndex,t,s,comm);
 )

  resumeOnRecv=true;
  ampi *dis = getAmpiInstance(disComm);
  while(1) {
      //This is done to take into account the case in which an ampi 
      // thread has migrated while waiting for a message
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmGet(dis->msgs, 3, tags, sts);
    if (msg) break;
    dis->thread->suspend();
    dis = getAmpiInstance(disComm);
  }
  dis->resumeOnRecv=false;

  if(sts)
    ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
  int status = dis->processMessage(msg, t, s, buf, count, type);
  if (status != 0) return status;

  _LOG_E_BEGIN_AMPI_PROCESSING(thisIndex,s,count)
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_RESUME(thread->getThread(), msg, "RECV_RESUME", curLog);
#endif

  delete msg;
  return 0;
}

void
ampi::probe(int t, int s, int comm, int *sts)
{
  int tags[3];
#if CMK_BLUEGENE_CHARM
  void *curLog;		// store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
  TRACE_BG_AMPI_SUSPEND();
#endif

  AmpiMsg *msg = 0;
  resumeOnRecv=true;
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, sts);
    if (msg) break;
    thread->suspend();
  }
  resumeOnRecv=false;
  if(sts)
    ((MPI_Status*)sts)->MPI_LENGTH = msg->length;
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_RESUME(thread->getThread(), msg, "PROBE_RESUME", curLog);
#endif
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


const int MPI_BCAST_COMM=MPI_COMM_WORLD+1000;
void
ampi::bcast(int root, void* buf, int count, int type,MPI_Comm destcomm)
{
  const ampiCommStruct &dest=comm2CommStruct(destcomm);
  int rootIdx=dest.getIndexForRank(root);
  if(rootIdx==thisIndex) {
#if 0//AMPI_COMLIB
    ciBcast.beginIteration();
    comlibProxy.generic(makeAmpiMsg(-1,MPI_BCAST_TAG,0, buf,count,type, MPI_BCAST_COMM));
#else
    thisProxy.generic(makeAmpiMsg(-1,MPI_BCAST_TAG,0, buf,count,type, MPI_BCAST_COMM));
#endif
  }
  if(-1==recv(MPI_BCAST_TAG,0, buf,count,type, MPI_BCAST_COMM)) CkAbort("AMPI> Error in broadcast");
  nbcasts++;
}

void
ampi::bcastraw(void* buf, int len, CkArrayID aid)
{
  AmpiMsg *msg = new (len, 0) AmpiMsg(-1, MPI_BCAST_TAG, -1, 0, len, 0);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa.generic(msg);
}


AmpiMsg* 
ampi::Alltoall_RemoteIGet(int disp, int cnt, MPI_Datatype type, int tag)
{
  CkAssert(tag==MPI_ATA_TAG && AlltoallGetFlag);
  int unit;
  CkDDT_DataType *ddt = getDDT()->getType(type);
  unit = ddt->getSize(1);
  int totalsize = unit*cnt;

  AmpiMsg *msg = new (totalsize, 0) AmpiMsg(-1, -1, -1, thisIndex,totalsize,myComm.getComm());
  char* addr = (char*)Alltoallbuff+disp*unit;
  ddt->serialize((char*)msg->data, addr, cnt, (-1));
  return msg;
}

int MPI_null_copy_fn (MPI_Comm comm, int keyval, void *extra_state,
			void *attr_in, void *attr_out, int *flag){
  (*flag) = 0;
  return (MPI_SUCCESS);
}
int MPI_dup_fn(MPI_Comm comm, int keyval, void *extra_state,
			void *attr_in, void *attr_out, int *flag){
  (*(void **)attr_out) = attr_in;
  (*flag) = 1;
  return (MPI_SUCCESS);
}
int MPI_null_delete_fn (MPI_Comm comm, int keyval, void *attr, void *extra_state ){
  return (MPI_SUCCESS);
}


void AmpiSeqQ::init(int numP) 
{
  elements.init(numP);
}

AmpiSeqQ::~AmpiSeqQ () {
}
  
void AmpiSeqQ::pup(PUP::er &p) {
  p|out;
  p|elements;
}

void AmpiSeqQ::putOutOfOrder(int srcIdx, AmpiMsg *msg)
{
  AmpiOtherElement &el=elements[srcIdx];
#ifndef CMK_OPTIMIZE
  if (msg->seq<el.seqIncoming)
    CkAbort("AMPI Logic error: received late out-of-order message!\n");
#endif
  out.enq(msg);
  el.nOut++; // We have another message in the out-of-order queue
}

AmpiMsg *AmpiSeqQ::getOutOfOrder(int srcIdx)
{
  AmpiOtherElement &el=elements[srcIdx];
  if (el.nOut==0) return 0; // No more out-of-order left.
  // Walk through our out-of-order queue, searching for our next message:
  for (int i=0;i<out.length();i++) {
    AmpiMsg *msg=out.deq();
    if (msg->srcIdx==srcIdx && msg->seq==el.seqIncoming) {
      el.seqIncoming++;
      el.nOut--; // We have one less message out-of-order
      return msg;
    }
    else
      out.enq(msg);
  }
  // We walked the whole queue-- ours is not there.
  return 0;
}


void AmpiRequestList::pup(PUP::er &p) { 
	CmiMemoryCheck();
	if(!CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC)){
		return;
	}

	p(blklen); //Allocated size of block
	p(len); //Number of used elements in block
	if(p.isUnpacking()){
		makeBlock(blklen,len);
	}
	int count=0;
	for(int i=0;i<len;i++){
		char nonnull;
		if(!p.isUnpacking()){
			if(block[i] == NULL){
				nonnull = 0;
			}else{
				nonnull = block[i]->getType();
			}
		}	
		p(nonnull);
		if(nonnull != 0){
			if(p.isUnpacking()){
				switch(nonnull){
					case 1:
						block[i] = new PersReq;
						break;
					case 2:	
						block[i] = new IReq;
						break;
					case 3:	
						block[i] = new ATAReq;
						break;
				}
			}	
			block[i]->pup(p);
			count++;
		}else{
			block[i] = 0;
		}
	}
	if(p.isDeleting()){
		freeBlock();
	}
	CmiMemoryCheck();
}

//------------------ External Interface -----------------
ampiParent *getAmpiParent(void) {
  ampiParent *p = CtvAccess(ampiPtr);
#ifndef CMK_OPTIMIZE
  if (p==NULL) CkAbort("Cannot call MPI routines before AMPI is initialized.\n");
#endif
  return p;
}

ampi *getAmpiInstance(MPI_Comm comm) {
  ampi *ptr=getAmpiParent()->comm2ampi(comm);
#ifndef CMK_OPTIMIZE
  if (ptr==NULL) CkAbort("AMPI's getAmpiInstance> null pointer\n");
#endif
  return ptr;
}

inline static AmpiRequestList *getReqs(void) {
  return &(getAmpiParent()->ampiReqs);
}

inline void checkComm(MPI_Comm comm){
#ifndef CMK_OPTIMIZE
  getAmpiParent()->checkComm(comm);
#endif
}

inline void checkRequest(MPI_Request req){
#ifndef CMK_OPTIMIZE
  getReqs()->checkRequest(req);
#endif
}

inline void checkRequests(int n, MPI_Request* reqs){
#ifndef CMK_OPTIMIZE
  AmpiRequestList* reqlist = getReqs();
  for(int i=0;i<n;i++)
    reqlist->checkRequest(reqs[i]);
#endif
}

CDECL void AMPI_Migrate(void)
{
  AMPIAPI("AMPI_Migrate");
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_SUSPEND();
#endif
  TCHARM_Migrate();
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_START(getAmpiInstance(MPI_COMM_WORLD)->getThread(), "AMPI_MIGRATE")
#endif
}


CDECL void AMPI_Evacuate(void)
{
  TCHARM_Evacuate();
}



CDECL void AMPI_Migrateto(int destPE)
{
  AMPIAPI("AMPI_MigrateTo");
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_SUSPEND();
#endif
  TCHARM_Migrate_to(destPE);
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_START(getAmpiInstance(MPI_COMM_WORLD)->getThread(), "AMPI_MIGRATETO")
#endif
}

CDECL void AMPI_MigrateTo(int destPE)
{
	AMPI_Migrateto(destPE);
}

CDECL void AMPI_Async_Migrate(void)
{
  AMPIAPI("AMPI_Async_Migrate");
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_SUSPEND();
#endif
  TCHARM_Async_Migrate();
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_START(getAmpiInstance(MPI_COMM_WORLD)->getThread(), "AMPI_MIGRATE")
#endif
}

CDECL void AMPI_Allow_Migrate(void)
{
  AMPIAPI("AMPI_Allow_Migrate");
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_SUSPEND();
#endif
  TCHARM_Allow_Migrate();
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_START(getAmpiInstance(MPI_COMM_WORLD)->getThread(), "AMPI_ALLOW_MIGRATE")
#endif
}

CDECL void AMPI_Setmigratable(MPI_Comm comm, int mig){
#if CMK_LBDB_ON
  //AMPIAPI("AMPI_Setmigratable");
  ampi *ptr=getAmpiInstance(comm);
  ptr->setMigratable(mig);
#else
  CkPrintf("Warning: MPI_Setmigratable and load balancing are not supported in this version.\n");
#endif
}

CDECL int AMPI_Init(int *p_argc, char*** p_argv)
{
  if (nodeinit_has_been_called) {
    AMPIAPI("AMPI_Init");
    char **argv;
    if (p_argv) argv=*p_argv;
    else argv=CkGetArgv();
    ampiInit(argv);
    if (p_argc) *p_argc=CmiGetArgc(argv);
  }
  else
  { /* Charm hasn't been started yet! */
    CkAbort("AMPI_Init> Charm is not initialized!");
  }
  return 0;
}

CDECL int AMPI_Initialized(int *isInit)
{
  if (nodeinit_has_been_called) {
        AMPIAPI("AMPI_Initialized");     /* in case charm init not called */
  	*isInit=CtvAccess(ampiInitDone);
  }
  else /* !nodeinit_has_been_called */ {
  	*isInit=nodeinit_has_been_called;
  }
  return 0;
}

CDECL int AMPI_Finalized(int *isFinalized)
{
    AMPIAPI("AMPI_Initialized");     /* in case charm init not called */
    *isFinalized=CtvAccess(ampiFinalized);
    return 0;
}

CDECL int AMPI_Comm_rank(MPI_Comm comm, int *rank)
{
  //AMPIAPI("AMPI_Comm_rank");
  *rank = getAmpiInstance(comm)->getRank(comm);
  return 0;
}

CDECL
int AMPI_Comm_size(MPI_Comm comm, int *size)
{
  //AMPIAPI("AMPI_Comm_size");
  *size = getAmpiInstance(comm)->getSize(comm);
  return 0;
}

CDECL
int AMPI_Comm_compare(MPI_Comm comm1,MPI_Comm comm2, int *result)
{
  AMPIAPI("AMPI_Comm_compare");
  if(comm1==comm2) *result=MPI_IDENT;
  else{
    int equal=1;
    CkVec<int> ind1, ind2;
    ind1 = getAmpiInstance(comm1)->getIndices();
    ind2 = getAmpiInstance(comm2)->getIndices();
    if(ind1.size()==ind2.size()){
      for(int i=0;i<ind1.size();i++)
        if(ind1[i] != ind2[i]) { equal=0; break; }
    }
    if(equal==1) *result=MPI_CONGRUENT;
    else *result=MPI_UNEQUAL;
  }
  return 0;
}

CDECL void AMPI_Exit(int /*exitCode*/)
{
  AMPIAPI("AMPI_Exit");
  TCHARM_Done();
}
FDECL void FTN_NAME(MPI_EXIT,mpi_exit)(int *exitCode)
{
  AMPI_Exit(*exitCode);
}

CDECL
int AMPI_Finalize(void)
{
  AMPIAPI("AMPI_Finalize");
#if PRINT_IDLE
  CkPrintf("[%d] Idle time %fs.\n", CkMyPe(), totalidle);
#endif
#if AMPI_COUNTER
    getAmpiParent()->counters.output(getAmpiInstance(MPI_COMM_WORLD)->getRank(MPI_COMM_WORLD));
#endif
  CtvAccess(ampiFinalized)=1;
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_SUSPEND();
#endif
//  getAmpiInstance(MPI_COMM_WORLD)->outputCounter();
  AMPI_Exit(0);
  return 0;
}

CDECL
int AMPI_Send(void *msg, int count, MPI_Datatype type, int dest,
                        int tag, MPI_Comm comm)
{
  AMPIAPI("AMPI_Send");
  ampi *ptr = getAmpiInstance(comm);
#if AMPI_COMLIB
  if(enableStreaming){  
    ptr->getStreaming().beginIteration();
    ptr->comlibsend(tag,ptr->getRank(comm),msg,count,type,dest,comm);
  } else
#endif
    ptr->send(tag, ptr->getRank(comm), msg, count, type, dest, comm);
#if AMPI_COUNTER
  getAmpiParent()->counters.send++;
#endif
  return 0;
}

CDECL
int AMPI_Recv(void *msg, int count, MPI_Datatype type, int src, int tag,
              MPI_Comm comm, MPI_Status *status)
{
	AMPIAPI("AMPI_Recv");
	ampi *ptr = getAmpiInstance(comm);

	if(-1==ptr->recv(tag,src,msg,count,type, comm, (int*) status)) CkAbort("AMPI> Error in MPI_Recv");

#if AMPI_COUNTER
  getAmpiParent()->counters.recv++;
#endif
	return 0;
}

CDECL
int AMPI_Probe(int src, int tag, MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("AMPI_Probe");
  ampi *ptr = getAmpiInstance(comm);
  ptr->probe(tag,src, comm, (int*) status);
  return 0;
}

CDECL
int AMPI_Iprobe(int src,int tag,MPI_Comm comm,int *flag,MPI_Status *status)
{
  AMPIAPI("AMPI_Iprobe");
  ampi *ptr = getAmpiInstance(comm);
  *flag = ptr->iprobe(tag,src,comm,(int*) status);
  return 0;
}

CDECL
int AMPI_Sendrecv(void *sbuf, int scount, int stype, int dest,
                  int stag, void *rbuf, int rcount, int rtype,
                  int src, int rtag, MPI_Comm comm, MPI_Status *sts)
{
  AMPIAPI("AMPI_Sendrecv");
  int se=MPI_Send(sbuf,scount,stype,dest,stag,comm);
  int re=MPI_Recv(rbuf,rcount,rtype,src,rtag,comm,sts);
  if (se) return se;
  else return re;
}

CDECL
int AMPI_Sendrecv_replace(void* buf, int count, MPI_Datatype datatype,
                         int dest, int sendtag, int source, int recvtag,
                         MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("AMPI_Sendrecv_replace");
  return AMPI_Sendrecv(buf, count, datatype, dest, sendtag,
                      buf, count, datatype, source, recvtag, comm, status);
}


CDECL
int AMPI_Barrier(MPI_Comm comm)
{
  AMPIAPI("AMPI_Barrier");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Barrier not allowed for Inter-communicator!");
#if CMK_BLUEGENE_CHARM
  void *barrierLog;		// store current log in timeline
  TRACE_BG_AMPI_BARRIER_START(barrierLog);
#endif
  //HACK: Use collective operation as a barrier.
  AMPI_Allreduce(NULL,NULL,0,MPI_INT,MPI_SUM,comm);
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_BARRIER_END(barrierLog);
#endif
#if AMPI_COUNTER
  getAmpiParent()->counters.barrier++;
#endif
  return 0;
}

CDECL
int AMPI_Bcast(void *buf, int count, MPI_Datatype type, int root,
                         MPI_Comm comm)
{
  AMPIAPI("AMPI_Bcast");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Bcast not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return 0;
  ampi *ptr = getAmpiInstance(comm);
  ptr->bcast(root, buf, count, type,comm);
#if AMPI_COUNTER
  getAmpiParent()->counters.bcast++;
#endif
  return 0;
}

/// This routine is called with the results of a Reduce or AllReduce
const int MPI_REDUCE_SOURCE=0;
const int MPI_REDUCE_COMM=MPI_COMM_WORLD;
void ampi::reduceResult(CkReductionMsg *msg)
{
	MSG_ORDER_DEBUG(printf("[%d] reduceResult called \n",thisIndex));
  ampi::sendraw(MPI_REDUCE_TAG, MPI_REDUCE_SOURCE, msg->getData(), msg->getSize(),
             thisArrayID,thisIndex);
  delete msg;
}

static CkReductionMsg *makeRednMsg(CkDDT_DataType *ddt,const void *inbuf,int count,int type,MPI_Op op)
{
  int szdata = ddt->getSize(count);
  int szhdr = sizeof(AmpiOpHeader);
  AmpiOpHeader newhdr(op,type,count,szdata); 
  CkReductionMsg *msg=CkReductionMsg::buildNew(szdata+szhdr,NULL,AmpiReducer);
  memcpy(msg->getData(),&newhdr,szhdr);
  ddt->serialize((char*)inbuf, (char*)msg->getData()+szhdr, count, 1);
  return msg;
}

// Copy the MPI datatype "type" from inbuf to outbuf
static int copyDatatype(MPI_Comm comm,MPI_Datatype type,int count,const void *inbuf,void *outbuf) {
  // ddts don't have "copy", so fake it by serializing into a temp buffer, then
  //  deserializing into the output.
  ampi *ptr = getAmpiInstance(comm);
  CkDDT_DataType *ddt=ptr->getDDT()->getType(type);
  int len=ddt->getSize(count);
  char *serialized=new char[len];
  ddt->serialize((char*)inbuf,(char*)serialized,count,1);
  ddt->serialize((char*)outbuf,(char*)serialized,count,-1); 
  delete [] serialized;		// < memory leak!  // gzheng 
  
  return MPI_SUCCESS;
}

CDECL
int AMPI_Reduce(void *inbuf, void *outbuf, int count, int type, MPI_Op op,
               int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Reduce");
  if(op == MPI_OP_NULL) CkAbort("MPI_Reduce called with MPI_OP_NULL!!!");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Reduce not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,type,count,inbuf,outbuf);
  ampi *ptr = getAmpiInstance(comm);
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,op);

  int rootIdx=ptr->comm2CommStruct(comm).getIndexForRank(root);
  CkCallback reduceCB(CkIndex_ampi::reduceResult(0),CkArrayIndex1D(rootIdx),ptr->getProxy(),true);
  msg->setCallback(reduceCB);
	MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Reduce called on comm %d root %d \n",ptr->thisIndex,comm,rootIdx));
  ptr->contribute(msg);
  if (ptr->thisIndex == rootIdx){
    /*HACK: Use recv() to block until reduction data comes back*/
    if(-1==ptr->recv(MPI_REDUCE_TAG, MPI_REDUCE_SOURCE, outbuf, count, type, MPI_REDUCE_COMM))
      CkAbort("AMPI>MPI_Reduce called with different values on different processors!");
  }
#if AMPI_COUNTER
  getAmpiParent()->counters.reduce++;
#endif
  return 0;
}

CDECL
int AMPI_Allreduce(void *inbuf, void *outbuf, int count, int type,
                  MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("AMPI_Allreduce");
  if(op == MPI_OP_NULL) CkAbort("MPI_Allreduce called with MPI_OP_NULL!!!");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Allreduce not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,type,count,inbuf,outbuf);
  ampi *ptr = getAmpiInstance(comm);
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,op);
  CkCallback allreduceCB(CkIndex_ampi::reduceResult(0),ptr->getProxy());
  msg->setCallback(allreduceCB);
  ptr->contribute(msg);

  /*HACK: Use recv() to block until the reduction data comes back*/
  if(-1==ptr->recv(MPI_REDUCE_TAG, MPI_REDUCE_SOURCE, outbuf, count, type, MPI_REDUCE_COMM))
    CkAbort("AMPI> MPI_Allreduce called with different values on different processors!");
#if AMPI_COUNTER
  getAmpiParent()->counters.allreduce++;
#endif
  return 0;
}

CDECL
int AMPI_Iallreduce(void *inbuf, void *outbuf, int count, int type,
                   MPI_Op op, MPI_Comm comm, MPI_Request* request)
{
  AMPIAPI("AMPI_Allreduce");
  checkRequest(*request);
  if(op == MPI_OP_NULL) CkAbort("MPI_Iallreduce called with MPI_OP_NULL!!!");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Iallreduce not allowed for Inter-communicator!");
  ampi *ptr = getAmpiInstance(comm);
  if(comm==MPI_COMM_SELF) {
    // Just do a local copy: no need to do a reduction:
    return copyDatatype(comm,type,count,inbuf,outbuf);
  }
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,op);
  CkCallback allreduceCB(CkIndex_ampi::reduceResult(0),ptr->getProxy());
  msg->setCallback(allreduceCB);
  ptr->contribute(msg);

  // using irecv instead recv to non-block the call and get request pointer
  AmpiRequestList* reqs = getReqs();
  IReq *newreq = new IReq(outbuf,count,type,MPI_REDUCE_SOURCE,MPI_REDUCE_TAG,MPI_REDUCE_COMM);
  *request = reqs->insert(newreq);
  return 0;
}

CDECL
int AMPI_Reduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("AMPI_Reduce_scatter");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Reduce_scatter not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,datatype,recvcounts[0],sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int count=0;
  int *displs = new int [size];
  int len;
  void *tmpbuf;

  //under construction
  for(int i=0;i<size;i++){
    displs[i] = count;
    count+= recvcounts[i];
  }
  len = ptr->getDDT()->getType(datatype)->getSize(count);
  tmpbuf = malloc(len);
  AMPI_Reduce(sendbuf, tmpbuf, count, datatype, op, 0, comm);
  AMPI_Scatterv(tmpbuf, recvcounts, displs, datatype,
               recvbuf, recvcounts[ptr->getRank(comm)], datatype, 0, comm);
  free(tmpbuf);
  delete [] displs;	// < memory leak ! // gzheng
  return 0;
}

/***** MPI_Scan algorithm (from MPICH) *******
   recvbuf = sendbuf;
   partial_scan = sendbuf;
   mask = 0x1;
   while (mask < size) {
      dst = rank^mask;
      if (dst < size) {
         send partial_scan to dst;
         recv from dst into tmp_buf;
         if (rank > dst) {
            partial_scan = tmp_buf + partial_scan;
            recvbuf = tmp_buf + recvbuf;
         }
         else {
            if (op is commutative)
               partial_scan = tmp_buf + partial_scan;
            else {
               tmp_buf = partial_scan + tmp_buf;
               partial_scan = tmp_buf;
            }
         }
      }
      mask <<= 1;
   }
 ***** MPI_Scan algorithm (from MPICH) *******/

void applyOp(MPI_Datatype datatype, MPI_Op op, int count, void* invec, void* inoutvec) { // inoutvec[i] = invec[i] op inoutvec[i]
  (op)(invec,inoutvec,&count,&datatype);
}
CDECL
int AMPI_Scan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm ){
  AMPIAPI("AMPI_Scan");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Scan not allowed for Inter-communicator!");
  MPI_Status sts;
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int blklen = ptr->getDDT()->getType(datatype)->getSize(count);
  int rank = ptr->getRank(comm);
  int mask = 0x1;
  int dst;
  void* tmp_buf = malloc(blklen);
  void* partial_scan = malloc(blklen);
  
  memcpy(recvbuf, sendbuf, blklen);
  memcpy(partial_scan, sendbuf, blklen);
  while(mask < size){
    dst = rank^mask;
    if(dst < size){
      AMPI_Sendrecv(partial_scan,count,datatype,dst,MPI_SCAN_TAG,
		   tmp_buf,count,datatype,dst,MPI_SCAN_TAG,comm,&sts);
      if(rank > dst){
        (op)(tmp_buf,partial_scan,&count,&datatype);
	(op)(tmp_buf,recvbuf,&count,&datatype);
      }else {
        (op)(partial_scan,tmp_buf,&count,&datatype);
        memcpy(partial_scan,tmp_buf,blklen);
      }
      mask <<= 1;
    }
  }

  free(tmp_buf);
  free(partial_scan);
#if AMPI_COUNTER
  getAmpiParent()->counters.scan++;
#endif
  return 0;
}

CDECL
int AMPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op){
  //AMPIAPI("AMPI_Op_create");
  *op = function;
  return 0;
}

CDECL
int AMPI_Op_free(MPI_Op *op){
  //AMPIAPI("AMPI_Op_free");
  *op = MPI_OP_NULL;
  return 0;
}


CDECL
double AMPI_Wtime(void)
{
//  AMPIAPI("AMPI_Wtime");
#if CMK_BLUEGENE_CHARM
  return BgGetTime();
#else
  return TCHARM_Wall_timer();
#endif
}

CDECL
double AMPI_Wtick(void){
  //AMPIAPI("AMPI_Wtick");
  return 1e-6;
}

int PersReq::start(){
  if(sndrcv == 1) { // send request
    ampi *aptr=getAmpiInstance(comm);
    aptr->send(tag, aptr->getRank(comm), buf, count, type, src, comm);
  }
  return 0;
}

CDECL
int AMPI_Start(MPI_Request *request)
{
  AMPIAPI("AMPI_Start");
  checkRequest(*request);
  AmpiRequestList *reqs = getReqs();
  if(-1==(*reqs)[*request]->start()){
    CkAbort("MPI_Start could be used only on persistent communication requests!");
  }
  return 0;
}

CDECL
int AMPI_Startall(int count, MPI_Request *requests){
  AMPIAPI("AMPI_Startall");
  checkRequests(count,requests);
  AmpiRequestList *reqs = getReqs();
  for(int i=0;i<count;i++){
    if(-1==(*reqs)[requests[i]]->start())
      CkAbort("MPI_Start could be used only on persistent communication requests!");
  }
  return 0;
}

/* organize the indices of requests into a vector of a vector: 
 * level 1 is different msg envelope matches
 * level 2 is (posting) ordered requests of with envelope
 * each time multiple completion call loop over first elem of level 1
 * and move the matched to the NULL request slot.   
 * warning: this does not work with I-Alltoall requests */
inline int areInactiveReqs(int count, MPI_Request* reqs){ // if count==0 then all inactive
  for(int i=0;i<count;i++){
    if(reqs[i]!=MPI_REQUEST_NULL)
      return 0;
  }
  return 1;
}
inline int matchReq(MPI_Request ia, MPI_Request ib){
  checkRequest(ia);  
  checkRequest(ib);
  AmpiRequestList* reqs = getReqs();
  AmpiRequest *a, *b;
  if(ia==MPI_REQUEST_NULL && ib==MPI_REQUEST_NULL) return 1;
  if(ia==MPI_REQUEST_NULL || ib==MPI_REQUEST_NULL) return 0;
  a=(*reqs)[ia];  b=(*reqs)[ib];
  if(a->tag != b->tag) return 0;
  if(a->src != b->src) return 0;
  if(a->comm != b->comm) return 0;
  return 1;
}
inline void swapInt(int& a,int& b){
  int tmp;
  tmp=a; a=b; b=tmp;
}
inline void sortedIndex(int n, int* arr, int* idx){
  int i,j;
  for(i=0;i<n;i++) 
    idx[i]=i;
  for (i=0; i<n-1; i++) 
    for (j=0; j<n-1-i; j++)
      if (arr[idx[j+1]] < arr[idx[j]]) 
	swapInt(idx[j+1],idx[j]);
}
CkVec<CkVec<int> > *vecIndex(int count, int* arr){
  CkAssert(count!=0);
  int *newidx = new int [count];
  int flag;
  sortedIndex(count,arr,newidx);
  CkVec<CkVec<int> > *vec = new CkVec<CkVec<int> >;
  CkVec<int> slot;
  vec->push_back(slot);
  (*vec)[0].push_back(newidx[0]);
  for(int i=1;i<count;i++){
    flag=0;
    for(int j=0;j<vec->size();j++){
      if(matchReq(arr[newidx[i]],arr[((*vec)[j])[0]])){
        ((*vec)[j]).push_back(newidx[i]);
	flag++;
      }
    }
    if(!flag){
      CkVec<int> newslot;
      newslot.push_back(newidx[i]);
      vec->push_back(newslot);
    }else{
      CkAssert(flag==1);
    }
  }
  delete [] newidx;
  return vec;
}
void vecPrint(CkVec<CkVec<int> > vec, int* arr){
  printf("vec content: ");
  for(int i=0;i<vec.size();i++){
    printf("{");
    for(int j=0;j<(vec[i]).size();j++){
      printf(" %d ",arr[(vec[i])[j]]);
    }
    printf("} ");
  }
  printf("\n");
}

int PersReq::wait(MPI_Status *sts){
	if(sndrcv == 2) {
		if(-1==getAmpiInstance(comm)->recv(tag, src, buf, count, type, comm, (int*)sts))
			CkAbort("AMPI> Error in persistent request wait");
#if CMK_BLUEGENE_CHARM
  		_TRACE_BG_TLINE_END(&event);
#endif
	}
	return 0;
}

int IReq::wait(MPI_Status *sts){
  	ampi *aptr = getAmpiInstance(comm);
	while (status == false) {
  		aptr->resumeOnRecv=true;
		aptr->block();
	}
	aptr->resumeOnRecv=false;
        if(sts) {
          sts->MPI_TAG = tag;
          sts->MPI_SOURCE = src;
          sts->MPI_COMM = comm;
          sts->MPI_LENGTH = length;
        }
#if 0
	if(-1==getAmpiInstance(comm)->recv(tag, src, buf, count, type, comm, (int*)sts))
		CkAbort("AMPI> Error in non-blocking request wait");
#endif
#if CMK_BLUEGENE_CHARM
  	_TRACE_BG_TLINE_END(&event);
#endif
	return 0;
}

int ATAReq::wait(MPI_Status *sts){
	int i;
	for(i=0;i<count;i++){
		if(-1==getAmpiInstance(myreqs[i].comm)->recv(myreqs[i].tag, myreqs[i].src, myreqs[i].buf,
				myreqs[i].count, myreqs[i].type, myreqs[i].comm, (int *)sts))
			CkAbort("AMPI> Error in alltoall request wait");
#if CMK_BLUEGENE_CHARM
  		_TRACE_BG_TLINE_END(&myreqs[i].event);
#endif
	}
#if CMK_BLUEGENE_CHARM
  	TRACE_BG_AMPI_NEWSTART(getAmpiInstance(MPI_COMM_WORLD)->getThread(), "ATAReq", myreqs[i].event, count);
  	_TRACE_BG_TLINE_END(&event);
#endif
	return 0;
}

CDECL
int AMPI_Wait(MPI_Request *request, MPI_Status *sts)
{
  AMPIAPI("AMPI_Wait");
  if(*request == MPI_REQUEST_NULL){
    stsempty(*sts);
    return 0;
  }
  checkRequest(*request);
  AmpiRequestList* reqs = getReqs();
  AMPI_DEBUG("MPI_Wait: request=%d, reqs.size=%d, &reqs=%d\n",*request,reqs->size(),reqs);
  (*reqs)[*request]->wait(sts);
  if((*reqs)[*request]->getType() != 1) { // only free non-blocking request
    reqs->free(*request);
    *request = MPI_REQUEST_NULL;
  }
  return 0;
}

CDECL
int AMPI_Waitall(int count, MPI_Request request[], MPI_Status sts[])
{
  AMPIAPI("AMPI_Waitall");
  if(count==0) return MPI_SUCCESS;
  checkRequests(count,request);
  int i,j,oldPe;
  AmpiRequestList* reqs = getReqs();
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  for(i=0;i<reqvec->size();i++){
    for(j=0;j<((*reqvec)[i]).size();j++){
      if(request[((*reqvec)[i])[j]] == MPI_REQUEST_NULL){
        stsempty(sts[((*reqvec)[i])[j]]);
        continue;
      }
      oldPe = CkMyPe();
      AmpiRequest *waitReq = ((*reqs)[request[((*reqvec)[i])[j]]]);
      waitReq->wait(&sts[((*reqvec)[i])[j]]);
			if(oldPe != CkMyPe()){
#if 1
			reqs = getReqs();
			reqvec  = vecIndex(count,request);
#endif
			}
    }
  }
#if CMK_BLUEGENE_CHARM
  TRACE_BG_AMPI_WAITALL(reqs);   // setup forward and backward dependence
#endif
  // free memory of requests
  for(i=0;i<count;i++){ 
    if(request[i] == MPI_REQUEST_NULL)
      continue;
    if((*reqs)[request[i]]->getType() != 1) { // only free non-blocking request
      reqs->free(request[i]);
      request[i] = MPI_REQUEST_NULL;
    }
  }
  delete reqvec;
  return 0;
}

CDECL
int AMPI_Waitany(int count, MPI_Request *request, int *idx, MPI_Status *sts)
{
  AMPIAPI("AMPI_Waitany");
  
  USER_CALL_DEBUG("AMPI_Waitany("<<count<<")");
  checkRequests(count,request);
  if(areInactiveReqs(count,request)){
    *idx=MPI_UNDEFINED;
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  int flag=0;
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  while(count>0){ /* keep looping until some request finishes: */
    for(int i=0;i<reqvec->size();i++){
      AMPI_Test(&request[((*reqvec)[i])[0]], &flag, sts);
      if(flag == 1 && sts->MPI_COMM != 0){ // to skip MPI_REQUEST_NULL
        *idx = ((*reqvec)[i])[0];
  	USER_CALL_DEBUG("AMPI_Waitany returning "<<*idx);
        return 0;
      }
    }
    /* no requests have finished yet-- schedule and try again */
    AMPI_Yield(MPI_COMM_WORLD);
  }
  *idx = MPI_UNDEFINED;
  USER_CALL_DEBUG("AMPI_Waitany returning UNDEFINED");
	delete reqvec;
  return 0;
}

CDECL
int AMPI_Waitsome(int incount, MPI_Request *array_of_requests, int *outcount,
                 int *array_of_indices, MPI_Status *array_of_statuses)
{
  AMPIAPI("AMPI_Waitsome");
  checkRequests(incount,array_of_requests);
  if(areInactiveReqs(incount,array_of_requests)){
    *outcount=MPI_UNDEFINED;
    return MPI_SUCCESS;
  }
  MPI_Status sts;
  int i;
  int flag=0, realflag=0;
  CkVec<CkVec<int> > *reqvec = vecIndex(incount,array_of_requests);
  *outcount = 0;
  while(1){
    for(i=0;i<reqvec->size();i++){
      AMPI_Test(&array_of_requests[((*reqvec)[i])[0]], &flag, &sts);
      if(flag == 1){ 
        array_of_indices[(*outcount)]=((*reqvec)[i])[0];
	array_of_statuses[(*outcount)++]=sts;
        if(sts.MPI_COMM != 0)
	  realflag=1; // there is real(non null) request
      }
    }
    if(realflag && outcount>0) break;
  }
	delete reqvec;
  return 0;
}

CmiBool PersReq::test(MPI_Status *sts){
	if(sndrcv == 2) 	// recv request
		return getAmpiInstance(comm)->iprobe(tag, src, comm, (int*)sts);
	else			// send request
		return 1;

}
void PersReq::complete(MPI_Status *sts){
	if(-1==getAmpiInstance(comm)->recv(tag, src, buf, count, type, comm, (int*)sts))
		CkAbort("AMPI> Error in persistent request complete");
}

CmiBool IReq::test(MPI_Status *sts){
        if (status == true) {           
	  if(sts)
            sts->MPI_LENGTH = length;           
	  return true;
        }
        else {
          getAmpiInstance(comm)->yield();
          return false;
        }
/*
	return getAmpiInstance(comm)->iprobe(tag, src, comm, (int*)sts);
*/
}
void IReq::complete(MPI_Status *sts){
	wait(sts);
/*
	if(-1==getAmpiInstance(comm)->recv(tag, src, buf, count, type, comm, (int*)sts))
		CkAbort("AMPI> Error in non-blocking request complete");
*/
}

void IReq::receive(ampi *aptr, AmpiMsg *msg)
{
    int sts = aptr->processMessage(msg, tag, src, buf, count, type);
    status = (sts == 0);
    length = msg->length;
    delete msg;
}

CmiBool ATAReq::test(MPI_Status *sts){
	int i, flag=1;
	for(i=0;i<count;i++){
		flag *= getAmpiInstance(myreqs[i].comm)->iprobe(myreqs[i].tag, myreqs[i].src,
					myreqs[i].comm, (int*) sts);
	}
	return flag;
}
void ATAReq::complete(MPI_Status *sts){
	int i;
	for(i=0;i<count;i++){
		if(-1==getAmpiInstance(myreqs[i].comm)->recv(myreqs[i].tag, myreqs[i].src, myreqs[i].buf,
						myreqs[i].count, myreqs[i].type, myreqs[i].comm, (int*)sts))
			CkAbort("AMPI> Error in alltoall request complete");
	}
}

CDECL
int AMPI_Test(MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("AMPI_Test");
  checkRequest(*request);
  if(*request==MPI_REQUEST_NULL) {
    *flag = 1;
    stsempty(*sts);
    return 0;
  }
  AmpiRequestList* reqs = getReqs();
  if(1 == (*flag = (*reqs)[*request]->test(sts))){
    if((*reqs)[*request]->getType() != 1) { // only free non-blocking request
      (*reqs)[*request]->complete(sts);
      reqs->free(*request);
      *request = MPI_REQUEST_NULL;
    }
  }
  return 0;
}

CDECL
int AMPI_Testany(int count, MPI_Request *request, int *index, int *flag, MPI_Status *sts){
  AMPIAPI("AMPI_Testany");
  checkRequests(count,request);
  if(areInactiveReqs(count,request)){
    *flag=1;
    *index=MPI_UNDEFINED;
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  *flag=0;
  for(int i=0;i<reqvec->size();i++){
    AMPI_Test(&request[((*reqvec)[i])[0]], flag, sts);
    if(*flag==1 && sts->MPI_COMM!=0){ // skip MPI_REQUEST_NULL
      *index = ((*reqvec)[i])[0];
      return 0;
    }
  }
  *index = MPI_UNDEFINED;
	delete reqvec;
  return 0;
}

CDECL
int AMPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("AMPI_Testall");
  if(count==0) return MPI_SUCCESS;
  checkRequests(count,request);
  int tmpflag;
  int i,j;
  AmpiRequestList* reqs = getReqs();
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  *flag = 1;  
  for(i=0;i<reqvec->size();i++){
    for(j=0;j<((*reqvec)[i]).size();j++){
      if(request[((*reqvec)[i])[j]] == MPI_REQUEST_NULL)
        continue;
      tmpflag = (*reqs)[request[((*reqvec)[i])[j]]]->test(&sts[((*reqvec)[i])[j]]);
      *flag *= tmpflag;
    }
  }
  if(flag) 
    MPI_Waitall(count,request,sts);
	delete reqvec;	
  return 0;
}

CDECL
int AMPI_Testsome(int incount, MPI_Request *array_of_requests, int *outcount,
                 int *array_of_indices, MPI_Status *array_of_statuses)
{
  AMPIAPI("AMPI_Testsome");
  checkRequests(incount,array_of_requests);
  if(areInactiveReqs(incount,array_of_requests)){
    *outcount=MPI_UNDEFINED;
    return MPI_SUCCESS;
  }
  MPI_Status sts;
  int flag;
  int i;
  CkVec<CkVec<int> > *reqvec = vecIndex(incount,array_of_requests);
  *outcount = 0;
  for(i=0;i<reqvec->size();i++){
    AMPI_Test(&array_of_requests[((*reqvec)[i])[0]], &flag, &sts);
    if(flag == 1){
      array_of_indices[(*outcount)]=((*reqvec)[i])[0];
      array_of_statuses[(*outcount)++]=sts;
    }
  }
	delete reqvec;
  return 0;
}

CDECL
int AMPI_Request_free(MPI_Request *request){
  AMPIAPI("AMPI_Request_free");
  checkRequest(*request);
  if(*request==MPI_REQUEST_NULL) return 0;
  AmpiRequestList* reqs = getReqs();
  reqs->free(*request);
  return 0;
}

CDECL
int AMPI_Cancel(MPI_Request *request){
  AMPIAPI("AMPI_Request_free");
  return AMPI_Request_free(request);
}

CDECL
int AMPI_Recv_init(void *buf, int count, int type, int src, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("AMPI_Recv_init");
  AmpiRequestList* reqs = getReqs();
  PersReq *newreq = new PersReq(buf,count,type,src,tag,comm,2);
  *req = reqs->insert(newreq);
  return 0;
}

CDECL
int AMPI_Send_init(void *buf, int count, int type, int dest, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("AMPI_Send_init");
  AmpiRequestList* reqs = getReqs();
  PersReq *newreq = new PersReq(buf,count,type,dest,tag,comm,1);
  *req = reqs->insert(newreq);
  return 0;
}

static CkDDT *getDDT(void) {
  return getAmpiParent()->myDDT;
}

CDECL
int AMPI_Type_contiguous(int count, MPI_Datatype oldtype,
                         MPI_Datatype *newtype)
{
  AMPIAPI("AMPI_Type_contiguous");
  getDDT()->newContiguous(count, oldtype, newtype);
  return 0;
}

CDECL
int AMPI_Type_vector(int count, int blocklength, int stride,
                     MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_vector");
  getDDT()->newVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

CDECL
int AMPI_Type_hvector(int count, int blocklength, MPI_Aint stride, 
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_hvector");
  getDDT()->newHVector(count, blocklength, stride, oldtype, newtype);
  return 0 ;
}

CDECL
int AMPI_Type_indexed(int count, int* arrBlength, int* arrDisp, 
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_indexed");
  getDDT()->newIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

CDECL
int AMPI_Type_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp,
                       MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_hindexed");
  getDDT()->newHIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

CDECL
int AMPI_Type_struct(int count, int* arrBlength, int* arrDisp, 
                     MPI_Datatype* oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_struct");
  getDDT()->newStruct(count, arrBlength, arrDisp, oldtype, newtype);
  return 0 ;
}

CDECL
int AMPI_Type_commit(MPI_Datatype *datatype)
{
  AMPIAPI("AMPI_Type_commit");
  return 0;
}

CDECL
int AMPI_Type_free(MPI_Datatype *datatype)
{
  AMPIAPI("AMPI_Type_free");
  getDDT()->freeType(datatype);
  return 0;
}


CDECL
int AMPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent)
{
  AMPIAPI("AMPI_Type_extent");
  *extent = getDDT()->getExtent(datatype);
  return 0;
}

CDECL
int AMPI_Type_size(MPI_Datatype datatype, int *size)
{
  AMPIAPI("AMPI_Type_size");
  *size=getDDT()->getSize(datatype);
  return 0;
}

CDECL
int AMPI_Isend(void *buf, int count, MPI_Datatype type, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Isend");
  USER_CALL_DEBUG("AMPI_Isend("<<type<<","<<dest<<","<<tag<<","<<comm<<")");
  ampi *ptr = getAmpiInstance(comm);
#if AMPI_COMLIB
  if(enableStreaming){
    ptr->getStreaming().beginIteration();
    ptr->comlibsend(tag,ptr->getRank(comm),buf,count,type,dest,comm);
  } else
#endif
  ptr->send(tag, ptr->getRank(comm), buf, count, type, dest, comm);
  *request = MPI_REQUEST_NULL;
#if AMPI_COUNTER
  getAmpiParent()->counters.isend++;
#endif
  return 0;
}

CDECL
int AMPI_Irecv(void *buf, int count, MPI_Datatype type, int src,
              int tag, MPI_Comm comm, MPI_Request *request)
{
  if(src==MPI_PROC_NULL) { *request = MPI_REQUEST_NULL; return 0; }
  AMPIAPI("AMPI_Irecv");
  USER_CALL_DEBUG("AMPI_Irecv("<<type<<","<<src<<","<<tag<<","<<comm<<")");
  AmpiRequestList* reqs = getReqs();
  IReq *newreq = new IReq(buf,count,type,src,tag,comm);
  *request = reqs->insert(newreq);
    // if msg is here, shall we do receive right away ??
    // posted irecv
  // message already arraved?
  ampi *aptr = getAmpiInstance(comm);
  AmpiMsg *msg = aptr->getMessage(tag, src, comm, &newreq->tag);
  if (msg) {
    newreq->receive(aptr, msg);
  }
  else {
      // post receive
    int tags[3];
    tags[0] = tag; tags[1] = src; tags[2] = comm;
    CmmPut(aptr->posted_ireqs, 3, tags, newreq);
  }
  
#if AMPI_COUNTER
  getAmpiParent()->counters.irecv++;
#endif
  return 0;
}

CDECL
int AMPI_Ireduce(void *sendbuf, void *recvbuf, int count, int type, MPI_Op op,
                 int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ireduce");
  if(op == MPI_OP_NULL) CkAbort("MPI_Ireduce called with MPI_OP_NULL!!!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,type,count,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),sendbuf,count,type,op);
  int rootIdx=ptr->comm2CommStruct(comm).getIndexForRank(root);
  CkCallback reduceCB(CkIndex_ampi::reduceResult(0),CkArrayIndex1D(rootIdx),ptr->getProxy(),true);
  msg->setCallback(reduceCB);
  ptr->contribute(msg);

  if (ptr->thisIndex == rootIdx){
    // using irecv instead recv to non-block the call and get request pointer
    AmpiRequestList* reqs = getReqs();
    IReq *newreq = new IReq(recvbuf,count,type,0,MPI_REDUCE_TAG,MPI_REDUCE_COMM);
    *request = reqs->insert(newreq);
  }
  return 0;
}

CDECL
int AMPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
  AMPIAPI("AMPI_Allgather");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Allgather not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;
#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAllgather().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_GATHER_TAG, ptr->getRank(comm), sendbuf, sendcount,
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAllgather().endIteration();
  } else 
#endif
      for(i=0;i<size;i++) {
          ptr->send(MPI_GATHER_TAG, ptr->getRank(comm), sendbuf, sendcount,
                    sendtype, i, comm);
      }

  MPI_Status status;
  CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
  int itemsize = dttype->getSize(recvcount) ;

  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
#if AMPI_COUNTER
  getAmpiParent()->counters.allgather++;
#endif
  return 0;
}

CDECL
int AMPI_Iallgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request* request)
{
  AMPIAPI("AMPI_Iallgather");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Iallgather not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;
#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAllgather().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_GATHER_TAG, ptr->getRank(comm), sendbuf, sendcount,
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAllgather().endIteration();
  } else 
#endif
      for(i=0;i<size;i++) {
          ptr->send(MPI_GATHER_TAG, ptr->getRank(comm), sendbuf, sendcount,
                    sendtype, i, comm);
      }

  CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
  int itemsize = dttype->getSize(recvcount) ;

  // copy+paste from MPI_Irecv
  AmpiRequestList* reqs = getReqs();
  ATAReq *newreq = new ATAReq(size);
  for(i=0;i<size;i++){
    if(newreq->addReq(((char*)recvbuf)+(itemsize*i),recvcount,recvtype,i,MPI_GATHER_TAG,comm)!=(i+1))
      CkAbort("MPI_Iallgather: Error adding requests into ATAReq!");
  }
  *request = reqs->insert(newreq);
  AMPI_DEBUG("MPI_Iallgather: request=%d, reqs.size=%d, &reqs=%d\n",*request,reqs->size(),reqs);

  return 0;
}

CDECL
int AMPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int *recvcounts, int *displs,
                   MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Allgatherv");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Allgatherv not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;
#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAllgather().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_GATHER_TAG, ptr->getRank(comm), sendbuf, sendcount,
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAllgather().endIteration();
  } else
#endif 
      for(i=0;i<size;i++) {
          ptr->send(MPI_GATHER_TAG, ptr->getRank(comm), sendbuf, sendcount,
                    sendtype, i, comm);
      }

  MPI_Status status;
  CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
  int itemsize = dttype->getSize() ;

  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
#if AMPI_COUNTER
  getAmpiParent()->counters.allgather++;
#endif
  return 0;
}

CDECL
int AMPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Gather");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Gather not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;
  AMPI_Send(sendbuf, sendcount, sendtype, root, MPI_GATHER_TAG, comm);

  if(ptr->getRank(comm)==root) {
    MPI_Status status;
    CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
    int itemsize = dttype->getSize(recvcount) ;

    for(i=0;i<size;i++) {
      AMPI_Recv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
               i, MPI_GATHER_TAG, comm, &status);
    }
  }
#if AMPI_COUNTER
  getAmpiParent()->counters.gather++;
#endif
  return 0;
}

CDECL
int AMPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int *recvcounts, int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Gatherv");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Gatherv not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;

  AMPI_Send(sendbuf, sendcount, sendtype, root, MPI_GATHER_TAG, comm);

  if(ptr->getRank(comm) == root) {
    MPI_Status status;
    CkDDT_DataType* dttype = ptr->getDDT()->getType(recvtype) ;
    int itemsize = dttype->getSize();

    for(i=0;i<size;i++) {
      AMPI_Recv(((char*)recvbuf)+(itemsize*displs[i]), recvcounts[i], recvtype,
               i, MPI_GATHER_TAG, comm, &status);
    }
  }
#if AMPI_COUNTER
  getAmpiParent()->counters.gather++;
#endif
  return 0;
}

CDECL
int AMPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Scatter");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Scatter not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;

  if(ptr->getRank(comm)==root) {
      CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
      int itemsize = dttype->getSize(sendcount) ;
      for(i=0;i<size;i++) {
          ptr->send(MPI_SCATTER_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i),
                    sendcount, sendtype, i, comm);
      }
      //ptr->mcomlibEnd();
  }
  
  MPI_Status status;
  AMPI_Recv(recvbuf, recvcount, recvtype, root, MPI_SCATTER_TAG, comm, &status);

#if AMPI_COUNTER
  getAmpiParent()->counters.scatter++;
#endif
  return 0;
}

CDECL
int AMPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Scatterv");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Scatterv not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcounts[0],sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int i;

  if(ptr->getRank(comm) == root) {
      CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
      int itemsize = dttype->getSize() ;
      for(i=0;i<size;i++) {
          ptr->send(MPI_SCATTER_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*displs[i]),
                    sendcounts[i], sendtype, i, comm);
      }
      //ptr->mcomlibEnd();
  }
  
  MPI_Status status;
  AMPI_Recv(recvbuf, recvcount, recvtype, root, MPI_SCATTER_TAG, comm, &status);
  
#if AMPI_COUNTER
  getAmpiParent()->counters.scatter++;
#endif
  return 0;
}

CDECL
int AMPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoall");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Alltoall not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  CkDDT_DataType *dttype;
  int itemsize;
  int i;

#if 0
    // use irecv to post receives
  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  MPI_Request *reqs = new MPI_Request[size];
  for(i=0;i<size;i++) {
        AMPI_Irecv(((char*)recvbuf)+(itemsize*i), recvcount, recvtype,
              i, MPI_ATA_TAG, comm, &reqs[i]);
  }
  //AMPI_Yield(comm);
  //AMPI_Barrier(comm);

  dttype = ptr->getDDT()->getType(sendtype) ;
  itemsize = dttype->getSize(sendcount) ;
#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAlltoall().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount,
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAlltoall().endIteration();
  } else
#endif 
  {
      int r = rand();
      if (ptr->getRank(comm) %2 ==0)
      for(i=0;i<size;i++) {
          ptr->send(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount, sendtype, i, comm);
      }
      else
      for(i=size-1;i>=0;i--) {
          ptr->send(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount, sendtype, i, comm);
      }
  }
  
    // can use waitall if it is fixed for memory
  MPI_Status status;
  for (i=0;i<size;i++) AMPI_Wait(&reqs[i], &status);
/*
  MPI_Status *status = new MPI_Status[size];
  AMPI_Waitall(size, reqs, status);
  delete [] status;
*/

  delete [] reqs;
#else
    // use blocking recv
  dttype = ptr->getDDT()->getType(sendtype) ;
  itemsize = dttype->getSize(sendcount) ;
  int rank = ptr->getRank(comm);

  if ( itemsize <= AMPI_ALLTOALL_MEDIUM_MSG)
  {
#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAlltoall().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount,
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAlltoall().endIteration();
  } else
#endif 
  {
      for(i=0;i<size;i++) {
          int dst = (rank+i) % size;
          ptr->send(MPI_ATA_TAG, rank, ((char*)sendbuf)+(itemsize*dst), sendcount,
                    sendtype, dst, comm);
      }
  }
  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  MPI_Status status;
  for(i=0;i<size;i++) {
        int dst = (rank+i) % size;
        AMPI_Recv(((char*)recvbuf)+(itemsize*dst), recvcount, recvtype,
              dst, MPI_ATA_TAG, comm, &status);
  }
  }
  else {    // large messages
      /* Long message. Use pairwise exchange. If comm_size is a
         power-of-two, use exclusive-or to create pairs. Else send
         to rank+i, receive from rank-i. */

    int pof2;
    int src, dst;
      /* Is comm_size a power-of-two? */
    i = 1;
    while (i < size)
          i *= 2;
    if (i == size)
          pof2 = 1;
    else
          pof2 = 0;

      /* The i=0 case takes care of moving local data into recvbuf */
    for (i=0; i<size; i++) {
        if (pof2 == 1) {
              /* use exclusive-or algorithm */
              src = dst = rank ^ i;
        }
        else {
              src = (rank - i + size) % size;
              dst = (rank + i) % size;
        }

        MPI_Status status;
        MPI_Sendrecv(((char *)sendbuf + dst*itemsize),
                                   sendcount, sendtype, dst,
                                   MPI_ATA_TAG,
                                   ((char *)recvbuf + src*itemsize),
                                   recvcount, recvtype, src,
                                   MPI_ATA_TAG, comm, &status);
    }   // end of large message
  }
#endif

#if AMPI_COUNTER
  getAmpiParent()->counters.alltoall++;
#endif
  return 0;
}

CDECL
int AMPI_Alltoall2(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoall2");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Alltoall not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  CProxy_ampi pa(ptr->ckGetArrayID());
  int size = ptr->getSize(comm);
  CkDDT_DataType *dttype;
  int itemsize;
  int recvdisp;
  int myrank;
  int i;
  // Set flags for others to get
  ptr->setA2AIGetFlag((void*)sendbuf);
  MPI_Comm_rank(comm,&myrank);
  recvdisp = myrank*recvcount;

  AMPI_Barrier(comm);
    // post receives
  MPI_Request *reqs = new MPI_Request[size];
  for(i=0;i<size;i++) {
	  reqs[i] = pa[i].Alltoall_RemoteIGet(recvdisp, recvcount, recvtype,
MPI_ATA_TAG);
  }

  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  AmpiMsg *msg;
  for(i=0;i<size;i++) {
	  msg = (AmpiMsg*)CkWaitReleaseFuture(reqs[i]);
	  memcpy((char*)recvbuf+(itemsize*i), msg->data,itemsize);
	  delete msg;
  }
  
  delete [] reqs;
  AMPI_Barrier(comm);

  // Reset flags 
  ptr->resetA2AIGetFlag();
  
#if AMPI_COUNTER
  getAmpiParent()->counters.alltoall++;
#endif
  return 0;
}

CDECL
int AMPI_Ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ialltoall");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Ialltoall not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  ampi *ptr = getAmpiInstance(comm);
  AmpiRequestList* reqs = getReqs();
  int size = ptr->getSize(comm);
  int reqsSize = reqs->size();
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
  int itemsize = dttype->getSize(sendcount) ;
  int i;

#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAlltoall().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_ATA_TAG+reqsSize, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount,
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAlltoall().endIteration();
  } else
#endif
    for(i=0;i<size;i++) {
      ptr->send(MPI_ATA_TAG+reqsSize, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount,
		sendtype, i, comm);
    }
  
  // copy+paste from MPI_Irecv
  ATAReq *newreq = new ATAReq(size);
  for(i=0;i<size;i++){
    if(newreq->addReq(((char*)recvbuf)+(itemsize*i),recvcount,recvtype,i,MPI_ATA_TAG+reqsSize,comm)!=(i+1))
      CkAbort("MPI_Ialltoall: Error adding requests into ATAReq!");
  }
  *request = reqs->insert(newreq);
  AMPI_DEBUG("MPI_Ialltoall: request=%d, reqs.size=%d, &reqs=%d\n",*request,reqs->size(),reqs);
  return 0;
}

CDECL
int AMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                  MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoallv");
  if(getAmpiParent()->isInter(comm)) CkAbort("MPI_Alltoallv not allowed for Inter-communicator!");
  if(comm==MPI_COMM_SELF) return 0;
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
  int itemsize = dttype->getSize() ;
  int i;
#if AMPI_COMLIB
  if(comm == MPI_COMM_WORLD) {
      // commlib support
      ptr->getAlltoall().beginIteration();
      for(i=0;i<size;i++) {
          ptr->delesend(MPI_GATHER_TAG,ptr->getRank(comm),((char*)sendbuf)+(itemsize*sdispls[i]),sendcounts[i],
                        sendtype, i, comm, ptr->getComlibProxy());
      }
      ptr->getAlltoall().endIteration();
  } else
#endif
      for(i=0;i<size;i++) 
          ptr->send(MPI_GATHER_TAG,ptr->getRank(comm),((char*)sendbuf)+(itemsize*sdispls[i]),sendcounts[i],
                    sendtype, i, comm);
  
  MPI_Status status;
  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize() ;

  for(i=0;i<size;i++) {
    AMPI_Recv(((char*)recvbuf)+(itemsize*rdispls[i]), recvcounts[i], recvtype,
             i, MPI_GATHER_TAG, comm, &status);
  }
#if AMPI_COUNTER
  getAmpiParent()->counters.alltoall++;
#endif
  return 0;
}

CDECL
int AMPI_Comm_dup(int comm, int *newcomm)
{
  AMPIAPI("AMPI_Comm_dup");
  *newcomm = comm;
  return 0;
}

CDECL
int AMPI_Comm_split(int src,int color,int key,int *dest)
{
  AMPIAPI("AMPI_Comm_split");
  getAmpiInstance(src)->split(color,key,dest, 0);
  return 0;
}

CDECL
int AMPI_Comm_free(int *comm)
{
  AMPIAPI("AMPI_Comm_free");
  return 0;
}

CDECL
int AMPI_Comm_test_inter(MPI_Comm comm, int *flag){
  AMPIAPI("AMPI_Comm_test_inter");
  *flag = getAmpiParent()->isInter(comm);
  return 0;
}

CDECL
int AMPI_Comm_remote_size(MPI_Comm comm, int *size){
  AMPIAPI("AMPI_Comm_remote_size");
  *size = getAmpiParent()->getRemoteSize(comm);
  return 0;
}

CDECL
int AMPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group){
  AMPIAPI("AMPI_Comm_remote_group");
  *group = getAmpiParent()->getRemoteGroup(comm);
  return 0;
}

CDECL
int AMPI_Intercomm_create(MPI_Comm lcomm, int lleader, MPI_Comm rcomm, int rleader, int tag, MPI_Comm *newintercomm){
  AMPIAPI("AMPI_Intercomm_create");
  ampi *ptr = getAmpiInstance(lcomm);
  int root = ptr->getIndexForRank(lleader);
  CkVec<int> rvec;
  int lrank;
  AMPI_Comm_rank(lcomm,&lrank);

  if(lrank==lleader){
    int lsize, rsize;
    lsize = ptr->getSize(lcomm);
    int *larr = new int [lsize];
    int *rarr;
    CkVec<int> lvec = ptr->getIndices();
    MPI_Status sts;

    // local leader exchanges groupStruct with remote leader
    int i;
    for(i=0;i<lsize;i++)
      larr[i] = lvec[i];
    AMPI_Send(&lsize,1,MPI_INT,rleader,tag,rcomm);
    AMPI_Recv(&rsize,1,MPI_INT,rleader,tag,rcomm,&sts);

    rarr = new int [rsize];
    AMPI_Send(larr,lsize,MPI_INT,rleader,tag+1,rcomm);
    AMPI_Recv(rarr,rsize,MPI_INT,rleader,tag+1,rcomm,&sts);
    for(i=0;i<rsize;i++)
      rvec.push_back(rarr[i]);

    delete [] larr;
    delete [] rarr;

    if(rsize==0) CkAbort("MPI_Intercomm_create: remote size = 0! Does it really make sense to create an empty communicator?\n");
  }
  
  ptr->intercommCreate(rvec,root,newintercomm);
  return 0;
}

CDECL
int AMPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm){
  AMPIAPI("AMPI_Intercomm_merge");
  ampi *ptr = getAmpiInstance(intercomm);
  int lroot, rroot, lrank, lhigh, rhigh, first;
  lroot = ptr->getIndexForRank(0);
  rroot = ptr->getIndexForRemoteRank(0);
  lhigh = high;
  lrank = ptr->getRank(intercomm);

  if(lrank==0){
    MPI_Status sts;
    AMPI_Send(&lhigh,1,MPI_INT,0,10010,intercomm);
    AMPI_Recv(&rhigh,1,MPI_INT,0,10010,intercomm,&sts);

    if((lhigh && rhigh) || (!lhigh && !rhigh)){ // same value: smaller root goes first (first=1 if local goes first)
      first = (lroot < rroot);
    }else{ // different values, then high=false goes first
      first = (lhigh == false);
    }
  }

  ptr->intercommMerge(first, newintracomm);
  return 0;
}

CDECL
int AMPI_Abort(int comm, int errorcode)
{
  AMPIAPI("AMPI_Abort");
  CkAbort("AMPI: User called MPI_Abort!\n");
  return errorcode;
}

CDECL
int AMPI_Get_count(MPI_Status *sts, MPI_Datatype dtype, int *count){
  AMPIAPI("AMPI_Get_count");
  CkDDT_DataType* dttype = getDDT()->getType(dtype);
  int itemsize = dttype->getSize() ;
  *count = sts->MPI_LENGTH/itemsize;
  return 0;
}

CDECL
int AMPI_Type_lb(MPI_Datatype dtype, MPI_Aint* displacement){
  AMPIAPI("AMPI_Type_lb");
  *displacement = getDDT()->getLB(dtype);
  return 0;
}

CDECL
int AMPI_Type_ub(MPI_Datatype dtype, MPI_Aint* displacement){
  AMPIAPI("AMPI_Type_ub");
  *displacement = getDDT()->getUB(dtype);
  return 0;
}

CDECL
int AMPI_Address(void* location, MPI_Aint *address){
  AMPIAPI("AMPI_Address");
  *address = (MPI_Aint)(unsigned long)(char *)location;
  return 0;
}

CDECL
int AMPI_Get_elements(MPI_Status *sts, MPI_Datatype dtype, int *count){
  AMPIAPI("AMPI_Get_elements");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int basesize = dttype->getBaseSize() ;
  if(basesize==0) basesize=dttype->getSize();
  *count = sts->MPI_LENGTH/basesize;
  return 0;
}

CDECL
int AMPI_Pack(void *inbuf, int incount, MPI_Datatype dtype, void *outbuf,
              int outsize, int *position, MPI_Comm comm)
{
  AMPIAPI("AMPI_Pack");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize((char*)inbuf, ((char*)outbuf)+(*position), incount, 1);
  *position += (itemsize*incount);
  return 0;
}

CDECL
int AMPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
              int outcount, MPI_Datatype dtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Unpack");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize(((char*)inbuf+(*position)), (char*)outbuf, outcount, 1);
  *position += (itemsize*outcount);
  return 0;
}

CDECL
int AMPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz)
{
  AMPIAPI("AMPI_Pack_size");
  CkDDT_DataType* dttype = getDDT()->getType(datatype) ;
  *sz = incount*dttype->getSize() ;
  return 0;
}

CDECL
int AMPI_Get_processor_name(char *name, int *resultlen){
  AMPIAPI("AMPI_Get_processor_name");
  ampiParent *ptr = getAmpiParent();
  sprintf(name,"AMPI_VP[%d]_PE[%d]",ptr->thisIndex,ptr->getMyPe());
  *resultlen = strlen(name);
  return 0;
}

/* Error handling */
#if defined(USE_STDARG)
void error_handler(MPI_Comm *, int *, ...);
#else
void error_handler ( MPI_Comm *, int * );
#endif

CDECL
int AMPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler){
	AMPIAPI("AMPI_Errhandler_create");
	return MPI_SUCCESS;
}

CDECL
int AMPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler){
	AMPIAPI("AMPI_Errhandler_set");
	return MPI_SUCCESS;
}

CDECL
int AMPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler){
	AMPIAPI("AMPI_Errhandler_get");
	return MPI_SUCCESS;
}

CDECL
int AMPI_Errhandler_free(MPI_Errhandler *errhandler){
	AMPIAPI("AMPI_Errhandler_free");
	return MPI_SUCCESS;
}

CDECL
int AMPI_Error_class(int errorcode, int *errorclass){
	AMPIAPI("AMPI_Error_class");
	*errorclass = errorcode;
	return MPI_SUCCESS;
}

CDECL
int AMPI_Error_string(int errorcode, char *string, int *resultlen)
{
  AMPIAPI("AMPI_Error_string");
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


/* Group operations */
CDECL
int AMPI_Comm_group(MPI_Comm comm, MPI_Group *group)
{
  AMPIAPI("AMPI_Comm_Group");
  *group = getAmpiParent()->comm2group(comm);
  return 0;
}

CDECL
int AMPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_union");
  groupStruct vec1, vec2, newvec;
  ampiParent *ptr = getAmpiParent();
  vec1 = ptr->group2vec(group1);
  vec2 = ptr->group2vec(group2);
  newvec = unionOp(vec1,vec2);
  *newgroup = ptr->saveGroupStruct(newvec);
  return 0;
}

CDECL
int AMPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_intersection");
  groupStruct vec1, vec2, newvec;
  ampiParent *ptr = getAmpiParent();
  vec1 = ptr->group2vec(group1);
  vec2 = ptr->group2vec(group2);
  newvec = intersectOp(vec1,vec2);
  *newgroup = ptr->saveGroupStruct(newvec);
  return 0;
}

CDECL
int AMPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_difference");
  groupStruct vec1, vec2, newvec;
  ampiParent *ptr = getAmpiParent();
  vec1 = ptr->group2vec(group1);
  vec2 = ptr->group2vec(group2);
  newvec = diffOp(vec1,vec2);
  *newgroup = ptr->saveGroupStruct(newvec);
  return 0;
}

CDECL
int AMPI_Group_size(MPI_Group group, int *size)
{
  AMPIAPI("AMPI_Group_size");
  *size = (getAmpiParent()->group2vec(group)).size();
  return 0;
}

CDECL
int AMPI_Group_rank(MPI_Group group, int *rank)
{
  AMPIAPI("AMPI_Group_rank");
  *rank = getAmpiParent()->getRank(group);
  return 0;
}

CDECL
int AMPI_Group_translate_ranks (MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2)
{
  AMPIAPI("AMPI_Group_translate_ranks");
  ampiParent *ptr = getAmpiParent();
  groupStruct vec1, vec2;
  vec1 = ptr->group2vec(group1);
  vec2 = ptr->group2vec(group2);
  ranks2 = translateRanksOp(n, vec1, ranks1, vec2);
  return 0;
}

CDECL
int AMPI_Group_compare(MPI_Group group1,MPI_Group group2, int *result)
{
  AMPIAPI("AMPI_Group_compare");
  ampiParent *ptr = getAmpiParent();
  groupStruct vec1, vec2;
  vec1 = ptr->group2vec(group1);
  vec2 = ptr->group2vec(group2);
  *result = compareVecOp(vec1, vec2);
  return 0;
}

CDECL
int AMPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_incl");
  groupStruct vec, newvec;
  ampiParent *ptr = getAmpiParent();
  vec = ptr->group2vec(group);
  newvec = inclOp(n,ranks,vec);
  *newgroup = ptr->saveGroupStruct(newvec);
  return 0;
}
CDECL
int AMPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_excl");
  groupStruct vec, newvec;
  ampiParent *ptr = getAmpiParent();
  vec = ptr->group2vec(group);
  newvec = exclOp(n,ranks,vec);
  *newgroup = ptr->saveGroupStruct(newvec);
//outputOp(vec); outputOp(newvec);
  return 0;
}
CDECL
int AMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_range_incl");
  groupStruct vec, newvec;
  ampiParent *ptr = getAmpiParent();
  vec = ptr->group2vec(group);
  newvec = rangeInclOp(n,ranges,vec);
  *newgroup = ptr->saveGroupStruct(newvec);
  return 0;
}
CDECL
int AMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_range_excl");
  groupStruct vec, newvec;
  ampiParent *ptr = getAmpiParent();
  vec = ptr->group2vec(group);
  newvec = rangeExclOp(n,ranges,vec);
  *newgroup = ptr->saveGroupStruct(newvec);
  return 0;
}
CDECL
int AMPI_Group_free(MPI_Group *group)
{
  AMPIAPI("AMPI_Group_free");
  return 0;
}
CDECL
int AMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm)
{
  AMPIAPI("AMPI_Comm_create");
  groupStruct vec = getAmpiParent()->group2vec(group);
  if(vec.size()==0) CkAbort("AMPI> Abort: Does it really make sense to create an empty communicator?");
  getAmpiInstance(comm)->commCreate(vec, newcomm);
  return 0;
}

/* Charm++ Extentions to MPI standard: */
CDECL
void AMPI_Checkpoint(char *dname)
{
  AMPIAPI("AMPI_Checkpoint");
  AMPI_Barrier(MPI_COMM_WORLD);
  getAmpiParent()->startCheckpoint(dname);
}

CDECL
void AMPI_MemCheckpoint()
{
#if CMK_MEM_CHECKPOINT
  AMPIAPI("AMPI_Checkpoint");
  AMPI_Barrier(MPI_COMM_WORLD);
  getAmpiParent()->startCheckpoint("");
#else
  CmiPrintf("Error: In memory checkpoint/restart is not on! \n");
  CmiAbort("Error: recompile Charm++ with CMK_MEM_CHECKPOINT. \n");
#endif
}

CDECL
void AMPI_Print(char *str)
{
  AMPIAPI("AMPI_Print");
  ampiParent *ptr = getAmpiParent();
  CkPrintf("[%d] %s\n", ptr->thisIndex, str);
}

CDECL
int AMPI_Register(void *d, MPI_PupFn f)
{
  AMPIAPI("AMPI_Register");
  return TCHARM_Register(d,f);
}

CDECL
void *MPI_Get_userdata(int idx)
{
  AMPIAPI("AMPI_Get_userdata");
  return TCHARM_Get_userdata(idx);
}

CDECL
void AMPI_Register_main(MPI_MainFn mainFn,const char *name)
{
  AMPIAPI("AMPI_Register_main");
  if (TCHARM_Element()==0)
  { // I'm responsible for building the TCHARM threads:
    ampiCreateMain(mainFn,name,strlen(name));
  }
}
FDECL
void FTN_NAME(MPI_REGISTER_MAIN,mpi_register_main)
  (MPI_MainFn mainFn,const char *name,int nameLen)
{
  AMPIAPI("AMPI_register_main");
  if (TCHARM_Element()==0)
  { // I'm responsible for building the TCHARM threads:
    ampiCreateMain(mainFn,name,nameLen);
  }
}

CDECL
int AMPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void* extra_state){
  AMPIAPI("AMPI_Keyval_create");
  return getAmpiParent()->createKeyval(copy_fn,delete_fn,keyval,extra_state);
}

CDECL
int AMPI_Keyval_free(int *keyval){
  AMPIAPI("AMPI_Keyval_free");
  return getAmpiParent()->freeKeyval(keyval);
}

CDECL
int AMPI_Attr_put(MPI_Comm comm, int keyval, void* attribute_val){
  AMPIAPI("AMPI_Attr_put");
  return getAmpiParent()->putAttr(comm,keyval,attribute_val);
}

CDECL
int AMPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag){
  AMPIAPI("AMPI_Attr_get");
  return getAmpiParent()->getAttr(comm,keyval,attribute_val,flag);
}

CDECL
int AMPI_Attr_delete(MPI_Comm comm, int keyval){
  AMPIAPI("AMPI_Attr_delete");
  return getAmpiParent()->deleteAttr(comm,keyval);
}

CDECL
int AMPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods,
		 int *newrank) {
  AMPIAPI("AMPI_Cart_map");

  AMPI_Comm_rank(comm, newrank);

  return 0;
}

CDECL
int AMPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges,
		  int *newrank) {
  AMPIAPI("AMPI_Graph_map");
  AMPI_Comm_rank(comm, newrank);

  return 0;
}

CDECL
int AMPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, int *periods,
                   int reorder, MPI_Comm *comm_cart) {

  AMPIAPI("AMPI_Cart_create");

  /* Create new cartesian communicator. No attention is being paid to mapping
     virtual processes to processors, which ideally should be handled by the
     load balancer with input from virtual topology information.
     
     No reorder done here. reorder input is ignored, but still stored in the
     communicator with other VT info.
  */
  
  int newrank;
  AMPI_Cart_map(comm_old, ndims, dims, periods, &newrank);//no change in rank 

  ampiParent *ptr = getAmpiParent();
  groupStruct vec = ptr->group2vec(ptr->comm2group(comm_old));
  getAmpiInstance(comm_old)->cartCreate(vec, comm_cart);
  ampiCommStruct &c = ptr->getCart(*comm_cart);
  c.setndims(ndims);
  
  CkVec<int> dimsv;
  CkVec<int> periodsv;

  for (int i = 0; i < ndims; i++) {
    dimsv.push_back(dims[i]);
    periodsv.push_back(periods[i]);
    if ((periods[i] != 0) && (periods[i] != 1))
      CkAbort("MPI_Cart_create: periods should be all booleans\n");
  }

  c.setdims(dimsv);
  c.setperiods(periodsv);

  return 0;
}

CDECL
int AMPI_Graph_create(MPI_Comm comm_old, int nnodes, int *index, int *edges,
		     int reorder, MPI_Comm *comm_graph) {
  AMPIAPI("AMPI_Graph_create");
  
  /* No mapping done */
  int newrank;
  AMPI_Graph_map(comm_old, nnodes, index, edges, &newrank);
  
  ampiParent *ptr = getAmpiParent();
  groupStruct vec = ptr->group2vec(ptr->comm2group(comm_old));
  getAmpiInstance(comm_old)->graphCreate(vec, comm_graph);
  
  ampiCommStruct &c = ptr->getGraph(*comm_graph);
  c.setnvertices(nnodes);

  CkVec<int> index_;
  CkVec<int> edges_;

  int i;
  for (i = 0; i < nnodes; i++)
    index_.push_back(index[i]);
  
  c.setindex(index_);

  for (i = 0; i < index[nnodes - 1]; i++)
    edges_.push_back(edges[i]);

  c.setedges(edges_);

  return 0;
}

CDECL
int AMPI_Topo_test(MPI_Comm comm, int *status) {
  AMPIAPI("AMPI_Topo_test");
  
  ampiParent *ptr = getAmpiParent();
  
  if (ptr->isCart(comm))
    *status = MPI_CART;
  else if (ptr->isGraph(comm))
    *status = MPI_GRAPH;
  else *status = MPI_UNDEFINED;
  
  return 0;
}

CDECL
int AMPI_Cartdim_get(MPI_Comm comm, int *ndims) {
  AMPIAPI("AMPI_Cartdim_get");

  *ndims = getAmpiParent()->getCart(comm).getndims();

  return 0;
}

CDECL
int AMPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, 
		 int *coords){
  int i, ndims;

  AMPIAPI("AMPI_Cart_get");

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  ndims = c.getndims();
  int rank;

  AMPI_Comm_rank(comm, &rank);

  const CkVec<int> &dims_ = c.getdims();
  const CkVec<int> &periods_ = c.getperiods();
  
  for (i = 0; i < maxdims; i++) {
    dims[i] = dims_[i];
    periods[i] = periods_[i];
  }

  for (i = ndims - 1; i >= 0; i--) {
    if (i < maxdims)
      coords[i] = rank % dims_[i];
    rank = (int) (rank / dims_[i]);
  }

  return 0;
}

CDECL
int AMPI_Cart_rank(MPI_Comm comm, int *coords, int *rank) {
  AMPIAPI("AMPI_Cart_rank");

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  int ndims = c.getndims();
  const CkVec<int> &dims = c.getdims();
  const CkVec<int> &periods = c.getperiods();

  int prod = 1;
  int r = 0;

  for (int i = ndims - 1; i >= 0; i--) {
    if ((coords[i] < 0) || (coords[i] >= dims[i]))
      if (periods[i] == 1)
	if (coords[i] > 0)
	  coords[i] %= dims[i];
	else
	  coords[i] += (((-coords[i] / dims[i]) + 1) * dims[i]) % dims[i];
    r += prod * coords[i];
    prod *= dims[i];
  }

  *rank = r;

  return 0;
}

CDECL
int AMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords) {
  AMPIAPI("AMPI_Cart_coords");

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  int ndims = c.getndims();
  const CkVec<int> &dims = c.getdims();

  for (int i = ndims - 1; i >= 0; i--) {
    if (i < maxdims)
      coords[i] = rank % dims[i];
    rank = (int) (rank / dims[i]);
  }
  
  return 0;
}

CDECL
int AMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, 
		   int *rank_dest) {
  AMPIAPI("AMPI_Cart_shift");
  
  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  int ndims = c.getndims();
  const CkVec<int> &dims = c.getdims();
  const CkVec<int> &periods = c.getperiods();
  int *coords = new int[ndims];

  AMPI_Comm_rank(comm, rank_source);
  AMPI_Cart_coords(comm, *rank_source, ndims, coords);

  if ((direction < 0) || (direction >= ndims))
    CkAbort("MPI_Cart_shift: direction not within dimensions range");

  coords[direction] += disp;
  if (coords[direction] < 0)
    if (periods[direction] == 1) {
      coords[direction] += dims[direction];
      AMPI_Cart_rank(comm, coords, rank_dest);
    }
    else
      *rank_dest = MPI_PROC_NULL;
  else
    AMPI_Cart_rank(comm, coords, rank_dest);

  delete [] coords;
  return 0;
}

CDECL
int AMPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges) {
  AMPIAPI("AMPI_Graphdim_get");

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);
  *nnodes = c.getnvertices();
  const CkVec<int> &index = c.getindex();
  *nedges = index[(*nnodes) - 1];
  
  return 0;
}

CDECL
int AMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index, 
		  int *edges) {
  AMPIAPI("AMPI_Graph_get");

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);

  const CkVec<int> &index_ = c.getindex();
  const CkVec<int> &edges_ = c.getedges();

  if (maxindex > index_.size())
    maxindex = index_.size();

  int i;
  for (i = 0; i < maxindex; i++)
    index[i] = index_[i];

  for (i = 0; i < maxedges; i++)
    edges[i] = edges_[i];

  return 0;
} 

CDECL
int AMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors) {
  AMPIAPI("AMPI_Graph_neighbors_count");

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);

  const CkVec<int> &index = c.getindex();

  if ((rank >= index.size()) || (rank < 0))
    CkAbort("MPI_Graph_neighbors_count: rank not within range");

  if (rank == 0)
    *nneighbors = index[rank];
  else 
    *nneighbors = index[rank] - index[rank - 1];

  return 0;
}

CDECL
int AMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors,
			int *neighbors) {
  AMPIAPI("AMPI_Graph_neighbors");

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);
  const CkVec<int> &index = c.getindex();
  const CkVec<int> &edges = c.getedges();
  
  int numneighbors = (rank == 0) ? index[rank] : index[rank] - index[rank - 1];
  if (maxneighbors > numneighbors)
    maxneighbors = numneighbors;

  if (maxneighbors < 0)
    CkAbort("MPI_Graph_neighbors: maxneighbors < 0");

  if ((rank >= index.size()) || (rank < 0))
    CkAbort("MPI_Graph_neighbors: rank not within range");

  if (rank == 0)
    for (int i = 0; i < maxneighbors; i++)
      neighbors[i] = edges[i];
  else
    for (int i = 0; i < maxneighbors; i++)
      neighbors[i] = edges[index[rank - 1] + i];

  return 0;
}

/* Factorization code by Orion. Idea thrashed out by Orion and Prakash */

/**
  Return the integer "d'th root of n"-- the largest
  integer r such that
        r^d <= n
*/
int integerRoot(int n,int d) {
        double epsilon=0.001; /* prevents roundoff in "floor" */
        return (int)floor(pow(n+epsilon,1.0/d));
}

/**
  Factorize "n" into "d" factors, stored in "dims[0..d-1]".
  All the factors must be greater than or equal to m.
  The factors are chosen so that they are all as near together
  as possible (technically, chosen so that the increasing-size
  ordering is lexicagraphically as large as possible).
*/

bool factors(int n, int d, int *dims, int m) {
        if (d==1)
        { /* Base case */
                if (n>=m) { /* n is an acceptable factor */
                        dims[0]=n;
                        return true;
                }
        }
        else { /* induction case */
                int k_up=integerRoot(n,d);
                for (int k=k_up;k>=m;k--)
                if (n%k==0) { /* k divides n-- try it as a factor */
                        dims[0]=k;
                        if (factors(n/k,d-1,&dims[1],k))
                                return true;
                }
        }
        /* If we fall out here, there were no factors available */
        return false;
}

CDECL
int AMPI_Dims_create(int nnodes, int ndims, int *dims) {
  AMPIAPI("AMPI_Dims_create");

  int i, n, d, *pdims;

  n = nnodes;
  d = ndims;

  for (i = 0; i < ndims; i++)
    if (dims[i] != 0)
      if (n % dims[i] != 0)
	CkAbort("MPI_Dims_Create: Value in dimensions array infeasible!");
      else {
	n = n / dims[i];
	d--;
      }

  pdims = new int[d];

  if (!factors(n, d, pdims, 1))
    CkAbort("MPI_Dims_Create: Factorization failed. Wonder why?");

  int j = 0;
  for (i = 0; i < ndims; i++)
    if (dims[i] == 0) {
      dims[i] = pdims[j];
      j++;
    }

  delete [] pdims;

  return 0;
}

/* Implemented with call to MPI_Comm_Split. Color and key are single integer
   encodings of the lost and preserved dimensions, respectively,
   of the subgraphs.
*/

CDECL
int AMPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *newcomm) {
  AMPIAPI("AMPI_Cart_sub");

  int i, *coords, ndims, rank;
  int color = 1, key = 1;

  AMPI_Comm_rank(comm, &rank);
  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  ndims = c.getndims();
  const CkVec<int> &dims = c.getdims();
  int num_remain_dims = 0;

  coords = new int [ndims];
  AMPI_Cart_coords(comm, rank, ndims, coords);

  for (i = 0; i < ndims; i++)
    if (remain_dims[i]) {
      /* key single integer encoding*/
      key = key * dims[i] + coords[i];
      num_remain_dims++;
    }
    else
      /* color */
      color = color * dims[i] + coords[i];

  getAmpiInstance(comm)->split(color, key, newcomm, CART_TOPOL);

  ampiCommStruct &newc = getAmpiParent()->getCart(*newcomm);
  newc.setndims(num_remain_dims);
  CkVec<int> dimsv;
  const CkVec<int> &periods = c.getperiods();
  CkVec<int> periodsv;

  for (i = 0; i < ndims; i++)
    if (remain_dims[i]) {
      dimsv.push_back(dims[i]);
      periodsv.push_back(periods[i]);
    }

  newc.setdims(dimsv);
  newc.setperiods(periodsv);

  delete [] coords;
  return 0;
}

void _registerampif(void)
{
  _registerampi();
}

void AMPI_Datatype_iscontig(MPI_Datatype datatype, int *flag){
  *flag = getDDT()->isContig(datatype);
}

CDECL
int AMPI_Type_get_envelope(MPI_Datatype datatype, int *ni, int *na, int *nd, int *combiner){
  AMPIAPI("AMPI_Type_get_envelope");
  return getDDT()->getEnvelope(datatype,ni,na,nd,combiner);
}

CDECL
int AMPI_Type_get_contents(MPI_Datatype datatype, int ni, int na, int nd, int i[], MPI_Aint a[], MPI_Datatype d[]){
  AMPIAPI("AMPI_Type_get_contents");
  return getDDT()->getContents(datatype,ni,na,nd,i,a,d);
}

/******** AMPI-specific (not standard MPI) calls *********/

CDECL
int AMPI_Suspend(int comm) {
	AMPIAPI("AMPI_Suspend");
	getAmpiInstance(comm)->block();
	return 0;
}

CDECL
int AMPI_Yield(int comm) {
	AMPIAPI("AMPI_Yield");
	getAmpiInstance(comm)->yield();
	return 0;
}

CDECL
int AMPI_Resume(int dest, int comm) {
	AMPIAPI("AMPI_Resume");
	getAmpiInstance(comm)->getProxy()[dest].unblock();
	return 0;
}

CDECL
int AMPI_System(const char *cmd) {
	return TCHARM_System(cmd);
}


#include "ampi.def.h"

