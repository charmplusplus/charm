
/* define system parameters */
#define INBUFFER_SIZE	32

#define CYCLES_PER_HOP     5
#define CYCLES_PER_CORNER  75
#define CYCLE_TIME_FACTOR  0.001  /* one cycle = nanosecond = 10^(-3) us */
/* end of system parameters */

#define MAX_HANDLERS	100

static int arg_argc;
static char **arg_argv;

static int bgSize = 0;

CpvStaticDeclare(int, numX);	/* size of bluegene nodes in cube */
CpvStaticDeclare(int, numY);
CpvStaticDeclare(int, numZ);
CpvStaticDeclare(int, numCth);	/* number of threads */
CpvStaticDeclare(int, numWth);
CpvStaticDeclare(int, numNodes);	/* number of bg nodes on this PE */

typedef char ThreadType;
const char UNKNOWN_THREAD=0, COMM_THREAD=1, WORK_THREAD=2;

#define cva CpvAccess
#define cta CtvAccess

#define tMYID		cta(threadinfo)->id
#define tMYGLOBALID	cta(threadinfo)->globalId
#define tTHREADTYPE	cta(threadinfo)->type
#define tMYNODE		cta(threadinfo)->myNode
#define tSTARTTIME	tMYNODE->startTime
#define tCURRTIME	cta(threadinfo)->currTime
#define tHANDLETAB	cta(threadinfo)->handlerTable
#define tMYX		tMYNODE->x
#define tMYY		tMYNODE->y
#define tMYZ		tMYNODE->z
#define tMYNODEID	tMYNODE->id
#define tCOMMTHQ	tMYNODE->commThQ
#define tINBUFFER	cva(inBuffer)[tMYNODE->id]
#define tMSGBUFFER	cva(msgBuffer)[tMYNODE->id]
#define tUSERDATA	tMYNODE->udata
#define tTHREADTABLE    tMYNODE->threadTable
#define tAFFINITYQ      tMYNODE->affinityQ[tMYID]
#define tNODEQ          tMYNODE->nodeQ
#define tSTARTED        tMYNODE->started



/*****************************************************************************
   used internally, define BG Node to real Processor mapping
*****************************************************************************/

class BlockMapInfo {
public:
  /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = bgSize / CmiNumPes();
    m = bgSize % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map global serial number to (x,y,z) ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    *x = seq / (cva(numY) * cva(numZ));
    *y = (seq - *x * cva(numY) * cva(numZ)) / cva(numZ);
    *z = (seq - *x * cva(numY) * cva(numZ)) % cva(numZ);
  }


    /* calculate global serial number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    return x*(cva(numY) * cva(numZ)) + y*cva(numZ) + z;
  }

    /* map (x,y,z) to emulator PE ++++ */
  inline static int XYZ2PE(int x, int y, int z) {
    return Global2PE(XYZ2Global(x,y,z));
  }

  inline static int XYZ2Local(int x, int y, int z) {
    return Global2Local(XYZ2Global(x,y,z));
  }

    /* local node index number to x y z ++++ */
  inline static void Local2XYZ(int num, int *x, int *y, int *z)  {
    Global2XYZ(Local2Global(num), x, y, z);
  }

    /* map global serial node number to PE ++++ */
  inline static int Global2PE(int num) { 
    int n = bgSize/CmiNumPes();
    int bn = bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiNumPes(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      if (num >= start && num <= end) return i;
      start = end+1;
    }
    CmiAbort("Global2PE: unknown pe!");
  }

    /* map global serial node ID to local node array index  ++++ */
  inline static int Global2Local(int num) { 
    int n = bgSize/CmiNumPes();
    int bn = bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiNumPes(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      if (num >= start && num <= end) return num-start;
      start = end+1;
    }
    CmiAbort("Global2Local:unknown pe!");
  }

    /* map local node index to global serial node id ++++ */
  inline static int Local2Global(int num) { 
    int n = bgSize/CmiNumPes();
    int bn = bgSize%CmiNumPes();
    int start = 0; 
    int end = 0;
    for (int i=0; i< CmiMyPe(); i++) {
      end = start + n-1;
      if (i<bn) end++;
      start = end+1;
    }
    return start+num;
  }
};

class CyclicMapInfo {
public:
  /* return the number of bg nodes on this physical emulator PE */
  inline static int numLocalNodes()
  {
    int n, m;
    n = bgSize / CmiNumPes();
    m = bgSize % CmiNumPes();
    if (CmiMyPe() < m) n++;
    return n;
  }

    /* map global serial number to (x,y,z) ++++ */
  inline static void Global2XYZ(int seq, int *x, int *y, int *z) {
    *x = seq / (cva(numY) * cva(numZ));
    *y = (seq - *x * cva(numY) * cva(numZ)) / cva(numZ);
    *z = (seq - *x * cva(numY) * cva(numZ)) % cva(numZ);
  }


    /* calculate global serial number of (x,y,z) ++++ */
  inline static int XYZ2Global(int x, int y, int z) {
    return x*(cva(numY) * cva(numZ)) + y*cva(numZ) + z;
  }

    /* map (x,y,z) to emulator PE ++++ */
  inline static int XYZ2PE(int x, int y, int z) {
    return Global2PE(XYZ2Global(x,y,z));
  }

  inline static int XYZ2Local(int x, int y, int z) {
    return Global2Local(XYZ2Global(x,y,z));
  }

    /* local node index number to x y z ++++ */
  inline static void Local2XYZ(int num, int *x, int *y, int *z)  {
    Global2XYZ(Local2Global(num), x, y, z);
  }

    /* map global serial node number to PE ++++ */
  inline static int Global2PE(int num) { return num % CmiNumPes(); }

    /* map global serial node ID to local node array index  ++++ */
  inline static int Global2Local(int num) { return num/CmiNumPes(); }

    /* map local node index to global serial node id ++++ */
  inline static int Local2Global(int num) { return CmiMyPe()+num*CmiNumPes();}
};

