/*Charm++ Multiblock CFD Framework:
C++ implementation file

This is the under-the-hood implementation file for MBLOCK.
Milind Bhandarkar
*/
#ifndef _MBLOCK_IMPL_H
#define _MBLOCK_IMPL_H

#include <stdio.h>

#include "mblock.decl.h"
#include "mblock.h"
#include "patch.h"

extern CkChareID _mainhandle;
extern CkArrayID _mblockaid;
extern int _nchunks;

#define CHK(p) do{if((p)==0)CkAbort("MBLK>Memory Allocation failure.");}while(0)

// temporary Datatype representation
// will go away once MPI user-defined datatypes are ready
struct DType {
  int base_type;
  int vec_len;
  int init_offset; // offset of field in bytes from the beginning of data
  int distance; // distance in bytes between successive field values
  DType(void) {}
  DType(const DType& dt) : 
    base_type(dt.base_type), vec_len(dt.vec_len), init_offset(dt.init_offset),
    distance(dt.distance) {}
  void operator=(const DType& dt) {
    base_type = dt.base_type; 
    vec_len = dt.vec_len; 
    init_offset = dt.init_offset;
    distance = dt.distance;
  }
  DType( const int b,  const int v=1,  const int i=0,  const int d=0) : 
    base_type(b), vec_len(v), init_offset(i) {
    distance = (d ? d : length());
  }
  int length(const int nitems=1) const {
    int blen;
    switch(base_type) {
      case MBLK_BYTE : blen = 1; break;
      case MBLK_INT : blen = sizeof(int); break;
      case MBLK_REAL : blen = sizeof(float); break;
      case MBLK_DOUBLE : blen = sizeof(double); break;
    }
    return blen * vec_len * nitems;
  }
};

class DataMsg : public CMessage_DataMsg
{
 public:
  int seqnum;
  int from;
  int dtype;
  void *data;
  int patchno;
  DataMsg(int s, int f, int d, int p) : 
    seqnum(s), from(f), dtype(d), patchno(p) { data = (void*) (this+1); }
  DataMsg(void) { data = (void*) (this+1); }
  static void *pack(DataMsg *);
  static DataMsg *unpack(void *);
  static void *alloc(int, size_t, int*, int);
};

class ChunkMsg : public CMessage_ChunkMsg
{
  public:
    char prefix[128];
    int ndims;
    ChunkMsg(const char *_p, const int n) : ndims(n) { strcpy(prefix, _p); }
};

#define MBLK_MAXDT 20
#define MBLK_MAXUDATA 20
#define MBLK_MAXBC 20

class chunk: public ArrayElement1D
{
private:
  DType dtypes[MBLK_MAXDT];
  int ntypes;

  CmmTable messages; // messages to be processed

  int nudata;
  void *userdata[MBLK_MAXUDATA];
  MBLK_PupFn pup_ud[MBLK_MAXUDATA];
  MBLK_BcFn bcfns[MBLK_MAXBC];

  void callDriver(void);

  blockLoc sbc, ebc;
  int seqnum;
  int wait_seqnum; // update # that we ar waiting for, -1 if not waiting
  int nRecvPatches;
  int nRecd;
  void *ogrid; // grid pointer for receiving updates

 public:
  block *b;

  chunk(void);
  chunk(CkMigrateMessage *msg){}
  ~chunk();
  
  //entry methods
  void run(ChunkMsg*);
  void recv(DataMsg *);
  void reductionResult(DataMsg *);

  int new_DT(int base_type, int vec_len=1, int init_offset=0, int distance=0) {
    if(ntypes==MBLK_MAXDT)
      return (-1);
    dtypes[ntypes] = DType(base_type, vec_len, init_offset, distance);
    ntypes++;
    return ntypes-1;
  }
  void update(int fid, void *ingrid, void *outgrid);
  int test_update(void);
  int wait_update(void);
  void reduce_field(int fid, const void *nodes, void *outbuf, int op);
  void reduce(int fid, const void *inbuf, void *outbuf, int op);
  int id(void) { return thisIndex; }
  int total(void) { return numElements; }
  void print(void);

  int register_bc(const int bcnum, const MBLK_BcFn bcfn)
  {
    if(bcnum<0 || bcnum>=MBLK_MAXBC) {
      CkError("MBLK> BC index should be between 0 and %d!\n", MBLK_MAXBC-1);
      return MBLK_FAILURE;
    }
    bcfns[bcnum] = bcfn;
    return MBLK_SUCCESS;
  }
  
  int apply_bc(const int bcnum, void *grid)
  {
    if(bcfns[bcnum] == 0) {
      CkError("MBLK> BC handler not registered for bcnum %d!\n", bcnum);
      return MBLK_FAILURE;
    }
    int n = b->getPatches();
    for (int i=0; i< n; i++) {
      patch *p = b->getPatch(i);
      if(p->type == patch::ext && ((externalBCpatch*)p)->bcNo == bcnum) {
        sbc = p->start; ebc = p->end;
        bcfns[bcnum](grid);
      }
    }
    return MBLK_SUCCESS;
  }

  int apply_bc_all(void *grid)
  {
    int i;
    for(i=0;i<MBLK_MAXBC;i++)
      if(bcfns[i] != 0)
        apply_bc(i, grid);
    return MBLK_SUCCESS;
  }

  int get_boundary_extent(int *start, int *end)
  {
    start[0] = sbc[0]; start[1] = sbc[1]; start[2] = sbc[2];
    end[0] = ebc[0]; end[1] = ebc[1]; end[2] = ebc[2];
    return MBLK_SUCCESS;
  }

  int register_userdata(void *_userdata,MBLK_PupFn _pup_ud)
  {
    if(nudata==MBLK_MAXUDATA) {
      CkAbort("MBLK> UserData registration limit exceeded.!\n");
    }
    userdata[nudata] = _userdata;
    pup_ud[nudata] = _pup_ud;
    return nudata++;
  }
  int check_userdata(int n);
  void *get_userdata(int n) { return userdata[check_userdata(n)]; }
    
  void pup(PUP::er &p);
  void readyToMigrate(void)
  {
    // CkPrintf("[%d] going to sync\n", thisIndex);
    AtSync();
    thread_suspend();
  }
  void ResumeFromSync(void)
  {
    // CkPrintf("[%d] returned from sync\n", thisIndex);
    thread_resume();
  }

 private:
  CthThread tid; // waiting thread, 0 if no one is waiting
  void thread_suspend(void); //Thread will block until resume
  void thread_resume(void);  //Start thread running again
  void start_running(void)
  {
    thisArray->the_lbdb->ObjectStart(ldHandle);
  }
  void stop_running(void)
  {
    thisArray->the_lbdb->ObjectStop(ldHandle);
  }
  void send(int fid, const void *grid);
  void update_field(DataMsg *m);
};



class main : public Chare
{
  int numdone;
 public:
  char prefix[128];
  int ndims;
  int nblocks;
  main(CkArgMsg *);
  void done(void);
};

/*Decide how to declare C functions that are called from Fortran--
  some fortran compiles expect all caps; some all lowercase, 
  but with a trailing underscore.*/
#if CMK_FORTRAN_USES_ALLCAPS
# define FTN_NAME(caps,nocaps) caps  /*Declare name in all caps*/
#else
# if CMK_FORTRAN_USES_x__
#  define FTN_NAME(caps,nocaps) nocaps##_ /*No caps, extra underscore*/
# else
#  define FTN_NAME(caps,nocaps) nocaps /*Declare name without caps*/
# endif /*__*/
#endif /*ALLCAPS*/

#define CDECL extern "C" /*Function declaration for C linking*/
#define FDECL extern "C" /*Function declaration for Fortran linking*/

#if MBLK_FORTRAN
FDECL void FTN_NAME(INIT,init_) (void);
FDECL void FTN_NAME(DRIVER,driver_) (void);
#else
extern void init(void);
extern void driver(void);
#endif

#endif


