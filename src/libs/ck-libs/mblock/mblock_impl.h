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

  void pup(PUP::er &p) {
    p(base_type);
    p(vec_len);
    p(init_offset);
    p(distance);
  }

};

class DataMsg : public CMessage_DataMsg
{
 public:
  int seqnum;
  int from;
  int fid;
  void *data;
  int patchno;
  DataMsg(int s, int f, int fid_, int p) : 
    seqnum(s), from(f), fid(fid_), patchno(p) { data = (void*) (this+1); }
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

class field_t {
 public:
  DType dt; //Describes 1D data layout
  blockDim dim; //User-allocated dimensions of field
  int arrayBytes; //Bytes in user-allocated array
  bool forVoxel; //Field is for a voxel, not a node attribute
  field_t(const DType &dt_,const blockDim &dim_,bool forVoxel_)
    :dt(dt_),dim(dim_),forVoxel(forVoxel_) {
    arrayBytes=dim.getSize()*dt.distance;
  }

  field_t() { }
  void pup(PUP::er &p) {
    dt.pup(p);
    dim.pup(p);
    p(forVoxel);
  }
};

class chunk: public ArrayElement1D
{
private:
  int nfields;
  field_t *fields[MBLK_MAXDT];

  int nudata;
  void *userdata[MBLK_MAXUDATA];
  MBLK_PupFn pup_ud[MBLK_MAXUDATA];

  struct {
    MBLK_BcFn fn;
    extrudeMethod m;
  } bcs[MBLK_MAXBC];

  CmmTable messages; // messages for updates yet to be initiated
  int seqnum; //Number of most recently initiated update
  struct {
    int nRecd; //Number of messages received so far
    extrudeMethod m;
    int wait_seqnum; // update # that we ar waiting for, -1 if not waiting
    void *ogrid; // grid pointer for receiving updates
  } update;
  int nRecvPatches;  //Number of messages to expect (a constant)

  void *reduce_output;//Result of reduction goes here

  void callDriver(void);

 public:
  block *b;

  chunk(ChunkMsg*);
  chunk(CkMigrateMessage *msg){}
  ~chunk();
  
  //entry methods
  void run(void);
  void recv(DataMsg *);
  void reductionResult(DataMsg *);

  int add_field(field_t *f) {
    if (nfields==MBLK_MAXDT) return -1;
    fields[nfields]=f;
    nfields++;
    return nfields-1;
  }

  void start_update(int fid,const extrudeMethod &m,void *ingrid, void *outgrid);
  int test_update(void);
  int wait_update(void);
  void reduce_field(int fid, const void *nodes, void *outbuf, int op);
  void reduce(int fid, const void *inbuf, void *outbuf, int op);
  int id(void) { return thisIndex; }
  int total(void) { return numElements; }
  void print(void);

  int register_bc(const int bcnum, const MBLK_BcFn bcfn,const extrudeMethod &m)
  {
    if(bcnum<0 || bcnum>=MBLK_MAXBC) {
      CkError("MBLK> BC index should be between 0 and %d!\n", MBLK_MAXBC-1);
      return MBLK_FAILURE;
    }
    bcs[bcnum].fn = bcfn;
    bcs[bcnum].m=m;
    return MBLK_SUCCESS;
  }

  void apply_single_bc(patch *p,void *p1,void *p2) {
    int bcNo=((externalBCpatch *)p)->bcNo;
    blockSpan s=p->getExtents(bcs[bcNo].m,true);
    int start[3],end[3];
#if MBLK_FORTRAN
    s.start=s.start+blockLoc(1,1,1);
    s.end=s.end+blockLoc(1,1,1);
#endif
    s.getInt3(start,end);
    (bcs[bcNo].fn)(p1,p2,start,end);
  }

  int apply_bc(const int bcnum, void *p1,void *p2)
  {
    if(bcs[bcnum].fn == 0) {
      CkError("MBLK> BC handler not registered for bcnum %d!\n", bcnum);
      return MBLK_FAILURE;
    }
    int n = b->getPatches();
    for (int i=0; i< n; i++) {
      patch *p = b->getPatch(i);
      if(p->type == patch::ext && ((externalBCpatch*)p)->bcNo == bcnum) 
	apply_single_bc(p,p1,p2);
    }
    return MBLK_SUCCESS;
  }

  int apply_bc_all(void *p1,void *p2)
  {
    int i;
    int n = b->getPatches();
    for(i=0;i<n;i++) {
      patch *p = b->getPatch(i);
      if(p->type == patch::ext)
	apply_single_bc(p,p1,p2);      
    }
    return MBLK_SUCCESS;
  }

  static void copy(const blockLoc &src,int *dest,int shift=0) {
    dest[0]=src[0]+shift; dest[1]=src[1]+shift; dest[2]=src[2]+shift;
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
#if CMK_LBDB_ON
    AtSync();
    thread_suspend();
#endif
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
#if CMK_LBDB_ON
    thisArray->the_lbdb->ObjectStart(ldHandle);
#endif
  }
  void stop_running(void)
  {
#if CMK_LBDB_ON
    thisArray->the_lbdb->ObjectStop(ldHandle);
#endif
  }
  void send(int fid,const extrudeMethod &m,const void *grid);
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


