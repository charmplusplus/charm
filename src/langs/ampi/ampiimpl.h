/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _AMPIIMPL_H
#define _AMPIIMPL_H

#include "ampi.h"
#include "ampi.decl.h"
#include "ampimain.decl.h"
#include "ddt.h"
#include <sys/stat.h> // for mkdir

#define AMPI_MAX_COMM 8

#ifdef FNAME
#undef FNAME
#endif

#if CMK_FORTRAN_USES_TWOSCORE
#  define FNAME(x) x##__
#elif CMK_FORTRAN_USES_ONESCORE
#  define FNAME(x) x##_
#else
#  define FNAME(x) x
#endif

#if AMPI_FORTRAN
#  if CMK_FORTRAN_USES_ALLCAPS
#    define ampi_setup         AMPI_SETUP
#    define ampi_register_main AMPI_REGISTER_MAIN
#    define ampi_main          AMPI_MAIN
#  else
#    define ampi_setup         FNAME(ampi_setup)
#    define ampi_register_main FNAME(ampi_register_main)
#    define ampi_main          FNAME(ampi_main)
#  endif
#else
#  define ampi_setup         AMPI_Setup
#  define ampi_register_main AMPI_Register_main
#  define ampi_main          AMPI_Main
#endif

extern "C" void ampi_setup(void);
extern "C" void ampi_register_main(void (*)(int, char **), char *, int);

struct ampi_redn_spec
{
  int type;
  int root;
};

struct ampi_comm_struct
{
  CkArrayID aid;
  void (*mainfunc)(int, char **);
  char *name;
  int nobj;
  ampi_redn_spec rspec;
};

class AmpiStartMsg : public CMessage_AmpiStartMsg
{
  public:
    int commidx;
    AmpiStartMsg(int _idx) : commidx(_idx) {}
};

class ampimain : public Chare
{
  int nobjs;
  int numDone;
  int qwait;
  public:
    static CkChareID handle;
    static ampi_comm_struct ampi_comms[AMPI_MAX_COMM];
    static int ncomms;
    static void register_main(void (*)(int, char **), char *, int);
    ampimain(CkArgMsg *);
    ampimain(CkMigrateMessage *m) {}
    void done(void);
    void checkpoint(void);
    void checkpointOnQd(void);
};

static inline void 
itersDone(void) 
{ 
  CProxy_ampimain pm(ampimain::handle); 
  pm.done(); 
}

#define AMPI_BCAST_TAG  1025
#define AMPI_BARR_TAG   1026
#define AMPI_REDUCE_TAG 1027
#define AMPI_GATHER_TAG 1028

#if 0
// This is currently not used.
class BlockMap : public CkArrayMap {
 public:
  BlockMap(void) {}
  BlockMap(CkMigrateMessage *m) {}
  int registerArray(CkArrayMapRegisterMessage *m) {
    delete m;
    return 0;
  }
  int procNum(int /*arrayHdl*/,const CkArrayIndex &idx) {
    int elem=*(int *)idx.data();
    int penum =  (elem/(32/CkNumPes()));
    CkPrintf("%d mapped to %d proc\n", elem, penum);
    return penum;
  }
};
#endif

#define MyAlign8(x) (((x)+7)&(~7))

class PersReq {
  public:
    int sndrcv; // 1 if send , 2 if recv
    void *buf;
    int count;
    int type;
    int proc;
    int tag;
    int comm;
    int nextfree, prevfree;
};

class ArgsInfo : public CMessage_ArgsInfo {
  public:
    int argc;
    char **argv;
    ArgsInfo(void) { argc = 0; argv=0; }
    ArgsInfo(int c, char **v) { argc = c; argv = v; }
    void setargs(int c, char**v) { argc = c; argv = v; }
    static void* pack(ArgsInfo*);
    static ArgsInfo* unpack(void*);
};

class DirMsg : public CMessage_DirMsg {
  public:
    char *dname;
    DirMsg(char* d) { dname = new char[strlen(d)+1]; strcpy(dname, d); }
    ~DirMsg() { delete[] dname; }
    static void *pack(DirMsg *m)
    {
      void *buf = CkAllocBuffer(m, strlen(m->dname)+1);
      strcpy((char*)buf, m->dname);
      delete m;
      return buf;
    }
    static DirMsg* unpack(void *buf)
    {
      DirMsg *m = (DirMsg*) CkAllocBuffer(buf, sizeof(DirMsg));
      m = new ((void*)m) DirMsg((char*)buf);
      CkFreeMsg(buf);
      return m;
    }
};

class AmpiMsg : public CMessage_AmpiMsg {
 public:
  int tag1, tag2, comm, length;
  void *data;

  AmpiMsg(void) { data = (char *)this + sizeof(AmpiMsg); }
  AmpiMsg(int t1, int t2, int l, int c):tag1(t1),tag2(t2),length(l),comm(c) {
    data = (char *)this + sizeof(AmpiMsg);
  }
  static void *alloc(int msgnum, size_t size, int *sizes, int pbits) {
    return CkAllocMsg(msgnum, size+sizes[0], pbits);
  }
  static void *pack(AmpiMsg *in) { return (void *) in; }
  static AmpiMsg *unpack(void *in) { return new (in) AmpiMsg; }
  static AmpiMsg* pup(PUP::er &p, AmpiMsg *m)
  {
    int length, tag1, tag2, comm;
    if(p.isPacking() || p.isSizing()) {
      tag1 = m->tag1;
      tag2 = m->tag2;
      comm = m->comm;
      length = m->length;
    }
    p(tag1); p(tag2); p(comm); p(length);
    if(p.isUnpacking()) {
      m = new (&length, 0) AmpiMsg(tag1, tag2, length, comm);
    }
    p(m->data, length);
    if(p.isDeleting()) {
      delete m;
      m = 0;
    }
    return m;
  }
};

#define AMPI_MAXUDATA 20

class ampi : public ArrayElement1D {
    char str[128];
  protected:
    void prepareCtv(void);
  public: // entry methods
    ampi(AmpiStartMsg *);
    ampi(CkMigrateMessage *msg) {}
    ~ampi();
    void run(ArgsInfo *);
    void run(void);
    void generic(AmpiMsg *);
    void migrate(void)
    {
      AtSync();
    }
    void saveState(void)
    {
      FILE *fp = fopen(str, "wb");
      if(fp!=0) {
        PUP::toDisk p(fp); p.becomeUserlevel();
        pup(p);
      } else {
        CkError("Cannot checkpoint to file %s! Continuing...\n");
      }
      if(cthread_id) {
        CthAwaken(cthread_id);
        cthread_id = 0;
      }
      return;
    }
    void checkpoint(DirMsg *msg);
    void restart(DirMsg *);
    void restartThread(char *dname)
    {
      sprintf(str, "%s/%d/%d.cpt", dname, commidx, thisIndex);
      FILE *fp = fopen(str, "rb");
      if(fp!=0) {
        PUP::fromDisk p(fp); p.becomeUserlevel();
        pup(p);
        if(cthread_id) {
          CthAwaken(cthread_id);
          cthread_id = 0;
        }
      } else {
        CkAbort("Canot open restart file for reading!\n");
      }
      return;
    }
    void start_running(void)
    {
      thisArray->the_lbdb->ObjectStart(ldHandle);
    }
    void stop_running(void)
    {
      thisArray->the_lbdb->ObjectStop(ldHandle);
    }

  public: // to be used by AMPI_* functions
    void send(int t1, int t2, void* buf, int count, int type, int idx, int comm);
    static void sendraw(int t1, int t2, void* buf, int len, CkArrayID aid, 
                        int idx);
    void recv(int t1,int t2,void* buf,int count,int type,int comm,int *sts=0);
    void probe(int t1,int t2,int comm,int *sts);
    int iprobe(int t1,int t2,int comm,int *sts);
    void barrier(void);
    void bcast(int root, void* buf, int count, int type);
    static void bcastraw(void* buf, int len, CkArrayID aid);
    int register_userdata(void *, AMPI_PupFn);
    void *get_userdata(int);
  public:
    int commidx;
    CmmTable msgs;
    CthThread thread_id;
    CthThread mthread_id;
    CthThread cthread_id;
    int nbcasts;
    PersReq requests[100];
    int nrequests;
    PersReq irequests[100];
    int nirequests;
    int firstfree;
    DDT *myDDT ;
    int nudata;
    void *userdata[AMPI_MAXUDATA];
    AMPI_PupFn pup_ud[AMPI_MAXUDATA];

    virtual void pup(PUP::er &p);
    virtual void start(void); // should be overloaded in derived class
    void ResumeFromSync(void)
    {
      if (mthread_id)
      {
        CthAwaken(mthread_id);
        mthread_id = 0;
      }
    }
};

#endif
