#ifndef _FEM_H
#define _FEM_H

#include <stdio.h>
#include "fem.decl.h"

extern CkChareID _mainhandle;
extern CkArrayID _femaid;
extern unsigned int _nchunks;

// base types: keep in sync with femf.h
#define FEM_BYTE   0
#define FEM_INT    1
#define FEM_REAL   2
#define FEM_DOUBLE 3

// temporary Datatype representation
// will go away once MPI user-defined datatypes are ready
struct DType {
  int base_type;
  int vec_len;
  DType(void) {}
  DType(DType& dt) : base_type(dt.base_type), vec_len(dt.vec_len) {}
  DType(int b, int v) : base_type(b), vec_len(v) {}
  DType(int b) : base_type(b), vec_len(1) {}
  int length(int nitems=1) {
    int blen;
    switch(base_type) {
      case FEM_BYTE : blen = 1; break;
      case FEM_INT : blen = sizeof(int); break;
      case FEM_REAL : blen = sizeof(float); break;
      case FEM_DOUBLE : blen = sizeof(double); break;
    }
    return blen * vec_len * nitems;
  }
};

class DataMsg : public CMessage_DataMsg
{
 public:
  int from;
  int len;
  void *data;
  int tag;
  DataMsg(int t, int f, int l) : 
    tag(t), from(f), len(l) { data = (void*) (this+1); }
  DataMsg(void) { data = (void*) (this+1); }
  static void *pack(DataMsg *);
  static DataMsg *unpack(void *);
  static void *alloc(int, size_t, int*, int);
  void *getData(void)  { return data; }
};

class main : public Chare
{
  int numdone;
 public:
  main(CkArgMsg *);
  void done(void);
};

#define MAXDT 20

class chunk : public ArrayElement1D
{
  int numNodes; // number of local nodes
  int numElems; // number of local elements
  int numPes; // number of Pes I need to communicate with
  int *gNodeNums; // global node numbers for local nodes [numNodes]
  int numNodesPerElem; // number of Nodes per element *a constant*
  int *gElemNums; // global element numbers for local elements [numElems]
  int *conn; // elem->node connectivity, a 2-D array stored in row-order
  int *peNums; // which Pes I need to communicate with [numPes]
  int *numNodesPerPe; // How many Pes I need to communicate [numPes]
  int **nodesPerPe; // Pes->nodes map
  int *gPeToIdx; // which local index does the global PeNum map to [TotalPes]

  DType dtypes[MAXDT];
  int nsize; // size of the app's node type, separation between values
  int ntypes;

  CmmTable messages; // messages to be processed
  int wait_for; // which tag is tid waiting for ? 0 if not waiting
  CthThread tid; // waiting thread, 0 if no one is waiting

  int seqnum; // sequence number for update operation
  int nRecd; // number of messages received for this seqnum
  void *curnodes; // data addr for current update operation
  int curfid; // field descriptor for current update operation

 public:
  chunk(void);
  chunk(CkMigrateMessage *) {}
  void run(void);
  void recv(DataMsg *);
  int new_DT(int base_type, int vec_len) {
    if(ntypes==MAXDT) {
      CkAbort("FEM: registered datatypes limit exceeded.");
    }
    dtypes[ntypes] = DType(base_type, vec_len);
    ntypes++;
    return ntypes-1;
  }
  void update(int fid, void *nodes);
  void reduce(int fid, void *nodes, void *outbuf);
  void set_node_size(int n) { nsize = n; }
 private:
  void update_field(DataMsg *);
  void send(int fid, void *nodes);
  void readNodes(FILE*);
  void readElems(FILE*);
  void readComm(FILE*);
  void readChunk(void);
};

void FEM_Done(void);
void FEM_Set_Node_Size(int nsize);
int FEM_Create_Field(int base_type, int vec_len);
void FEM_Update_Field(int fid, void *nodes);
void FEM_Reduce_Field(int fid, void *nodes, void *outbuf);

// Fortran Bindings

extern "C" void fem_set_node_size_(int *nsize);
extern "C" int fem_create_field_(int *bt, int *vl);
extern "C" void fem_update_field_(int *fid, void *nodes);
extern "C" void fem_reduce_field_(int *fid, void *nodes, void *outbuf);

// Utility functions for Fortran

extern "C" int offsetof_(void *, void *);

// to be provided by the application
extern "C" void init_(void);
extern "C" void driver_(int *, int *, int *, int *, int *);
extern "C" void finalize_(void);

#endif
