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

// reduction operations: keep in synch with femf.h
#define FEM_SUM 0
#define FEM_MAX 1
#define FEM_MIN 2

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
  int dtype;
  void *data;
  int tag;
  DataMsg(int t, int f, int d) : 
    tag(t), from(f), dtype(d) { data = (void*) (this+1); }
  DataMsg(void) { data = (void*) (this+1); }
  static void *pack(DataMsg *);
  static DataMsg *unpack(void *);
  static void *alloc(int, size_t, int*, int);
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
  int *isPrimary; // 1 true if primary (i.e. owner of a shared node)
  int numNodesPerElem; // number of Nodes per element *a constant*
  int *gElemNums; // global element numbers for local elements [numElems]
  int *conn; // elem->node connectivity, a 2-D array stored in row-order
  int *peNums; // which Pes I need to communicate with [numPes]
  int *numNodesPerPe; // How many Pes I need to communicate [numPes]
  int **nodesPerPe; // Pes->nodes map
  int *gPeToIdx; // which local index does the global PeNum map to [TotalPes]

  DType dtypes[MAXDT];
  int ntypes;

  CmmTable messages; // messages to be processed
  int wait_for; // which tag is tid waiting for ? 0 if not waiting
  CthThread tid; // waiting thread, 0 if no one is waiting

  int seqnum; // sequence number for update operation
  int nRecd; // number of messages received for this seqnum
  void *curbuf; // data addr for current update operation

 public:
  chunk(void);
  chunk(CkMigrateMessage *) {}
  void run(void);
  void recv(DataMsg *);
  void result(DataMsg *);
  int new_DT(int base_type, int vec_len=1, int init_offset=0, int distance=0) {
    if(ntypes==MAXDT) {
      CkAbort("FEM: registered datatypes limit exceeded.");
    }
    dtypes[ntypes] = DType(base_type, vec_len, init_offset, distance);
    ntypes++;
    return ntypes-1;
  }
  void update(int fid, void *nodes);
  void reduce_field(int fid, void *nodes, void *outbuf, int op);
  void reduce(int fid, void *inbuf, void *outbuf, int op);
  void readField(int fid, void *nodes, char *fname);
  int id(void) { return thisIndex; }
  int total(void) { return numElements; }
 private:
  void update_field(DataMsg *);
  void send(int fid, void *nodes);
  void readNodes(FILE*);
  void readElems(FILE*);
  void readComm(FILE*);
  void readChunk(void);
};

void FEM_Done(void);
int FEM_Create_Field(int base_type, int vec_len, int init_offset, int distance);
void FEM_Update_Field(int fid, void *nodes);
void FEM_Reduce_Field(int fid, void *nodes, void *outbuf, int op);
void FEM_Reduce(int fid, void *inbuf, void *outbuf, int op);
int FEM_My_Partition(void);
int FEM_Num_Partitions(void);
void FEM_Read_Field(int fid, void *nodes, char *fname);

// Fortran Bindings

extern "C" int fem_create_field_(int *bt, int *vl, int *io, int *d);
extern "C" void fem_update_field_(int *fid, void *nodes);
extern "C" void fem_reduce_field_(int *fid, void *nodes, void *outbuf, int *op);
extern "C" void fem_reduce_(int *fid, void *inbuf, void *outbuf, int *op);
extern "C" int fem_my_partition_(void);
extern "C" int fem_num_partitions_(void);
// FIXME: correct fortran-c interoperability issue for passing character arrays
extern "C" void fem_read_field_(int *fid, void *nodes, char *fname);

// Utility functions for Fortran

extern "C" int offsetof_(void *, void *);

// to be provided by the application
extern "C" void init_(void);
extern "C" void driver_(int *, int *, int *, int *, int *, int *);
extern "C" void finalize_(void);

#endif
