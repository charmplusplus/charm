#include "fem.h"

CkChareID _mainhandle;
CkArrayID _femaid;
unsigned int _nchunks;

CtvStaticDeclare(chunk*, _femptr);

void *
DataMsg::pack(DataMsg *in)
{
  return (void*) in;
}

DataMsg *
DataMsg::unpack(void *in)
{
  return new (in) DataMsg;
}

void *
DataMsg::alloc(int mnum, size_t size, int *sizes, int pbits)
{
  return CkAllocMsg(mnum, size+sizes[0], pbits);
}

#define PACK(buf,sz) do { memcpy(pos,(buf),(sz)); pos += (sz); } while(0)

void *
ChunkMsg::pack(ChunkMsg *in)
{
  int totalsize = sizeof(ChunkMsg);
  totalsize += in->nnodes*sizeof(int); // gNodeNums
  totalsize += in->nnodes*sizeof(int); // primaryPart
  totalsize += in->nelems*sizeof(int); // gElemNums
  totalsize += in->nelems*in->nconn*sizeof(int); // conn
  totalsize += in->npes*sizeof(int); // peNums
  totalsize += in->npes*sizeof(int); // numNodesPerPe
  int tnodes=0;
  int i;
  for(i=0;i<(in->npes);i++) { tnodes += in->numNodesPerPe[i]; }
  totalsize += tnodes*sizeof(int); // nodesPerPe
  void *retmsg = CkAllocBuffer(in, totalsize); CHK(retmsg);
  char *pos = (char *) retmsg;
  PACK(in, sizeof(ChunkMsg));
  PACK(in->gNodeNums, in->nnodes*sizeof(int));
  PACK(in->primaryPart, in->nnodes*sizeof(int));
  PACK(in->gElemNums, in->nelems*sizeof(int));
  PACK(in->conn, in->nelems*in->nconn*sizeof(int));
  PACK(in->peNums, in->npes*sizeof(int));
  PACK(in->numNodesPerPe, in->npes*sizeof(int));
  PACK(in->nodesPerPe, tnodes*sizeof(int));
  delete[] in->gNodeNums;
  delete[] in->primaryPart;
  delete[] in->gElemNums;
  delete[] in->conn;
  delete[] in->peNums;
  delete[] in->numNodesPerPe;
  delete[] in->nodesPerPe;
  delete in;

  return retmsg;
}

#define UNPACK(buf,sz) do { buf = (int *) pos; pos += (sz); } while(0)

ChunkMsg *
ChunkMsg::unpack(void *in)
{
  ChunkMsg* msg = new (in) ChunkMsg;
  char *pos = (char *) in + sizeof(ChunkMsg);
  UNPACK(msg->gNodeNums,msg->nnodes*sizeof(int));
  UNPACK(msg->primaryPart,msg->nnodes*sizeof(int));
  UNPACK(msg->gElemNums,msg->nelems*sizeof(int));
  UNPACK(msg->conn,msg->nelems*msg->nconn*sizeof(int));
  UNPACK(msg->peNums, msg->npes*sizeof(int));
  UNPACK(msg->numNodesPerPe, msg->npes*sizeof(int));
  msg->nodesPerPe = (int*) pos;

  return msg;
}

static void 
_allReduceHandler(void *, int datasize, void *data)
{
  // the reduction handler is called on processor 0
  // with available reduction results
  DataMsg *dmsg = new (&datasize, 0) DataMsg(0,datasize,0); CHK(dmsg);
  memcpy(dmsg->data, data, datasize);
  CProxy_chunk carr(_femaid);
  // broadcast the reduction results to all array elements
  carr.result(dmsg);
}

static main* _mainptr = 0;

main::main(CkArgMsg *am)
{
  int i;
  _nchunks = CkNumPes();
  for(i=1;i<am->argc;i++) {
    if(strncmp(am->argv[i], "+vp", 3) == 0) {
      if (strlen(am->argv[i]) > 3) {
        sscanf(am->argv[i], "+vp%d", &_nchunks);
      } else {
        if (am->argv[i+1]) {
          sscanf(am->argv[i+1], "%d", &_nchunks);
        }
      }
      break;
    }
  }
  _femaid = CProxy_chunk::ckNew(_nchunks);
  CProxy_chunk farray(_femaid);
  farray.setReductionClient(_allReduceHandler, 0);
  _mainhandle = thishandle;
  numdone = 0;
  isMeshSet = 0;
  // call application-specific initialization
  _mainptr = this;
#if FEM_FORTRAN
#if CMK_FORTRAN_USES_ALLCAPS
  INIT();
#else // fortran-uses-trailing-undescore
  init_();
#endif
#else // C/C++
  init();
#endif // Fortran
  _mainptr = 0;
  if(!isMeshSet) {
    for(i=0;i<_nchunks;i++) {
      farray[i].run();
    }
  }
  delete am;
}

static void c2r(int nx, int ny, int *mat)
{
  int i, j;
  int *newmat =  new int[nx*ny]; CHK(newmat);
  for(i=0;i<nx;i++)
    for(j=0;j<ny;j++)
      newmat[i*ny+j] = mat[j*nx+i];
  for(i=0;i<(nx*ny);i++) mat[i] = newmat[i];
  delete[] newmat;
}

static void r2c(int nx, int ny, int *mat)
{
  int i, j;
  int *newmat =  new int[nx*ny]; CHK(newmat);
  for(i=0;i<nx;i++)
    for(j=0;j<ny;j++)
      newmat[j*nx+i] = mat[i*ny+j];
  for(i=0;i<(nx*ny);i++) mat[i] = newmat[i];
  delete[] newmat;
}

extern "C" void
METIS_PartMeshNodal(int*,int*,int*,int*,int*,int*,int*,int*,int*);

extern void fem_map(int nelem, int nnodes, int esize, int *connmat,
                    int nparts, int *epart, ChunkMsg *msgs[]);
void
main::setMesh(int nelem, int nnodes, int ctype, int *connmat, int *xconn)
{
  int *epart = new int[nelem]; CHK(epart);
  int esize = (ctype==FEM_TRIANGULAR) ? 3:
              (ctype==FEM_HEXAHEDRAL) ? 8:
	      4;
  int numflag, ecut;
#if FEM_FORTRAN
  numflag = 1;
#else
  numflag = 0;
#endif
  if(_nchunks==1) { // Metis cannot handle this case
    int i;
    for(i=0;i<nelem;i++) { epart[i] = numflag; }
    ecut = 0;
  } else {
    int *npart = new int[nnodes]; CHK(npart);
    // pass mesh to metis to be partitioned. xconn is always in row order
    METIS_PartMeshNodal(&nelem, &nnodes, xconn, &ctype, &numflag, 
                       (int*)&_nchunks, &ecut, epart, npart);
    delete[] npart;
  }
  // call the map function to compute communication info needed by the framework
  ChunkMsg **msgs = new ChunkMsg*[_nchunks]; CHK(msgs);
  fem_map(nelem, nnodes, esize, connmat, _nchunks, epart, msgs);
  delete[] epart;
  // send messages to individual chunks with these partitions
  isMeshSet = 1;
  CProxy_chunk farray(_femaid);
  // each chunk is sent a message containing its meshdata
  for(int i=0;i<_nchunks;i++) {
    farray[i].run(msgs[i]);
  }
  delete[] msgs;
}

extern "C" void 
FEM_Set_Mesh(int nelem, int nnodes, int ctype, int *connmat)
{
  if(_mainptr == 0) {
    CkAbort("FEM_Set_Mesh can be called from within _init only.\n");
  }
#if FEM_FORTRAN
  int esize = (ctype==FEM_TRIANGULAR) ? 3:
              (ctype==FEM_HEXAHEDRAL) ? 8:
	      4;
  // make the connmat row major
  c2r(nelem, esize, connmat);
#endif
  _mainptr->setMesh(nelem, nnodes, ctype, connmat, connmat);
#if FEM_FORTRAN
  // make the connmat column major
  r2c(nelem, esize, connmat);
#endif
}

extern "C" void 
FEM_Set_Mesh_Transform(int nelem, int nnodes, int ctype, int *connmat,
                       int *permute)
{
  if(_mainptr == 0) {
    CkAbort("FEM_Set_Mesh_Transform can be called from within _init only.\n");
  }
  int esize = (ctype==FEM_TRIANGULAR) ? 3:
              (ctype==FEM_HEXAHEDRAL) ? 8:
	      4;
#if FEM_FORTRAN
  // make the connmat row major
  c2r(nelem, esize, connmat);
#endif
  // check if permute array is indeed a permutation
  // it is a permutation if the mapped indices are in the range 0..esize-1
  // and if every index occurs just once.
  int i, j;
  int *freq = new int[esize]; CHK(freq);
  for(i=0;i<esize;i++) { freq[i] = 0; }
#if FEM_FORTRAN
  for(i=0;i<esize;i++) {
    if(permute[i] < 1 || permute[i] > esize) {
      CkAbort("FEM> transform vector not a permutation.\n");
    }
  }
  for(i=0;i<esize;i++) { freq[permute[i]-1]++; }
#else
  for(i=0;i<esize;i++) {
    if(permute[i] < 0 || permute[i] >= esize) {
      CkAbort("FEM> transform vector not a permutation.\n");
    }
  }
  for(i=0;i<esize;i++) { freq[permute[i]]++; }
#endif
  for(i=0;i<esize;i++) {
    if(freq[i]!=1) {
      CkAbort("FEM> transform vector not a permutation.\n");
    }
  }
  delete[] freq;
  // allocate conn, and transform
  int *conn = new int[nelem*esize]; CHK(conn);
  for(i=0;i<nelem;i++) {
    for(j=0;j<esize;j++) {
      conn[i*esize+j] = connmat[i*esize+permute[j]];
    }
  }
  _mainptr->setMesh(nelem, nnodes, ctype, connmat, conn);
#if FEM_FORTRAN
  // make the connmat column major
  r2c(nelem, esize, connmat);
#endif
  delete[] conn;
}

void
main::done(void)
{
  numdone++;
  if (numdone == _nchunks) {
    // call application-specific finalization
    _mainptr = this;
#if FEM_FORTRAN
#if CMK_FORTRAN_USES_ALLCAPS
    FINALIZE();
#else // fortran-uses-trailing-undescore
    finalize_();
#endif
#else // C/C++
    finalize();
#endif // Fortran
    _mainptr = 0;
    CkExit();
  }
}

template<class d>
void sum(const int len, d* lhs, d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs++ += *rhs++;
  }
}

template<class d>
void max(const int len, d* lhs, d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = (*lhs > *rhs) ? *lhs : *rhs;
    lhs++; rhs++;
  }
}

template<class d>
void min(const int len, d* lhs, d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = (*lhs < *rhs) ? *lhs : *rhs;
    lhs++; rhs++;
  }
}

static inline void
combine(const DType& dt, void* lhs, void* rhs, int op)
{
  switch(op) {
    case FEM_SUM:
      switch(dt.base_type) {
        case FEM_BYTE : 
          sum(dt.vec_len,(unsigned char*)lhs, (unsigned char*)rhs); 
          break;
        case FEM_INT : sum(dt.vec_len,(int*)lhs, (int*) rhs); break;
        case FEM_REAL : sum(dt.vec_len,(float*)lhs, (float*) rhs); break;
        case FEM_DOUBLE : sum(dt.vec_len,(double*)lhs, (double*) rhs); break;
      }
      break;
    case FEM_MAX:
      switch(dt.base_type) {
        case FEM_BYTE : 
          max(dt.vec_len,(unsigned char*)lhs, (unsigned char*)rhs); 
          break;
        case FEM_INT : max(dt.vec_len,(int*)lhs, (int*) rhs); break;
        case FEM_REAL : max(dt.vec_len,(float*)lhs, (float*) rhs); break;
        case FEM_DOUBLE : max(dt.vec_len,(double*)lhs, (double*) rhs); break;
      }
      break;
    case FEM_MIN:
      switch(dt.base_type) {
        case FEM_BYTE : 
          min(dt.vec_len,(unsigned char*)lhs, (unsigned char*)rhs); 
          break;
        case FEM_INT : min(dt.vec_len,(int*)lhs, (int*) rhs); break;
        case FEM_REAL : min(dt.vec_len,(float*)lhs, (float*) rhs); break;
        case FEM_DOUBLE : min(dt.vec_len,(double*)lhs, (double*) rhs); break;
      }
      break;
  }
}

chunk::chunk(void)
{
  ntypes = 0;
  new_DT(FEM_BYTE);
  new_DT(FEM_INT);
  new_DT(FEM_REAL);
  new_DT(FEM_DOUBLE);

  messages = CmmNew();
  seqnum = 1;
  wait_for = 0;
  tid = 0;
}

void
chunk::callDriver(void)
{
  // call the application-specific driver
  doneCalled = 0;
#if FEM_FORTRAN
  r2c(numElems, numNodesPerElem, conn);
#if CMK_FORTRAN_USES_ALLCAPS
  DRIVER(&numNodes, gNodeNums, &numElems, gElemNums, &numNodesPerElem, conn);
#else // fortran-uses-trailing-undescore
  driver_(&numNodes, gNodeNums, &numElems, gElemNums, &numNodesPerElem, conn);
#endif
#else // C/C++
  driver(numNodes, gNodeNums, numElems, gElemNums, numNodesPerElem, conn);
#endif // Fortran
  FEM_Done();
}

void
chunk::run(ChunkMsg *msg)
{
  CtvInitialize(chunk*, _femptr);
  CtvAccess(_femptr) = this;
  readChunk(msg);
  callDriver();
}

void
chunk::run(void)
{
  CtvInitialize(chunk*, _femptr);
  CtvAccess(_femptr) = this;
  readChunk();
  callDriver();
}

void
chunk::recv(DataMsg *dm)
{
  if (dm->tag == wait_for) {
    update_field(dm); // update the appropriate field value
    delete dm;
    nRecd++;
    if(nRecd==numPes) {
      wait_for = 0; // done waiting for seqnum
      CthAwaken(tid); // awaken the waiting thread
      tid = 0;
    }
  } else {
    CmmPut(messages, 1, &(dm->tag), dm);
  }
}

void
chunk::send(int fid, void *nodes)
{
  int i, j;
  for(i=0;i<numPes;i++) {
    int dest = peNums[i];
    int num = numNodesPerPe[i];
    int len = dtypes[fid].length(num);
    DataMsg *msg = new (&len, 0) DataMsg(seqnum, thisIndex, fid); CHK(msg);
    len = dtypes[fid].length();
    void *data = msg->data;
    void *src = (void *) ((char *)nodes + dtypes[fid].init_offset);
    for(j=0;j<num;j++) {
      src = (void *)((char*)nodes+(nodesPerPe[i][j]*dtypes[fid].distance));
      memcpy(data, src, len);
      data = (void*) ((char*)data + len);
    }
    CProxy_chunk cp(thisArrayID);
    cp[dest].recv(msg);
  }
}

void
chunk::update(int fid, void *nodes)
{
  // first send my field values to all the processors that need it
  seqnum++;
  send(fid, nodes);
  curbuf = nodes;
  nRecd = 0;
  // now, if any of the field values have been received already,
  // process them
  DataMsg *dm;
  while (dm = (DataMsg*)CmmGet(messages, 1, &seqnum, 0)) {
    update_field(dm);
    delete dm;
    nRecd++;
  }
  // if any field values are still required, put myself to sleep
  if (nRecd != numPes) {
    wait_for = seqnum;
    tid = CthSelf();
    CthSuspend();
    wait_for = 0;
    tid = 0;
  }
}

void
chunk::update_field(DataMsg *msg)
{
  void *data = msg->data;
  int from = gPeToIdx[msg->from];
  int nnodes = numNodesPerPe[from];
  int i;
  for(i=0;i<nnodes;i++) {
    int cnum = nodesPerPe[from][i];
    void *cnode = (void*) ((char*)curbuf+cnum*dtypes[msg->dtype].distance);
    combine(dtypes[msg->dtype], cnode, data, FEM_SUM);
    data = (void *)((char*)data+(dtypes[msg->dtype].length()));
  }
}

void
chunk::reduce_field(int fid, void *nodes, void *outbuf, int op)
{
  // first reduce over local nodes
  DType *dt = &dtypes[fid];
  void *src = (void *) ((char *) nodes + dt->init_offset);
  for(int i=0; i<numNodes; i++) {
    if(isPrimary[i]) {
      combine(*dt, outbuf, src, op);
    }
    src = (void *)((char *)src + dt->distance);
  }
  // and now reduce over partitions
  reduce(fid, outbuf, outbuf, op);
}

void
chunk::reduce(int fid, void *inbuf, void *outbuf, int op)
{
  int len = dtypes[fid].length();
  if(numElements==1) {
    memcpy(outbuf,inbuf,len);
    return;
  }
  CkReduction::reducerType rtype;
  switch(op) {
    case FEM_SUM:
      switch(dtypes[fid].base_type) {
        case FEM_INT: rtype = CkReduction::sum_int; break;
        case FEM_REAL: rtype = CkReduction::sum_float; break;
        case FEM_DOUBLE: rtype = CkReduction::sum_double; break;
      }
      break;
    case FEM_MAX:
      switch(dtypes[fid].base_type) {
        case FEM_INT: rtype = CkReduction::max_int; break;
        case FEM_REAL: rtype = CkReduction::max_float; break;
        case FEM_DOUBLE: rtype = CkReduction::max_double; break;
      }
      break;
    case FEM_MIN:
      switch(dtypes[fid].base_type) {
        case FEM_INT: rtype = CkReduction::min_int; break;
        case FEM_REAL: rtype = CkReduction::min_float; break;
        case FEM_DOUBLE: rtype = CkReduction::min_double; break;
      }
      break;
  }
  contribute(len, inbuf, rtype);
  curbuf = outbuf;
  tid = CthSelf();
  CthSuspend();
}

void
chunk::result(DataMsg *msg)
{
  //msg->from used as length
  memcpy(curbuf, msg->data, msg->from);
  CthAwaken(tid);
  tid = 0;
  delete msg;
}

void
chunk::readField(int fid, void *nodes, char *fname)
{
  int btype = dtypes[fid].base_type;
  int typelen = dtypes[fid].vec_len;
  int btypelen = dtypes[fid].length()/typelen;
  char *data = (char *)nodes + dtypes[fid].init_offset;
  int distance = dtypes[fid].distance;
  fp = fopen(fname, "r");
  if(fp==0) {
    CkError("Cannot open file %s for reading.\n", fname);
    CkAbort("Exiting");
  }
  char str[80];
  char* pos;
  const char* fmt;
  int i, j, curline;
#if FEM_FORTRAN
  curline = 1;
#else
  curline = 0;
#endif
  switch(btype) {
    case FEM_INT: fmt = "%d%n"; break;
    case FEM_REAL: fmt = "%f%n"; break;
    case FEM_DOUBLE: fmt = "%lf%n"; break;
  }
  for(i=0;i<numNodes;i++) {
    // skip lines to the next local node
    for(j=curline;j<gNodeNums[i];j++)
      fgets(str,80,fp);
    curline = gNodeNums[i]+1;
    fgets(str,80,fp);
    int curnode, numchars;
    sscanf(str,"%d%n",&curnode,&numchars);
    pos = str + numchars;
    if(curnode != gNodeNums[i]) {
      CkError("Expecting info for node %d, got %d\n", gNodeNums[i], curnode);
      CkAbort("Exiting");
    }
    for(j=0;j<typelen;j++) {
      sscanf(pos, fmt, data+(j*btypelen), &numchars);
      pos += numchars;
    }
    data += distance;
  }
}

void
chunk::readNodes(ChunkMsg *msg)
{
  if(msg==0) {
    fscanf(fp, "%d", &numNodes);
    gNodeNums = new int[numNodes]; CHK(gNodeNums);
    isPrimary = new int[numNodes]; CHK(isPrimary);
    for(int i=0;i<numNodes;i++) {
      fscanf(fp, "%d%d", &gNodeNums[i], &isPrimary[i]);
      isPrimary[i] = ((isPrimary[i]==thisIndex) ? 1 : 0);
    }
  } else {
    numNodes = msg->nnodes;
    gNodeNums = new int[numNodes]; CHK(gNodeNums);
    isPrimary = new int[numNodes]; CHK(isPrimary);
    for(int i=0;i<numNodes;i++) {
      gNodeNums[i] = msg->gNodeNums[i];
      isPrimary[i] = ((msg->primaryPart[i]==thisIndex) ? 1 : 0);
    }
  }
}

void
chunk::readElems(ChunkMsg *msg)
{
  if(msg==0) {
    fscanf(fp, "%d%d", &numElems, &numNodesPerElem);
    gElemNums = new int[numElems]; CHK(gElemNums);
    conn = new int[numElems*numNodesPerElem]; CHK(conn);

    for(int i=0; i<numElems; i++) {
      fscanf(fp, "%d", &gElemNums[i]);
      for(int j=0;j<numNodesPerElem;j++) {
        fscanf(fp, "%d", &conn[i*numNodesPerElem+j]);
      }
    }
  } else {
    numElems = msg->nelems;
    numNodesPerElem = msg->nconn;
    gElemNums = new int[numElems]; CHK(gElemNums);
    conn = new int[numElems*numNodesPerElem]; CHK(conn);
    for(int i=0; i<numElems; i++) {
      gElemNums[i] = msg->gElemNums[i];
      for(int j=0;j<numNodesPerElem;j++) {
        conn[i*numNodesPerElem+j] = msg->conn[i*numNodesPerElem+j];
      }
    }
  }
}

void
chunk::readComm(ChunkMsg *msg)
{
  gPeToIdx = new int[numElements]; CHK(gPeToIdx);
  for(int p=0;p<numElements;p++) {
    gPeToIdx[p] = (-1);
  }
  if(msg==0) {
    fscanf(fp, "%d", &numPes);
    peNums = new int[numPes]; CHK(peNums);
    numNodesPerPe = new int[numPes]; CHK(numNodesPerPe);
    nodesPerPe = new int*[numPes]; CHK(nodesPerPe);
    for(int i=0;i<numPes;i++) {
      fscanf(fp, "%d%d", &peNums[i], &numNodesPerPe[i]);
      gPeToIdx[peNums[i]] = i;
      nodesPerPe[i] = new int[numNodesPerPe[i]]; CHK(nodesPerPe[i]);
      for(int j=0;j<numNodesPerPe[i];j++) {
        fscanf(fp, "%d", &nodesPerPe[i][j]);
      }
    }
  } else {
    numPes = msg->npes;
    peNums = new int[numPes]; CHK(peNums);
    numNodesPerPe = new int[numPes]; CHK(numNodesPerPe);
    nodesPerPe = new int*[numPes]; CHK(nodesPerPe);
    int k = 0;
    for(int i=0;i<numPes;i++) {
      peNums[i] = msg->peNums[i];
      numNodesPerPe[i] = msg->numNodesPerPe[i];
      gPeToIdx[peNums[i]] = i;
      nodesPerPe[i] = new int[numNodesPerPe[i]]; CHK(nodesPerPe[i]);
      for(int j=0;j<numNodesPerPe[i];j++) {
        nodesPerPe[i][j] = msg->nodesPerPe[k++];
      }
    }
  }
}

void
chunk::readChunk(ChunkMsg *msg)
{
  if(msg==0) {
    char fname[32];
    sprintf(fname, "meshdata.Pe%d", thisIndex);
    fp = fopen(fname, "r");
    if(fp==0) {
      CkAbort("FEM: unable to open input file.\n");
    }
    readNodes();
    readElems();
    readComm();
    fclose(fp);
  } else {
    readNodes(msg);
    readElems(msg);
    readComm(msg);
  }
}

void
chunk::print(void)
{
  // FIXME: str will eventually overflow. replace it by xstr
  char str[1024];
  char tmpstr[80];
  CkPrintf("[%d] Number of Elements = %d\n", thisIndex, numElems);
  CkPrintf("[%d] Number of Nodes = %d\n", thisIndex, numNodes);
  CkPrintf("[%d] Number of Comms = %d\n", thisIndex, numPes);
  int i, j;
  sprintf(str, "[%d] List of Elements:\n", thisIndex);
  for(i=0;i<numElems;i++) {
    sprintf(tmpstr, "  %d\n", gElemNums[i]);
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
  sprintf(str, "[%d] List of Nodes: (num, prim)\n", thisIndex);
  for(i=0;i<numNodes;i++) {
    sprintf(tmpstr, "  %d  %d\n", gNodeNums[i],isPrimary[i]);
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
  sprintf(str, "[%d] Connectivity:\n", thisIndex);
  for(i=0;i<numElems;i++) {
    sprintf(tmpstr, "  [%d] ", gElemNums[i]);
    strcat(str,tmpstr);
    for(j=0;j<numNodesPerElem;j++) {
#if FEM_FORTRAN
      sprintf(tmpstr, "%d ", gNodeNums[conn[j*numElems+i]-1]);
#else
      sprintf(tmpstr, "%d ", gNodeNums[conn[i*numNodesPerElem+j]]);
#endif
      strcat(str,tmpstr);
    }
    sprintf(tmpstr, "\n");
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
  sprintf(str, "[%d] CommInfo: (penum, numnodes)\n", thisIndex);
  for(i=0;i<numPes;i++) {
    sprintf(tmpstr, "  [%d]  %d\n    ", peNums[i], numNodesPerPe[i]);
    strcat(str,tmpstr);
    for(j=0;j<numNodesPerPe[i];j++) {
      sprintf(tmpstr, "%d ", gNodeNums[nodesPerPe[i][j]]);
      strcat(str,tmpstr);
    }
    sprintf(tmpstr, "\n");
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
}

extern "C" void 
FEM_Done(void)
{
  chunk *cptr = CtvAccess(_femptr);
  if(!cptr->doneCalled) {
    CProxy_main mainproxy(_mainhandle);
    mainproxy.done();
    cptr->doneCalled = 1;
  }
}

extern "C" int 
FEM_Create_Field(int base_type, int vec_len, int init_offset, int distance)
{
  chunk *cptr = CtvAccess(_femptr);
  return cptr->new_DT(base_type, vec_len, init_offset, distance);
}

extern "C" void
FEM_Update_Field(int fid, void *nodes)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->update(fid, nodes);
}

extern "C" void
FEM_Reduce_Field(int fid, void *nodes, void *outbuf, int op)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->reduce_field(fid, nodes, outbuf, op);
}

extern "C" void
FEM_Reduce(int fid, void *inbuf, void *outbuf, int op)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->reduce(fid, inbuf, outbuf, op);
}

extern "C" void
FEM_Read_Field(int fid, void *nodes, char *fname)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->readField(fid, nodes, fname);
}

extern "C" int
FEM_My_Partition(void)
{
  chunk *cptr = CtvAccess(_femptr);
  return cptr->id();
}

extern "C" int
FEM_Num_Partitions(void)
{
  return _nchunks;
}

extern "C" void 
FEM_Print(char *str)
{
  if(_mainptr) {
    CkPrintf("%s\n", str);
  } else {
    chunk *cptr = CtvAccess(_femptr);
    CkPrintf("[%d] %s\n", cptr->thisIndex, str);
  }
}

extern "C" void 
FEM_Print_Partition(void)
{
  chunk *cptr = CtvAccess(_femptr);
  cptr->print();
}

// Fortran Bindings
#if FEM_FORTRAN
extern "C" int
#if CMK_FORTRAN_USES_ALLCAPS
FEM_CREATE_FIELD
#else
fem_create_field_
#endif
  (int *bt, int *vl, int *io, int *d)
{
  return FEM_Create_Field(*bt, *vl, *io, *d);
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_UPDATE_FIELD
#else
fem_update_field_
#endif
  (int *fid, void *nodes)
{
  FEM_Update_Field(*fid, nodes);
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_REDUCE_FIELD
#else
fem_reduce_field_
#endif
  (int *fid, void *nodes, void *outbuf, int *op)
{
  FEM_Reduce_Field(*fid, nodes, outbuf, *op);
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_REDUCE
#else
fem_reduce_
#endif
  (int *fid, void *inbuf, void *outbuf, int *op)
{
  FEM_Reduce(*fid, inbuf, outbuf, *op);
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_READ_FIELD
#else
fem_read_field_
#endif
  (int *fid, void *nodes, char *fname, int len)
{
  char *tmp = new char[len+1]; CHK(tmp);
  memcpy(tmp, fname, len);
  tmp[len] = '\0';
  FEM_Read_Field(*fid, nodes, tmp);
  delete[] tmp;
}

extern "C" int
#if CMK_FORTRAN_USES_ALLCAPS
FEM_MY_PARTITION
#else
fem_my_partition_
#endif
  (void)
{
  return FEM_My_Partition();
}

extern "C" int
#if CMK_FORTRAN_USES_ALLCAPS
FEM_NUM_PARTITIONS
#else
fem_num_partitions_
#endif
  (void)
{
  return FEM_Num_Partitions();
}

// Utility functions for Fortran

extern "C" int
#if CMK_FORTRAN_USES_ALLCAPS
OFFSETOF
#else
offsetof_
#endif
  (void *first, void *second)
{
  return (int)((char *)second - (char*)first);
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
FEM_PRINT
#else
fem_print_
#endif
  (char *str, int len)
{
  char *tmpstr = new char[len+1]; CHK(tmpstr);
  memcpy(tmpstr,str,len);
  tmpstr[len] = '\0';
  FEM_Print(tmpstr);
  delete[] tmpstr;
}

extern "C" void 
#if CMK_FORTRAN_USES_ALLCAPS
FEM_PRINT_PARTITION
#else
fem_print_partition_
#endif
  (void)
{
  chunk *ptr = CtvAccess(_femptr);
  ptr->print();
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_SET_MESH
#else
fem_set_mesh_
#endif
(int *nelem, int *nnodes, int *ctype, int *connmat)
{
  FEM_Set_Mesh(*nelem, *nnodes, *ctype, connmat);
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_SET_MESH_TRANSFORM
#else
fem_set_mesh_transform_
#endif
(int *nelem, int *nnodes, int *ctype, int *connmat, int *permute)
{
  FEM_Set_Mesh_Transform(*nelem, *nnodes, *ctype, connmat, permute);
}

extern "C" void
#if CMK_FORTRAN_USES_ALLCAPS
FEM_DONE
#else
fem_done_
#endif
  (void)
{
  FEM_Done();
}


#endif

#include "fem.def.h"
