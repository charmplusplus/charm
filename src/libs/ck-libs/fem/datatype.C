/*Charm++ Finite Element Framework:
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

Implementation file for FEM: FEM user data types.
These allow FEM data to be scooped right out of a user's
data structure.
*/
#include "charm.h" /* for CkAbort */
#include "datatype.h"
#include <string.h> /* for memcpy */
#include <math.h>
#include <limits.h>
#include <float.h> /*for FLT_MIN on non-Suns*/

int DType::type_size(int dataType)
{
    switch(dataType) {
      case FEM_BYTE : return 1; break;
      case FEM_INT : return sizeof(int); break;
      case FEM_REAL : return sizeof(float); break;
      case FEM_DOUBLE : return sizeof(double); break;
      case FEM_INDEX_0: return sizeof(int); break;
      case FEM_INDEX_1: return sizeof(int); break;
      default: CkAbort("Unrecognized data type field passed to FEM framework!\n");
    }
    return -1;
}

const char *DType::type_name(int dataType) 
{
    switch(dataType) {
      case FEM_BYTE : return "FEM_BYTE"; break;
      case FEM_INT : return "FEM_INT"; break;
      case FEM_REAL : return "FEM_REAL"; break;
      case FEM_DOUBLE : return "FEM_DOUBLE"; break;
      case FEM_INDEX_0: return "FEM_INDEX_0"; break;
      case FEM_INDEX_1: return "FEM_INDEX_1"; break;
      default: CkAbort("Unrecognized data type field passed to FEM framework!\n");
    }
    return "unknown";
}

/******************** Reduction Support **********************/
typedef unsigned char byte;

template<class d>
void sumFn(const int len, d* lhs, const d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs++ += *rhs++;
  }
}

/*Several compilers "helpfully" define max and min-- confusing us completely*/
#undef max 
#undef min

template<class d>
void maxFn(const int len, d* lhs, const d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = (*lhs > *rhs) ? *lhs : *rhs;
    lhs++; rhs++;
  }
}

template<class d>
void minFn(const int len, d* lhs, const d* rhs)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = (*lhs < *rhs) ? *lhs : *rhs;
    lhs++; rhs++;
  }
}

template<class d>
void assignFn(const int len, d* lhs, d val)
{
  int i;
  for(i=0;i<len;i++) {
    *lhs = val;
  }
}

//Force template-fn instantiations (for compilers too stupid to figure them out)

//Instantiate this routine for these arguments:
#define force_routine(fnName,tempParam,args) \
	template void fnName<tempParam> args

//Instaniate all routines for this data type
#define force_instantiate(datatype) \
	force_routine(sumFn,datatype,(const int len,datatype *lhs,const datatype *rhs));\
	force_routine(minFn,datatype,(const int len,datatype *lhs,const datatype *rhs));\
	force_routine(maxFn,datatype,(const int len,datatype *lhs,const datatype *rhs));\
	force_routine(assignFn,datatype,(const int len,datatype *lhs,datatype val));

force_instantiate(byte);
force_instantiate(int);
force_instantiate(float);
force_instantiate(double);

void reduction_initialize(const DType& dt, void *lhs, int op)
{
  switch(op) {
    case FEM_SUM:
      switch(dt.base_type) {
        case FEM_BYTE : assignFn(dt.vec_len,(byte*)lhs, (byte)0); break;
        case FEM_INT : assignFn(dt.vec_len,(int*)lhs, 0); break;
        case FEM_REAL : assignFn(dt.vec_len,(float*)lhs, (float)0.0); break;
        case FEM_DOUBLE : assignFn(dt.vec_len,(double*)lhs, 0.0); break;
        default: CkAbort("Invalid data type passed to FEM reduction!");
      }
      break;
    case FEM_MAX:
      switch(dt.base_type) {
        case FEM_BYTE : assignFn(dt.vec_len,(byte*)lhs, (byte)CHAR_MIN); break;
        case FEM_INT : assignFn(dt.vec_len,(int*)lhs, INT_MIN); break;
        case FEM_REAL : assignFn(dt.vec_len,(float*)lhs, FLT_MIN); break;
        case FEM_DOUBLE : assignFn(dt.vec_len,(double*)lhs, DBL_MIN); break;
      }
      break;
    case FEM_MIN:
      switch(dt.base_type) {
        case FEM_BYTE : assignFn(dt.vec_len,(byte*)lhs, (byte)CHAR_MAX); break;
        case FEM_INT : assignFn(dt.vec_len,(int*)lhs, INT_MAX); break;
        case FEM_REAL : assignFn(dt.vec_len,(float*)lhs, FLT_MAX); break;
        case FEM_DOUBLE : assignFn(dt.vec_len,(double*)lhs, DBL_MAX); break;
      }
      break;
  }
}


typedef void (*combineFn_BYTE)(const int len,byte *lhs,const byte *rhs);
typedef void (*combineFn_INT)(const int len,int *lhs,const int *rhs);
typedef void (*combineFn_REAL)(const int len,float *lhs,const float *rhs);
typedef void (*combineFn_DOUBLE)(const int len,double *lhs,const double *rhs);

//This odd-looking define selects the appropriate templated type
    // of "fn", casts it to a void* type, and returns it.
#define combine_switch(fn) \
      switch(dt.base_type) {\
        case FEM_BYTE : return (reduction_combine_fn)(combineFn_BYTE)fn;\
        case FEM_INT : return (reduction_combine_fn)(combineFn_INT)fn;\
        case FEM_REAL : return (reduction_combine_fn)(combineFn_REAL)fn;\
        case FEM_DOUBLE : return (reduction_combine_fn)(combineFn_DOUBLE)fn;\
      }\
      break;

reduction_combine_fn reduction_combine(const DType& dt, int op)
{
  switch(op) {
    case FEM_SUM: combine_switch(sumFn);
    case FEM_MIN: combine_switch(minFn);
    case FEM_MAX: combine_switch(maxFn);
    default: CkAbort("Invalid reduction type passed to FEM!");
  }
  return NULL;
}


/************************************************
"Gather" routines extract data distributed (nodeIdx)
through the user's array (in) and collect it into a message (out).
 */
#define gather_args (int nVal,int valLen, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

static void gather_general gather_args
{
  for(int i=0;i<nVal;i++) {
      const void *src = (const void *)(in+nodeIdx[i]*nodeScale);
      memcpy(out, src, valLen);
      out +=valLen;
  }
}

#define gather_doubles(n,copy) \
static void gather_double##n gather_args \
{ \
  double *od=(double *)out; \
  for(int i=0;i<nVal;i++) { \
      const double *src = (const double *)(in+nodeIdx[i]*nodeScale); \
      copy \
      od+=n; \
  } \
}

gather_doubles(1,od[0]=src[0];)
gather_doubles(2,od[0]=src[0];od[1]=src[1];)
gather_doubles(3,od[0]=src[0];od[1]=src[1];od[2]=src[2];)


  /**
   * For each record in nodes[0..nNodes-1], copy the
   * user data in v_in into the compressed data in v_out.
   */
void DType::gather(int nNodes,const int *nodes,
		   const void *v_in,void *v_out) const
{
  const char *in=(const char *)v_in;
  char *out=(char *)v_out;
  in += init_offset;
  //Try for a more specialized version if possible:
  if (base_type == FEM_DOUBLE) {
      switch(vec_len) {
      case 1: gather_double1(nNodes,length(),nodes,fdistance,in,out); return;
      case 2: gather_double2(nNodes,length(),nodes,fdistance,in,out); return;
      case 3: gather_double3(nNodes,length(),nodes,fdistance,in,out); return;
      }
  }
  
  //Otherwise, use the general version
  gather_general(nNodes,length(),nodes,fdistance,in,out);
}

/************************************************
"Scatter" routines are the opposite of gather.
 */
#define scatter_args (int nVal,int valLen, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

static void scatter_general scatter_args
{
  for(int i=0;i<nVal;i++) {
      void *dest = (void *)(out+nodeIdx[i]*nodeScale);
      memcpy(dest,in, valLen);
      in +=valLen;
  }
}

#define scatter_doubles(n,copy) \
static void scatter_double##n scatter_args \
{ \
  const double *src=(const double *)in; \
  for(int i=0;i<nVal;i++) { \
      double *od = (double *)(out+nodeIdx[i]*nodeScale); \
      copy \
      src+=n; \
  } \
}

scatter_doubles(1,od[0]=src[0];)
scatter_doubles(2,od[0]=src[0];od[1]=src[1];)
scatter_doubles(3,od[0]=src[0];od[1]=src[1];od[2]=src[2];)

  /**
   * For each field in the list nodes[0..nNodes-1], copy the
   * compressed data from v_in into the user data in v_out.
   */
void DType::scatter(int nNodes,const int *nodes,
		   const void *v_in,void *v_out) const
{
  const char *in=(const char *)v_in;
  char *out=(char *)v_out;
  out += init_offset;
  //Try for a more specialized version if possible:
  if (base_type == FEM_DOUBLE) {
      switch(vec_len) {
      case 1: scatter_double1(nNodes,length(),nodes,fdistance,in,out); return;
      case 2: scatter_double2(nNodes,length(),nodes,fdistance,in,out); return;
      case 3: scatter_double3(nNodes,length(),nodes,fdistance,in,out); return;
      }
  }
  //Otherwise, use the general version
  scatter_general(nNodes,length(),nodes,fdistance,in,out);
}


/***********************************************
"ScatterAdd" routines add the message data (in) to the
shared nodes distributed through the user's data (out).
 */
#define scatteradd_args (int nVal, \
		    const int *nodeIdx,int nodeScale, \
		    const char *in,char *out)

#define scatteradd_doubles(n,copy) \
static void scatteradd_double##n scatteradd_args \
{ \
  const double *id=(const double *)in; \
  for(int i=0;i<nVal;i++) { \
      double *targ = (double *)(out+nodeIdx[i]*nodeScale); \
      copy \
      id+=n; \
  } \
}

scatteradd_doubles(1,targ[0]+=id[0];)
scatteradd_doubles(2,targ[0]+=id[0];targ[1]+=id[1];)
scatteradd_doubles(3,targ[0]+=id[0];targ[1]+=id[1];targ[2]+=id[2];)

  /**
   * For each field in the list nodes[0..nNodes-1], add the
   * compressed data from v_in into the user data in v_out.
   */
void DType::scatteradd(int nNodes,const int *nodes,
		   const void *v_in,void *v_out) const
{
  const char *in=(const char *)v_in;
  char *out=(char *)v_out;
  out += init_offset;
  //Try for a more specialized version if possible:
  if (base_type == FEM_DOUBLE) {
      switch(vec_len) {
      case 1: scatteradd_double1(nNodes,nodes,fdistance,in,out); return;
      case 2: scatteradd_double2(nNodes,nodes,fdistance,in,out); return;
      case 3: scatteradd_double3(nNodes,nodes,fdistance,in,out); return;
      }
  }

  /*Otherwise we need the slow, general version:
    As a hack, we use the reduction combine function to do the add.*/
  reduction_combine_fn fn=reduction_combine(*this,FEM_SUM);
  int len=length();
  for(int i=0;i<nNodes;i++) {
    void *cnode = (void*) (out+nodes[i]*fdistance);
    fn(vec_len, cnode, in);
    in += len;
  }
}



