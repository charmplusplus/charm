/*Charm++ Finite Element Framework:
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

Implementation file for IDXL: IDXL user data types.
These allow data to be scooped right out of a user's
arrays.
*/
#include "idxl.h" /* for IDXL_Abort */
#include "idxl_layout.h"
#include <string.h> /* for memcpy */
#include <math.h>
#include <limits.h>
#include <float.h> /*for FLT_MIN on non-Suns*/

int IDXL_Layout::type_size(int dataType,const char *callingRoutine)
{
    switch(dataType) {
      case IDXL_BYTE : return 1; break;
      case IDXL_INT : return sizeof(int); break;
      case IDXL_REAL : return sizeof(float); break;
      case IDXL_DOUBLE : return sizeof(double); break;
      case IDXL_INDEX_0 : return sizeof(int); break;
      case IDXL_INDEX_1 : return sizeof(int); break;
      default: IDXL_Abort(callingRoutine,"Expected an IDXL data type, but got %d",dataType);
    }
    return -1;
}

const char *IDXL_Layout::type_name(int dataType,const char *callingRoutine) 
{
    switch(dataType) {
      case IDXL_BYTE : return "IDXL_BYTE"; break;
      case IDXL_INT : return "IDXL_INT"; break;
      case IDXL_REAL : return "IDXL_REAL"; break;
      case IDXL_DOUBLE : return "IDXL_DOUBLE"; break;
    }
    return "(unknown IDXL datatype)";
}

/******************** Reduction Support **********************/

// These functions apply this operation to a single value, and are called 
// directly.
template<class T>
inline void assignOne(T* lhs, T val) { *lhs = val; }

template<class T>
inline void assignOne(T* lhs, const T *rhs) { *lhs = *rhs; }

template<class T>
inline void sumOne(T* lhs, const T* rhs) { *lhs += *rhs; }

template<class T>
inline void prodOne(T* lhs, const T* rhs) { *lhs *= *rhs; }

template<class T>
inline void maxOne(T* lhs, const T* rhs) { *lhs = (*lhs > *rhs) ? *lhs : *rhs; }

template<class T>
inline void minOne(T* lhs, const T* rhs) { *lhs = (*lhs < *rhs) ? *lhs : *rhs; }

// reduction_combine_fn definitions.
// These functions apply the given operation to an entire user-formatted row src,
// accumulating the result into the compressed-format row dest.  
// FIXME: figure out how to define this using templates, not macros.
#define oneToFn(oneFn,gatherName) \
template<class T> \
void gatherName(byte *b_dest,const byte *src,const IDXL_Layout *srcLayout) { \
	src+=srcLayout->offset; \
	int skew=srcLayout->skew; \
	T *dest=(T *)b_dest; \
	int i=0, width=srcLayout->width; \
	for (i=0;i<width;i++) { \
		oneFn(dest,(const T *)src); \
		dest++; \
		src+=skew; \
	} \
}

oneToFn(sumOne,sumFn)
oneToFn(prodOne,prodFn)
oneToFn(maxOne,maxFn)
oneToFn(minOne,minFn)


template<class T>
void assignFn(int len,T *dest,T val) {
  for (int i=0;i<len;i++) dest[i]=val;
}


void reduction_initialize(const IDXL_Layout& dt, void *lhs, int op,const char *callingRoutine)
{
  switch(op) {
    case IDXL_SUM:
      switch(dt.type) {
        case IDXL_BYTE : assignFn(dt.width,(byte*)lhs, (byte)0); break;
        case IDXL_INT : assignFn(dt.width,(int*)lhs, 0); break;
        case IDXL_REAL : assignFn(dt.width,(float*)lhs, (float)0.0); break;
        case IDXL_DOUBLE : assignFn(dt.width,(double*)lhs, 0.0); break;
        default: IDXL_Abort(callingRoutine,"Invalid IDXL data type %d",dt.type);
      }
      break;
    case IDXL_PROD:
      switch(dt.type) {
        case IDXL_BYTE : assignFn(dt.width,(byte*)lhs, (byte)1); break;
        case IDXL_INT : assignFn(dt.width,(int*)lhs, 1); break;
        case IDXL_REAL : assignFn(dt.width,(float*)lhs, (float)1.0); break;
        case IDXL_DOUBLE : assignFn(dt.width,(double*)lhs, 1.0); break;
      }
      break;
    case IDXL_MAX:
      switch(dt.type) {
        case IDXL_BYTE : assignFn(dt.width,(byte*)lhs, (byte)CHAR_MIN); break;
        case IDXL_INT : assignFn(dt.width,(int*)lhs, INT_MIN); break;
        case IDXL_REAL : assignFn(dt.width,(float*)lhs, FLT_MIN); break;
        case IDXL_DOUBLE : assignFn(dt.width,(double*)lhs, DBL_MIN); break;
      }
      break;
    case IDXL_MIN:
      switch(dt.type) {
        case IDXL_BYTE : assignFn(dt.width,(byte*)lhs, (byte)CHAR_MAX); break;
        case IDXL_INT : assignFn(dt.width,(int*)lhs, INT_MAX); break;
        case IDXL_REAL : assignFn(dt.width,(float*)lhs, FLT_MAX); break;
        case IDXL_DOUBLE : assignFn(dt.width,(double*)lhs, DBL_MAX); break;
      }
      break;
    default: IDXL_Abort(callingRoutine,"Expected an IDXL reduction type, but got %d",op);
  }
}

//This odd-looking define selects the appropriate templated function and
// returns it as a function pointer.
#define idxl_type_return(type,fn) \
      switch(type) {\
        case IDXL_BYTE : return fn<byte>;\
        case IDXL_INT : return fn<int>;\
        case IDXL_REAL : return fn<float>;\
        case IDXL_DOUBLE : return fn<double>;\
      }

#define idxl_type_call(type,fn,args) \
      switch(type) {\
        case IDXL_BYTE : fn<byte> args;\
        case IDXL_INT : fn<int> args;\
        case IDXL_REAL : fn<float> args;\
        case IDXL_DOUBLE : fn<double> args;\
      }

reduction_combine_fn reduction_combine(const IDXL_Layout& dt, int op,const char *callingRoutine)
{
  switch(op) {
    case IDXL_SUM: idxl_type_return(dt.type, sumFn); break;
    case IDXL_PROD: idxl_type_return(dt.type, prodFn); break;
    case IDXL_MIN: idxl_type_return(dt.type, minFn); break;
    case IDXL_MAX: idxl_type_return(dt.type, maxFn); break;
    default: IDXL_Abort(callingRoutine,"Expected an IDXL reduction type, but got %d",op);
  }
  IDXL_Abort(callingRoutine,"Expected an IDXL data type, but got %d",dt.type);
  return NULL;
}


/************************* Gather ***********************
"Gather" routines extract data distributed (nodeIdx)
through the user's array (in) and collect it into a message (out).
 */

// Hopefully the compiler's common-subexpression elimination will
//   get rid of the multiplies in the inner loop.
template <class T>
void gatherUserData(const void *user,int nIndices,const int *indices,
		IDXL_LAYOUT_PARAM,void *v_compressed) 
{
	T *compressed=(T *)v_compressed;
	for (int r=0;r<nIndices;r++) {
		int sr=indices[r];
		for (int c=0;c<width;c++)
			compressed[c]=IDXL_LAYOUT_DEREF(T,user,sr,c);
		compressed+=width;
	}
}

  /**
   * For each record in nodes[0..nNodes-1], copy the
   * user data in v_in into the compressed data in v_out.
   */
void IDXL_Layout::gather(int nIndices,const int *indices,
		   const void *v_in,void *v_out) const
{
  idxl_type_call(this->type, gatherUserData, 
  	(v_in, nIndices,indices, IDXL_LAYOUT_CALL(*this), v_out)
  );
}

/************************ Scatter ************************
"Scatter" routines are the opposite of gather: they take
compressed data from a message (in) and copy it into selected
indices of the user array (out).
 */

template <class T>
void scatterUserData(void *user,int nIndices,const int *indices,
		IDXL_LAYOUT_PARAM,const void *v_compressed) 
{
	const T *compressed=(const T *)v_compressed;
	for (int r=0;r<nIndices;r++) {
		int sr=indices[r];
		for (int c=0;c<width;c++)
			IDXL_LAYOUT_DEREF(T,user,sr,c)=compressed[c];
		compressed+=width;
	}
}

  /**
   * For each field in the list nodes[0..nNodes-1], copy the
   * compressed data from v_in into the user data in v_out.
   */
void IDXL_Layout::scatter(int nIndices,const int *indices,
		   const void *v_in,void *v_out) const
{
  idxl_type_call(this->type, scatterUserData, 
  	(v_out, nIndices,indices, IDXL_LAYOUT_CALL(*this), v_in)
  );
}


/************************ ScatterAdd ***********************
"ScatterAdd" routines add the message data (in) to the
shared nodes distributed through the user's data (out).
 */
template <class T>
void scatterAddUserData(void *user,int nIndices,const int *indices,
		IDXL_LAYOUT_PARAM,const void *v_compressed) 
{
	const T *compressed=(const T *)v_compressed;
	for (int r=0;r<nIndices;r++) {
		int sr=indices[r];
		for (int c=0;c<width;c++)
			IDXL_LAYOUT_DEREF(T,user,sr,c)+=compressed[c];
		compressed+=width;
	}
}

  /**
   * For each field in the list nodes[0..nNodes-1], add the
   * compressed data from v_in into the user data in v_out.
   */
void IDXL_Layout::scatteradd(int nIndices,const int *indices,
		   const void *v_in,void *v_out) const
{
  idxl_type_call(this->type, scatterAddUserData, 
  	(v_out, nIndices,indices, IDXL_LAYOUT_CALL(*this), v_in)
  );
}



/********************** Data_list: *******************/
void IDXL_Layout_List::badLayout(IDXL_Layout_t l,const char *callingRoutine) const
{
	IDXL_Abort(callingRoutine,"Expected an IDXL_Layout_t, got %d",l);
}

IDXL_Layout_List::IDXL_Layout_List() {
	for (int i=0;i<MAX_DT;i++) list[i]=NULL;
}
void IDXL_Layout_List::pup(PUP::er &p) {
	for (int i=0;i<MAX_DT;i++) {
		int isNULL=(list[i]==NULL);
		p|isNULL;
		if (!isNULL) {
			if (list[i]==NULL) list[i]=new IDXL_Layout();
			p|*list[i];
		}
	}
}
IDXL_Layout_List::~IDXL_Layout_List() {
	empty();
}
/// Clear all stored layouts:
void IDXL_Layout_List::empty(void) {
	for (int i=0;i<MAX_DT;i++) 
		if (list[i]!=NULL) {
			delete list[i];
			list[i]=NULL;
		}
}

IDXL_Layout_t IDXL_Layout_List::put(const IDXL_Layout &dt) {
	for (int i=0;i<MAX_DT;i++) 
		if (list[i]==NULL) {
			list[i]=new IDXL_Layout(dt);
			return FIRST_DT+i;
		}
	// if we get here, the table is full:
	IDXL_Abort("","Registered too many IDXL_Layouts! (only have room for %d)",MAX_DT);
}
void IDXL_Layout_List::destroy(IDXL_Layout_t l,const char *callingRoutine) {
	check(l,callingRoutine);
	int i=l-FIRST_DT;
	delete list[i];
	list[i]=NULL;
}

