
#define AMPIMSGLOG    0

#define exit exit /*Supress definition of exit in ampi.h*/
#include "ampiimpl.h"
#include "tcharm.h"
#if CMK_TRACE_ENABLED && CMK_PROJECTOR
#include "ampiEvents.h" /*** for trace generation for projector *****/
#include "ampiProjections.h"
#endif

#if CMK_BIGSIM_CHARM
#include "bigsim_logs.h"
#endif

/* change this to MPI_ERRORS_RETURN to not abort on errors */
#define AMPI_ERRHANDLER MPI_ERRORS_ARE_FATAL

#define AMPI_PRINT_IDLE 0

/* change this define to "x" to trace all send/recv's */
#define MSG_ORDER_DEBUG(x) //x /* empty */
/* change this define to "x" to trace user calls */
#define USER_CALL_DEBUG(x) // ckout<<"vp "<<TCHARM_Element()<<": "<<x<<endl;
#define STARTUP_DEBUG(x) //ckout<<"ampi[pe "<<CkMyPe()<<"] "<< x <<endl;
#define FUNCCALL_DEBUG(x) //x /* empty */

/* For MPI_Get_library_version */
extern "C" const char * const CmiCommitID;

static CkDDT *getDDT(void) {
  return getAmpiParent()->myDDT;
}

/* if error checking is disabled, ampiErrhandler is defined as a macro in ampiimpl.h */
#if AMPI_ERROR_CHECKING
inline int ampiErrhandler(const char* func, int errcode) {
  if (AMPI_ERRHANDLER == MPI_ERRORS_ARE_FATAL && errcode != MPI_SUCCESS) {
    // Abort with a nice message of the form: 'func' failed with error code 'errstr'.
    //  where 'func' is the name of the failed AMPI_ function and 'errstr'
    //  is the string returned by AMPI_Error_string for errcode.
    int funclen = strlen(func);
    const char* filler = " failed with error code ";
    int fillerlen = strlen(filler);
    int errstrlen;
    char errstr[MPI_MAX_ERROR_STRING];
    AMPI_Error_string(errcode, errstr, &errstrlen);
    vector<char> str(funclen + fillerlen + errstrlen);
    strcpy(&str[0], func);
    strcat(&str[0], filler);
    strcat(&str[0], errstr);
    CkAbort(&str[0]);
  }
  return errcode;
}
#endif

inline int checkCommunicator(const char* func, MPI_Comm comm) {
  if (comm == MPI_COMM_NULL)
    return ampiErrhandler(func, MPI_ERR_COMM);
  return MPI_SUCCESS;
}

inline int checkCount(const char* func, int count) {
  if (count < 0)
    return ampiErrhandler(func, MPI_ERR_COUNT);
  return MPI_SUCCESS;
}

inline int checkData(const char* func, MPI_Datatype data) {
  if (data == MPI_DATATYPE_NULL)
    return ampiErrhandler(func, MPI_ERR_TYPE);
  return MPI_SUCCESS;
}

inline int checkTag(const char* func, int tag) {
  if ((tag != MPI_ANY_TAG && tag < 0) || (tag > MPI_TAG_UB_VALUE))
    return ampiErrhandler(func, MPI_ERR_TAG);
  return MPI_SUCCESS;
}

inline int checkRank(const char* func, int rank, MPI_Comm comm) {
  int size;
  AMPI_Comm_size(comm, &size);
  if (((rank >= 0) && (rank < size)) ||
      (rank == MPI_ANY_SOURCE)       ||
      (rank == MPI_PROC_NULL))
    return MPI_SUCCESS;
  return ampiErrhandler(func, MPI_ERR_RANK);
}

inline int checkBuf(const char* func, void *buf, int count) {
  if ((count != 0 && buf == NULL) || buf == MPI_IN_PLACE)
    return ampiErrhandler(func, MPI_ERR_BUFFER);
  return MPI_SUCCESS;
}

inline int errorCheck(const char* func, MPI_Comm comm, int ifComm, int count,
                      int ifCount, MPI_Datatype data, int ifData, int tag,
                      int ifTag, int rank, int ifRank, void *buf1, int ifBuf1,
                      void *buf2=0, int ifBuf2=0) {
  int ret;
  if (ifComm) {
    ret = checkCommunicator(func, comm);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  if (ifCount) {
    ret = checkCount(func, count);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  if (ifData) {
    ret = checkData(func, data);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  if (ifTag) {
    ret = checkTag(func, tag);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  if (ifRank) {
    ret = checkRank(func, rank, comm);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  if (ifBuf1) {
    ret = checkBuf(func, buf1, count);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  if (ifBuf2) {
    ret = checkBuf(func, buf2, count);
    if (ret != MPI_SUCCESS)
      return ampiErrhandler(func, ret);
  }
  return MPI_SUCCESS;
}

//------------- startup -------------
static mpi_comm_worlds mpi_worlds;

int _mpi_nworlds; /*Accessed by ampif*/
int MPI_COMM_UNIVERSE[MPI_MAX_COMM_WORLDS]; /*Accessed by user code*/

class AmpiComplex {
 public:
  float re, im;
  void operator+=(const AmpiComplex &a) {
    re+=a.re;
    im+=a.im;
  }
  void operator*=(const AmpiComplex &a) {
    float nu_re=re*a.re-im*a.im;
    im=re*a.im+im*a.re;
    re=nu_re;
  }
  int operator>(const AmpiComplex &a) {
    CkAbort("AMPI> Cannot compare complex numbers with MPI_MAX\n");
    return 0;
  }
  int operator<(const AmpiComplex &a) {
    CkAbort("AMPI> Cannot compare complex numbers with MPI_MIN\n");
    return 0;
  }
};

class AmpiDoubleComplex {
 public:
  double re, im;
  void operator+=(const AmpiDoubleComplex &a) {
    re+=a.re;
    im+=a.im;
  }
  void operator*=(const AmpiDoubleComplex &a) {
    double nu_re=re*a.re-im*a.im;
    im=re*a.im+im*a.re;
    re=nu_re;
  }
  int operator>(const AmpiDoubleComplex &a) {
    CkAbort("AMPI> Cannot compare double complex numbers with MPI_MAX\n");
    return 0;
  }
  int operator<(const AmpiDoubleComplex &a) {
    CkAbort("AMPI> Cannot compare double complex numbers with MPI_MIN\n");
    return 0;
  }
};

class AmpiLongDoubleComplex {
 public:
  long double re, im;
  void operator+=(const AmpiLongDoubleComplex &a) {
    re+=a.re;
    im+=a.im;
  }
  void operator*=(const AmpiLongDoubleComplex &a) {
    long double nu_re=re*a.re-im*a.im;
    im=re*a.im+im*a.re;
    re=nu_re;
  }
  int operator>(const AmpiLongDoubleComplex &a) {
    CkAbort("AMPI> Cannot compare long double complex numbers with MPI_MAX\n");
    return 0;
  }
  int operator<(const AmpiLongDoubleComplex &a) {
    CkAbort("AMPI> Cannot compare long double complex numbers with MPI_MIN\n");
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

/* For MPI_MAX, MPI_MIN, MPI_SUM, and MPI_PROD: */
#define MPI_OP_SWITCH(OPNAME) \
  int i; \
switch (*datatype) { \
  case MPI_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(char); } break; \
  case MPI_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed short int); } break; \
  case MPI_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed int); } break; \
  case MPI_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed long); } break; \
  case MPI_UNSIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned char); } break; \
  case MPI_UNSIGNED_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned short); } break; \
  case MPI_UNSIGNED: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned int); } break; \
  case MPI_UNSIGNED_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned long); } break; \
  case MPI_FLOAT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(float); } break; \
  case MPI_DOUBLE: for(i=0;i<(*len);i++) { MPI_OP_IMPL(double); } break; \
  case MPI_COMPLEX: for(i=0;i<(*len);i++) { MPI_OP_IMPL(AmpiComplex); } break; \
  case MPI_DOUBLE_COMPLEX: for(i=0;i<(*len);i++) { MPI_OP_IMPL(AmpiDoubleComplex); } break; \
  case MPI_LONG_LONG_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed long long); } break; \
  case MPI_SIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed char); } break; \
  case MPI_UNSIGNED_LONG_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned long long); } break; \
  case MPI_WCHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(wchar_t); } break; \
  case MPI_INT8_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int8_t); } break; \
  case MPI_INT16_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int16_t); } break; \
  case MPI_INT32_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int32_t); } break; \
  case MPI_INT64_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int64_t); } break; \
  case MPI_UINT8_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint8_t); } break; \
  case MPI_UINT16_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint16_t); } break; \
  case MPI_UINT32_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint32_t); } break; \
  case MPI_UINT64_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint64_t); } break; \
  case MPI_FLOAT_COMPLEX: for(i=0;i<(*len);i++) { MPI_OP_IMPL(AmpiComplex); } break; \
  case MPI_LONG_DOUBLE_COMPLEX: for(i=0;i<(*len);i++) { MPI_OP_IMPL(AmpiLongDoubleComplex); } break; \
  case MPI_AINT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(MPI_Aint); } break; \
  default: \
           ckerr << "Type " << *datatype << " with Op "#OPNAME" not supported." << endl; \
  CkAbort("Unsupported MPI datatype for MPI Op"); \
};\

/* For MPI_LAND, MPI_LOR, and MPI_LXOR: */
#define MPI_LOGICAL_OP_SWITCH(OPNAME) \
  int i; \
switch (*datatype) { \
  case MPI_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed short int); } break; \
  case MPI_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed int); } break; \
  case MPI_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed long); } break; \
  case MPI_UNSIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned char); } break; \
  case MPI_UNSIGNED_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned short); } break; \
  case MPI_UNSIGNED: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned int); } break; \
  case MPI_UNSIGNED_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned long); } break; \
  case MPI_LONG_LONG_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed long long); } break; \
  case MPI_SIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed char); } break; \
  case MPI_UNSIGNED_LONG_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned long long); } break; \
  case MPI_INT8_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int8_t); } break; \
  case MPI_INT16_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int16_t); } break; \
  case MPI_INT32_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int32_t); } break; \
  case MPI_INT64_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int64_t); } break; \
  case MPI_UINT8_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint8_t); } break; \
  case MPI_UINT16_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint16_t); } break; \
  case MPI_UINT32_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint32_t); } break; \
  case MPI_UINT64_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint64_t); } break; \
  case MPI_LOGICAL: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int); } break; \
  case MPI_C_BOOL: for(i=0;i<(*len);i++) { MPI_OP_IMPL(bool); } break; \
  default: \
           ckerr << "Type " << *datatype << " with Op "#OPNAME" not supported." << endl; \
  CkAbort("Unsupported MPI datatype for MPI Op"); \
};\

/* For MPI_BAND, MPI_BOR, and MPI_BXOR: */
#define MPI_BITWISE_OP_SWITCH(OPNAME) \
  int i; \
switch (*datatype) { \
  case MPI_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed short int); } break; \
  case MPI_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed int); } break; \
  case MPI_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed long); } break; \
  case MPI_UNSIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned char); } break; \
  case MPI_UNSIGNED_SHORT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned short); } break; \
  case MPI_UNSIGNED: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned int); } break; \
  case MPI_UNSIGNED_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned long); } break; \
  case MPI_LONG_LONG_INT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed long long); } break; \
  case MPI_SIGNED_CHAR: for(i=0;i<(*len);i++) { MPI_OP_IMPL(signed char); } break; \
  case MPI_UNSIGNED_LONG_LONG: for(i=0;i<(*len);i++) { MPI_OP_IMPL(unsigned long long); } break; \
  case MPI_INT8_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int8_t); } break; \
  case MPI_INT16_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int16_t); } break; \
  case MPI_INT32_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int32_t); } break; \
  case MPI_INT64_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(int64_t); } break; \
  case MPI_UINT8_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint8_t); } break; \
  case MPI_UINT16_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint16_t); } break; \
  case MPI_UINT32_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint32_t); } break; \
  case MPI_UINT64_T: for(i=0;i<(*len);i++) { MPI_OP_IMPL(uint64_t); } break; \
  case MPI_BYTE: for(i=0;i<(*len);i++) { MPI_OP_IMPL(char); } break; \
  case MPI_AINT: for(i=0;i<(*len);i++) { MPI_OP_IMPL(MPI_Aint); } break; \
  default: \
           ckerr << "Type " << *datatype << " with Op "#OPNAME" not supported." << endl; \
  CkAbort("Unsupported MPI datatype for MPI Op"); \
};\

void MPI_MAX_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  if(((type *)invec)[i] > ((type *)inoutvec)[i]) ((type *)inoutvec)[i] = ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_MAX)
#undef MPI_OP_IMPL
}

void MPI_MIN_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  if(((type *)invec)[i] < ((type *)inoutvec)[i]) ((type *)inoutvec)[i] = ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_MIN)
#undef MPI_OP_IMPL
}

void MPI_SUM_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] += ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_SUM)
#undef MPI_OP_IMPL
}

void MPI_PROD_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] *= ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_PROD)
#undef MPI_OP_IMPL
}

void MPI_REPLACE_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = ((type *)invec)[i];
  MPI_OP_SWITCH(MPI_REPLACE)
#undef MPI_OP_IMPL
}

void MPI_NO_OP_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  /* no-op */
}

void MPI_LAND_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = ((type *)inoutvec)[i] && ((type *)invec)[i];
  MPI_LOGICAL_OP_SWITCH(MPI_LAND)
#undef MPI_OP_IMPL
}

void MPI_BAND_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = ((type *)inoutvec)[i] & ((type *)invec)[i];
  MPI_BITWISE_OP_SWITCH(MPI_BAND)
#undef MPI_OP_IMPL
}

void MPI_LOR_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = ((type *)inoutvec)[i] || ((type *)invec)[i];
  MPI_LOGICAL_OP_SWITCH(MPI_LAND)
#undef MPI_OP_IMPL
}

void MPI_BOR_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = ((type *)inoutvec)[i] | ((type *)invec)[i];
  MPI_BITWISE_OP_SWITCH(MPI_BAND)
#undef MPI_OP_IMPL
}

void MPI_LXOR_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = (((type *)inoutvec)[i]&&(!((type *)invec)[i]))||(!(((type *)inoutvec)[i])&&((type *)invec)[i]);
  MPI_LOGICAL_OP_SWITCH(MPI_LAND)
#undef MPI_OP_IMPL
}

void MPI_BXOR_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
#define MPI_OP_IMPL(type) \
  ((type *)inoutvec)[i] = ((type *)inoutvec)[i] ^ ((type *)invec)[i];
  MPI_BITWISE_OP_SWITCH(MPI_BAND)
#undef MPI_OP_IMPL
}

#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

void MPI_MAXLOC_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;

  switch (*datatype) {
    case MPI_FLOAT_INT:
      for(i=0;i<(*len);i++){
        if(((FloatInt *)invec)[i].val > ((FloatInt *)inoutvec)[i].val)
          ((FloatInt *)inoutvec)[i] = ((FloatInt *)invec)[i];
        else if(((FloatInt *)invec)[i].val == ((FloatInt *)inoutvec)[i].val)
          ((FloatInt *)inoutvec)[i].idx = MIN(((FloatInt *)inoutvec)[i].idx, ((FloatInt *)invec)[i].idx);
      }
      break;
    case MPI_DOUBLE_INT:
      for(i=0;i<(*len);i++){
        if(((DoubleInt *)invec)[i].val > ((DoubleInt *)inoutvec)[i].val)
          ((DoubleInt *)inoutvec)[i] = ((DoubleInt *)invec)[i];
        else if(((DoubleInt *)invec)[i].val == ((DoubleInt *)inoutvec)[i].val)
          ((DoubleInt *)inoutvec)[i].idx = MIN(((DoubleInt *)inoutvec)[i].idx, ((DoubleInt *)invec)[i].idx);
      }
      break;
    case MPI_LONG_INT:
      for(i=0;i<(*len);i++){
        if(((LongInt *)invec)[i].val > ((LongInt *)inoutvec)[i].val)
          ((LongInt *)inoutvec)[i] = ((LongInt *)invec)[i];
        else if(((LongInt *)invec)[i].val == ((LongInt *)inoutvec)[i].val)
          ((LongInt *)inoutvec)[i].idx = MIN(((LongInt *)inoutvec)[i].idx, ((LongInt *)invec)[i].idx);
      }
      break;
    case MPI_2INT:
      for(i=0;i<(*len);i++){
        if(((IntInt *)invec)[i].val > ((IntInt *)inoutvec)[i].val)
          ((IntInt *)inoutvec)[i] = ((IntInt *)invec)[i];
        else if(((IntInt *)invec)[i].val == ((IntInt *)inoutvec)[i].val)
          ((IntInt *)inoutvec)[i].idx = MIN(((IntInt *)inoutvec)[i].idx, ((IntInt *)invec)[i].idx);
      }
      break;
    case MPI_SHORT_INT:
      for(i=0;i<(*len);i++){
        if(((ShortInt *)invec)[i].val > ((ShortInt *)inoutvec)[i].val)
          ((ShortInt *)inoutvec)[i] = ((ShortInt *)invec)[i];
        else if(((ShortInt *)invec)[i].val == ((ShortInt *)inoutvec)[i].val)
          ((ShortInt *)inoutvec)[i].idx = MIN(((ShortInt *)inoutvec)[i].idx, ((ShortInt *)invec)[i].idx);
      }
      break;
    case MPI_LONG_DOUBLE_INT:
      for(i=0;i<(*len);i++){
        if(((LongdoubleInt *)invec)[i].val > ((LongdoubleInt *)inoutvec)[i].val)
          ((LongdoubleInt *)inoutvec)[i] = ((LongdoubleInt *)invec)[i];
        else if(((LongdoubleInt *)invec)[i].val == ((LongdoubleInt *)inoutvec)[i].val)
          ((LongdoubleInt *)inoutvec)[i].idx = MIN(((LongdoubleInt *)inoutvec)[i].idx, ((LongdoubleInt *)invec)[i].idx);
      }
      break;
    case MPI_2FLOAT:
      for(i=0;i<(*len);i++){
        if(((FloatFloat *)invec)[i].val > ((FloatFloat *)inoutvec)[i].val)
          ((FloatFloat *)inoutvec)[i] = ((FloatFloat *)invec)[i];
        else if(((FloatFloat *)invec)[i].val == ((FloatFloat *)inoutvec)[i].val)
          ((FloatFloat *)inoutvec)[i].idx = MIN(((FloatFloat *)inoutvec)[i].idx, ((FloatFloat *)invec)[i].idx);
      }
      break;
    case MPI_2DOUBLE:
      for(i=0;i<(*len);i++){
        if(((DoubleDouble *)invec)[i].val > ((DoubleDouble *)inoutvec)[i].val)
          ((DoubleDouble *)inoutvec)[i] = ((DoubleDouble *)invec)[i];
        else if(((DoubleDouble *)invec)[i].val == ((DoubleDouble *)inoutvec)[i].val)
          ((DoubleDouble *)inoutvec)[i].idx = MIN(((DoubleDouble *)inoutvec)[i].idx, ((DoubleDouble *)invec)[i].idx);
      }
      break;
    default:
      ckerr << "Type " << *datatype << " with Op MPI_MAXLOC not supported." << endl;
      CkAbort("exiting");
  }
}

void MPI_MINLOC_USER_FN( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){
  int i;
  switch (*datatype) {
    case MPI_FLOAT_INT:
      for(i=0;i<(*len);i++){
        if(((FloatInt *)invec)[i].val < ((FloatInt *)inoutvec)[i].val)
          ((FloatInt *)inoutvec)[i] = ((FloatInt *)invec)[i];
        else if(((FloatInt *)invec)[i].val == ((FloatInt *)inoutvec)[i].val)
          ((FloatInt *)inoutvec)[i].idx = MIN(((FloatInt *)inoutvec)[i].idx, ((FloatInt *)invec)[i].idx);
      }
      break;
    case MPI_DOUBLE_INT:
      for(i=0;i<(*len);i++){
        if(((DoubleInt *)invec)[i].val < ((DoubleInt *)inoutvec)[i].val)
          ((DoubleInt *)inoutvec)[i] = ((DoubleInt *)invec)[i];
        else if(((DoubleInt *)invec)[i].val == ((DoubleInt *)inoutvec)[i].val)
          ((DoubleInt *)inoutvec)[i].idx = MIN(((DoubleInt *)inoutvec)[i].idx, ((DoubleInt *)invec)[i].idx);
      }
      break;
    case MPI_LONG_INT:
      for(i=0;i<(*len);i++){
        if(((LongInt *)invec)[i].val < ((LongInt *)inoutvec)[i].val)
          ((LongInt *)inoutvec)[i] = ((LongInt *)invec)[i];
        else if(((LongInt *)invec)[i].val == ((LongInt *)inoutvec)[i].val)
          ((LongInt *)inoutvec)[i].idx = MIN(((LongInt *)inoutvec)[i].idx, ((LongInt *)invec)[i].idx);
      }
      break;
    case MPI_2INT:
      for(i=0;i<(*len);i++){
        if(((IntInt *)invec)[i].val < ((IntInt *)inoutvec)[i].val)
          ((IntInt *)inoutvec)[i] = ((IntInt *)invec)[i];
        else if(((IntInt *)invec)[i].val == ((IntInt *)inoutvec)[i].val)
          ((IntInt *)inoutvec)[i].idx = MIN(((IntInt *)inoutvec)[i].idx, ((IntInt *)invec)[i].idx);
      }
      break;
    case MPI_SHORT_INT:
      for(i=0;i<(*len);i++){
        if(((ShortInt *)invec)[i].val < ((ShortInt *)inoutvec)[i].val)
          ((ShortInt *)inoutvec)[i] = ((ShortInt *)invec)[i];
        else if(((ShortInt *)invec)[i].val == ((ShortInt *)inoutvec)[i].val)
          ((ShortInt *)inoutvec)[i].idx = MIN(((ShortInt *)inoutvec)[i].idx, ((ShortInt *)invec)[i].idx);
      }
      break;
    case MPI_LONG_DOUBLE_INT:
      for(i=0;i<(*len);i++){
        if(((LongdoubleInt *)invec)[i].val < ((LongdoubleInt *)inoutvec)[i].val)
          ((LongdoubleInt *)inoutvec)[i] = ((LongdoubleInt *)invec)[i];
        else if(((LongdoubleInt *)invec)[i].val == ((LongdoubleInt *)inoutvec)[i].val)
          ((LongdoubleInt *)inoutvec)[i].idx = MIN(((LongdoubleInt *)inoutvec)[i].idx, ((LongdoubleInt *)invec)[i].idx);
      }
      break;
    case MPI_2FLOAT:
      for(i=0;i<(*len);i++){
        if(((FloatFloat *)invec)[i].val < ((FloatFloat *)inoutvec)[i].val)
          ((FloatFloat *)inoutvec)[i] = ((FloatFloat *)invec)[i];
        else if(((FloatFloat *)invec)[i].val == ((FloatFloat *)inoutvec)[i].val)
          ((FloatFloat *)inoutvec)[i].idx = MIN(((FloatFloat *)inoutvec)[i].idx, ((FloatFloat *)invec)[i].idx);
      }
      break;
    case MPI_2DOUBLE:
      for(i=0;i<(*len);i++){
        if(((DoubleDouble *)invec)[i].val < ((DoubleDouble *)inoutvec)[i].val)
          ((DoubleDouble *)inoutvec)[i] = ((DoubleDouble *)invec)[i];
        else if(((DoubleDouble *)invec)[i].val == ((DoubleDouble *)inoutvec)[i].val)
          ((DoubleDouble *)inoutvec)[i].idx = MIN(((DoubleDouble *)inoutvec)[i].idx, ((DoubleDouble *)invec)[i].idx);
      }
      break;
    default:
      ckerr << "Type " << *datatype << " with Op MPI_MINLOC not supported." << endl;
      CkAbort("exiting");
  }
}

/*
 * AMPI's generic reducer type, AmpiReducer, is used only
 * for MPI_Op/MPI_Datatype combinations that Charm++ does
 * not have built-in support for. AmpiReducer reduction
 * contributions all contain an AmpiOpHeader, that contains
 * the function pointer to an MPI_User_function* that is
 * applied to all contributions in AmpiReducerFunc().
 *
 * If AmpiReducer is used, the final reduction message will
 * have an additional sizeof(AmpiOpHeader) bytes in the
 * buffer before any user data. ampi::processRednMsg() strips
 * the header.
 *
 * If a non-commutative (user-defined) reduction is used,
 * ampi::processNoncommutativeRednMsg() strips the headers
 * and applies the op to all contributions in rank order.
 */
CkReduction::reducerType AmpiReducer;

// every msg contains a AmpiOpHeader structure before user data
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
  vector<char> ret(szhdr+szdata);
  char *retPtr = &ret[0];
  memcpy(retPtr,msgs[0]->getData(),szhdr+szdata);
  for(int i=1;i<nMsg;i++){
    (*func)((void *)((char *)msgs[i]->getData()+szhdr),(void *)(retPtr+szhdr),&len,&dtype);
  }
  CkReductionMsg *retmsg = CkReductionMsg::buildNew(szhdr+szdata,retPtr);
  return retmsg;
}

static CkReduction::reducerType getBuiltinReducerType(MPI_Datatype type, MPI_Op op)
{
  switch (type) {
    case MPI_INT32_T:
      if (getDDT()->getSize(MPI_INT32_T) != getDDT()->getSize(MPI_INT)) break;
      // else: fall thru to MPI_INT
    case MPI_INT:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_int;
        case MPI_MIN:  return CkReduction::min_int;
        case MPI_SUM:  return CkReduction::sum_int;
        case MPI_PROD: return CkReduction::product_int;
        case MPI_LAND: return CkReduction::logical_and_int;
        case MPI_LOR:  return CkReduction::logical_or_int;
        case MPI_LXOR: return CkReduction::logical_xor_int;
        case MPI_BAND: return CkReduction::bitvec_and_int;
        case MPI_BOR:  return CkReduction::bitvec_or_int;
        case MPI_BXOR: return CkReduction::bitvec_xor_int;
        default:       break;
      }
    case MPI_FLOAT:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_float;
        case MPI_MIN:  return CkReduction::min_float;
        case MPI_SUM:  return CkReduction::sum_float;
        case MPI_PROD: return CkReduction::product_float;
        default:       break;
      }
    case MPI_DOUBLE:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_double;
        case MPI_MIN:  return CkReduction::min_double;
        case MPI_SUM:  return CkReduction::sum_double;
        case MPI_PROD: return CkReduction::product_double;
        default:       break;
      }
    case MPI_INT8_T:
      if (getDDT()->getSize(MPI_INT8_T) != getDDT()->getSize(MPI_CHAR)) break;
      // else: fall thru to MPI_CHAR
    case MPI_CHAR:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_char;
        case MPI_MIN:  return CkReduction::min_char;
        case MPI_SUM:  return CkReduction::sum_char;
        case MPI_PROD: return CkReduction::product_char;
        default:       break;
      }
    case MPI_INT16_T:
      if (getDDT()->getSize(MPI_INT16_T) != getDDT()->getSize(MPI_SHORT)) break;
      // else: fall thru to MPI_SHORT
    case MPI_SHORT:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_short;
        case MPI_MIN:  return CkReduction::min_short;
        case MPI_SUM:  return CkReduction::sum_short;
        case MPI_PROD: return CkReduction::product_short;
        default:       break;
      }
    case MPI_LONG:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_long;
        case MPI_MIN:  return CkReduction::min_long;
        case MPI_SUM:  return CkReduction::sum_long;
        case MPI_PROD: return CkReduction::product_long;
        default:       break;
      }
    case MPI_INT64_T:
      if (getDDT()->getSize(MPI_INT64_T) != getDDT()->getSize(MPI_LONG_LONG)) break;
      // else: fall thru to MPI_LONG_LONG
    case MPI_LONG_LONG:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_long_long;
        case MPI_MIN:  return CkReduction::min_long_long;
        case MPI_SUM:  return CkReduction::sum_long_long;
        case MPI_PROD: return CkReduction::product_long_long;
        default:       break;
      }
    case MPI_UINT8_T:
      if (getDDT()->getSize(MPI_UINT8_T) != getDDT()->getSize(MPI_UNSIGNED_CHAR)) break;
      // else: fall thru to MPI_UNSIGNED_CHAR
    case MPI_UNSIGNED_CHAR:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_uchar;
        case MPI_MIN:  return CkReduction::min_uchar;
        case MPI_SUM:  return CkReduction::sum_uchar;
        case MPI_PROD: return CkReduction::product_uchar;
        default:       break;
      }
    case MPI_UINT16_T:
      if (getDDT()->getSize(MPI_UINT16_T) != getDDT()->getSize(MPI_UNSIGNED_SHORT)) break;
      // else: fall thru to MPI_UNSIGNED_SHORT
    case MPI_UNSIGNED_SHORT:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_ushort;
        case MPI_MIN:  return CkReduction::min_ushort;
        case MPI_SUM:  return CkReduction::sum_ushort;
        case MPI_PROD: return CkReduction::product_ushort;
        default:       break;
      }
    case MPI_UINT32_T:
      if (getDDT()->getSize(MPI_UINT32_T) != getDDT()->getSize(MPI_UNSIGNED)) break;
      // else: fall thru to MPI_UNSIGNED
    case MPI_UNSIGNED:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_uint;
        case MPI_MIN:  return CkReduction::min_uint;
        case MPI_SUM:  return CkReduction::sum_uint;
        case MPI_PROD: return CkReduction::product_uint;
        default:       break;
      }
    case MPI_UNSIGNED_LONG:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_ulong;
        case MPI_MIN:  return CkReduction::min_ulong;
        case MPI_SUM:  return CkReduction::sum_ulong;
        case MPI_PROD: return CkReduction::product_ulong;
        default:       break;
      }
    case MPI_UINT64_T:
      if (getDDT()->getSize(MPI_UINT64_T) != getDDT()->getSize(MPI_UNSIGNED_LONG_LONG)) break;
      // else: fall thru to MPI_UNSIGNED_LONG_LONG
    case MPI_UNSIGNED_LONG_LONG:
      switch (op) {
        case MPI_MAX:  return CkReduction::max_ulong_long;
        case MPI_MIN:  return CkReduction::min_ulong_long;
        case MPI_SUM:  return CkReduction::sum_ulong_long;
        case MPI_PROD: return CkReduction::product_ulong_long;
        default:       break;
      }
    case MPI_C_BOOL:
      switch (op) {
        case MPI_LAND: return CkReduction::logical_and_bool;
        case MPI_LOR:  return CkReduction::logical_or_bool;
        case MPI_LXOR: return CkReduction::logical_xor_bool;
        default:       break;
      }
    case MPI_LOGICAL:
      switch (op) {
        case MPI_LAND: return CkReduction::logical_and_int;
        case MPI_LOR:  return CkReduction::logical_or_int;
        case MPI_LXOR: return CkReduction::logical_xor_int;
        default:       break;
      }
    case MPI_BYTE:
      switch (op) {
        case MPI_BAND: return CkReduction::bitvec_and_bool;
        case MPI_BOR:  return CkReduction::bitvec_or_bool;
        case MPI_BXOR: return CkReduction::bitvec_xor_bool;
        default:       break;
      }
    default:
      break;
  }
  return CkReduction::invalid;
}

class Builtin_kvs{
 public:
  int tag_ub,host,io,wtime_is_global,appnum,universe_size;
  void* win_base;
  int win_disp_unit,win_create_flavor,win_model;
  MPI_Aint win_size;
  int ampi_tmp;
  Builtin_kvs(){
    tag_ub = MPI_TAG_UB_VALUE;
    host = MPI_PROC_NULL;
    io = 0;
    wtime_is_global = 0;
    appnum = 0;
    universe_size = 0;
    win_base = NULL;
    win_size = 0;
    win_disp_unit = 0;
    win_create_flavor = MPI_WIN_FLAVOR_CREATE;
    win_model = MPI_WIN_SEPARATE;
    ampi_tmp = 0;
  }
};

// ------------ startup support -----------
int _ampi_fallback_setup_count;
CDECL void AMPI_Setup(void);
FDECL void FTN_NAME(AMPI_SETUP,ampi_setup)(void);

FDECL void FTN_NAME(MPI_MAIN,mpi_main)(void);

/*Main routine used when missing MPI_Setup routine*/
CDECL
void AMPI_Fallback_Main(int argc,char **argv)
{
  AMPI_Main_cpp();
  AMPI_Main_cpp(argc,argv);
  AMPI_Main_c(argc,argv);
  FTN_NAME(MPI_MAIN,mpi_main)();
}

void ampiCreateMain(MPI_MainFn mainFn, const char *name,int nameLen);
/*Startup routine used if user *doesn't* write
  a TCHARM_User_setup routine.
 */
CDECL
void AMPI_Setup_Switch(void) {
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
CtvDeclare(void*,stackBottom);
CtvDeclare(int, ampiFinalized);
CkpvDeclare(Builtin_kvs, bikvs);

CDECL
long ampiCurrentStackUsage(void){
  int localVariable;

  unsigned long p1 =  (unsigned long)((void*)&localVariable);
  unsigned long p2 =  (unsigned long)(CtvAccess(stackBottom));

  if(p1 > p2)
    return p1 - p2;
  else
    return  p2 - p1;
}

FDECL
void FTN_NAME(AMPICURRENTSTACKUSAGE, ampicurrentstackusage)(void){
  long usage = ampiCurrentStackUsage();
  CkPrintf("[%d] Stack usage is currently %ld\n", CkMyPe(), usage);
}

CDECL
void AMPI_threadstart(void *data);
static int AMPI_threadstart_idx = -1;

static void ampiNodeInit(void)
{
  _mpi_nworlds=0;
  for(int i=0;i<MPI_MAX_COMM_WORLDS; i++)
  {
    MPI_COMM_UNIVERSE[i] = MPI_COMM_WORLD+1+i;
  }
  TCHARM_Set_fallback_setup(AMPI_Setup_Switch);

  AmpiReducer = CkReduction::addReducer(AmpiReducerFunc);

  CkAssert(AMPI_threadstart_idx == -1);    // only initialize once
  AMPI_threadstart_idx = TCHARM_Register_thread_function(AMPI_threadstart);

  nodeinit_has_been_called=1;

   // ASSUME NO ANYTIME MIGRATION and STATIC INSERTON
  _isAnytimeMigration = false;
  _isStaticInsertion = true;
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

static void ampiProcInit(void){
  CtvInitialize(ampiParent*, ampiPtr);
  CtvInitialize(int,ampiInitDone);
  CtvInitialize(int,ampiFinalized);
  CtvInitialize(void*,stackBottom);

  CkpvInitialize(Builtin_kvs, bikvs); // built-in key-values
  CkpvAccess(bikvs) = Builtin_kvs();

#if CMK_TRACE_ENABLED && CMK_PROJECTOR
  REGISTER_AMPI
#endif
  initAmpiProjections();

#if AMPIMSGLOG
  char **argv=CkGetArgv();
  msgLogWrite = CmiGetArgFlag(argv, "+msgLogWrite");
  if (CmiGetArgIntDesc(argv,"+msgLogRead", &msgLogRank, "Re-play message processing order for AMPI")) {
    msgLogRead = 1;
  }
  char *procs = NULL;
  if (CmiGetArgStringDesc(argv, "+msgLogRanks", &procs, "A list of AMPI processors to record , e.g. 0,10,20-30")) {
    msgLogRanks.set(procs);
  }
  CmiGetArgString(argv, "+msgLogFilename", &msgLogFilename);
  if (CkMyPe() == 0) {
    if (msgLogWrite) CkPrintf("Writing AMPI messages of rank %s to log: %s\n", procs?procs:"", msgLogFilename);
    if (msgLogRead) CkPrintf("Reading AMPI messages of rank %s from log: %s\n", procs?procs:"", msgLogFilename);
  }
#endif
}

#if AMPIMSGLOG
static inline int record_msglog(int rank){
  return msgLogRanks.includes(rank);
}
#endif

PUPfunctionpointer(MPI_MainFn)

class MPI_threadstart_t {
 public:
  MPI_MainFn fn;
  MPI_threadstart_t() {}
  MPI_threadstart_t(MPI_MainFn fn_):fn(fn_) {}
  void start(void) {
    char **argv=CmiCopyArgs(CkGetArgv());
    int argc=CkGetArgc();

    // Set a pointer to somewhere close to the bottom of the stack.
    // This is used for roughly estimating the stack usage later.
    CtvAccess(stackBottom) = &argv;

#if CMK_AMPI_FNPTR_HACK
    AMPI_Fallback_Main(argc,argv);
#else
    (fn)(argc,argv);
#endif
  }
  void pup(PUP::er &p) {
    p|fn;
  }
};
PUPmarshall(MPI_threadstart_t)

CDECL
void AMPI_threadstart(void *data)
{
  STARTUP_DEBUG("MPI_threadstart")
  MPI_threadstart_t t;
  pupFromBuf(data,t);
#if CMK_TRACE_IN_CHARM
  if(CpvAccess(traceOn)) CthTraceResume(CthSelf());
#endif
  t.start();
}

void ampiCreateMain(MPI_MainFn mainFn, const char *name,int nameLen)
{
  STARTUP_DEBUG("ampiCreateMain")
  int _nchunks=TCHARM_Get_num_chunks();
  //Make a new threads array:
  MPI_threadstart_t s(mainFn);
  memBuf b; pupIntoBuf(b,s);
  TCHARM_Create_data(_nchunks,AMPI_threadstart_idx,
                     b.getData(), b.getSize());
}

/* TCharm Semaphore ID's for AMPI startup */
#define AMPI_TCHARM_SEMAID 0x00A34100 /* __AMPI__ */
#define AMPI_BARRIER_SEMAID 0x00A34200 /* __AMPI__ */

static CProxy_ampiWorlds ampiWorldsGroup;

void ampiParent::initOps(void)
{
  ops.resize(MPI_NO_OP+1);
  ops[MPI_MAX]     = OpStruct(MPI_MAX_USER_FN);
  ops[MPI_MIN]     = OpStruct(MPI_MIN_USER_FN);
  ops[MPI_SUM]     = OpStruct(MPI_SUM_USER_FN);
  ops[MPI_PROD]    = OpStruct(MPI_PROD_USER_FN);
  ops[MPI_LAND]    = OpStruct(MPI_LAND_USER_FN);
  ops[MPI_BAND]    = OpStruct(MPI_BAND_USER_FN);
  ops[MPI_LOR]     = OpStruct(MPI_LOR_USER_FN);
  ops[MPI_BOR]     = OpStruct(MPI_BOR_USER_FN);
  ops[MPI_LXOR]    = OpStruct(MPI_LXOR_USER_FN);
  ops[MPI_BXOR]    = OpStruct(MPI_BXOR_USER_FN);
  ops[MPI_MAXLOC]  = OpStruct(MPI_MAXLOC_USER_FN);
  ops[MPI_MINLOC]  = OpStruct(MPI_MINLOC_USER_FN);
  ops[MPI_REPLACE] = OpStruct(MPI_REPLACE_USER_FN);
  ops[MPI_NO_OP]   = OpStruct(MPI_NO_OP_USER_FN);
}

/*
   Called from MPI_Init, a collective initialization call:
   creates a new AMPI array and attaches it to the current
   set of TCHARM threads.
 */
static ampi *ampiInit(char **argv)
{
  FUNCCALL_DEBUG(CkPrintf("Calling from proc %d for tcharm element %d\n", CkMyPe(), TCHARM_Element());)
  if (CtvAccess(ampiInitDone)) return NULL; /* Already called ampiInit */
  STARTUP_DEBUG("ampiInit> begin")

  MPI_Comm new_world;
  int _nchunks;
  CkArrayOptions opts;
  CProxy_ampiParent parent;
  if (TCHARM_Element()==0) //the rank of a tcharm object
  { /* I'm responsible for building the arrays: */
    STARTUP_DEBUG("ampiInit> creating arrays")

    // FIXME: Need to serialize global communicator allocation in one place.
    //Allocate the next communicator
    if(_mpi_nworlds == MPI_MAX_COMM_WORLDS)
    {
      CkAbort("AMPI> Number of registered comm_worlds exceeded limit.\n");
    }
    int new_idx=_mpi_nworlds;
    new_world=MPI_COMM_WORLD+new_idx;

    //Create and attach the ampiParent array
    CkArrayID threads;
    opts=TCHARM_Attach_start(&threads,&_nchunks);
    parent=CProxy_ampiParent::ckNew(new_world,threads,opts);
    STARTUP_DEBUG("ampiInit> array size "<<_nchunks);
  }
  int *barrier = (int *)TCharm::get()->semaGet(AMPI_BARRIER_SEMAID);

  FUNCCALL_DEBUG(CkPrintf("After BARRIER: sema size %d from tcharm's ele %d\n", TCharm::get()->sema.size(), TCHARM_Element());)

  if (TCHARM_Element()==0)
  {
    //Make a new ampi array
    CkArrayID empty;

    ampiCommStruct worldComm(new_world,empty,_nchunks);
    CProxy_ampi arr;
    arr=CProxy_ampi::ckNew(parent,worldComm,opts);

    //Broadcast info. to the mpi_worlds array
    // FIXME: remove race condition from MPI_COMM_UNIVERSE broadcast
    ampiCommStruct newComm(new_world,arr,_nchunks);
    if (ampiWorldsGroup.ckGetGroupID().isZero())
      ampiWorldsGroup=CProxy_ampiWorlds::ckNew(newComm);
    else
      ampiWorldsGroup.add(newComm);
    STARTUP_DEBUG("ampiInit> arrays created")
  }

  // Find our ampi object:
  ampi *ptr=(ampi *)TCharm::get()->semaGet(AMPI_TCHARM_SEMAID);
  CtvAccess(ampiInitDone)=1;
  CtvAccess(ampiFinalized)=0;
  STARTUP_DEBUG("ampiInit> complete")
#if CMK_BIGSIM_CHARM
    //  TRACE_BG_AMPI_START(ptr->getThread(), "AMPI_START");
    TRACE_BG_ADD_TAG("AMPI_START");
#endif

  getAmpiParent()->initOps(); // initialize reduction operations
  getAmpiParent()->setCommAttr(MPI_COMM_WORLD, MPI_UNIVERSE_SIZE, &_nchunks);
  ptr->setCommName("MPI_COMM_WORLD");

  getAmpiParent()->ampiInitCallDone = 0;

  CProxy_ampi cbproxy = ptr->getProxy();
  CkCallback cb(CkReductionTarget(ampi, allInitDone), cbproxy[0]);
  ptr->contribute(cb);

  ampiParent *thisParent = getAmpiParent();
  while(thisParent->ampiInitCallDone!=1){
    thisParent->getTCharmThread()->stop();
    /*
     * thisParent needs to be updated in case of the parent is being pupped.
     * In such case, thisParent got changed
     */
    thisParent = getAmpiParent();
  }

#if CMK_BIGSIM_CHARM
  BgSetStartOutOfCore();
#endif

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
  void pup(PUP::er &p)  { }
  void add(const ampiCommStruct &nextWorld) {
    int new_idx=nextWorld.getComm()-(MPI_COMM_WORLD);
    mpi_worlds[new_idx]=nextWorld;
    if (_mpi_nworlds<=new_idx) _mpi_nworlds=new_idx+1;
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
  userAboutToMigrateFn=NULL;
  userJustMigratedFn=NULL;
  myDDT=&myDDTsto;
  prepareCtv();

  init();

  thread->semaPut(AMPI_BARRIER_SEMAID,&barrier);
  AsyncEvacuate(false);
}

ampiParent::ampiParent(CkMigrateMessage *msg):CBase_ampiParent(msg) {
  thread=NULL;
  worldPtr=NULL;
  myDDT=&myDDTsto;

  init();

  AsyncEvacuate(false);
}

PUPfunctionpointer(MPI_MigrateFn)

void ampiParent::pup(PUP::er &p) {
  p|threads;
  p|worldNo;
  p|worldStruct;
  myDDT->pup(p);
  p|splitComm;
  p|groupComm;
  p|cartComm;
  p|graphComm;
  p|interComm;
  p|intraComm;

  p|groups;
  p|winStructList;
  p|infos;
  p|ops;

  p|ampiReqs;

  p|kvlist;
  p|RProxyCnt;
  p|tmpRProxy;

  p|userAboutToMigrateFn;
  p|userJustMigratedFn;

  p|ampiInitCallDone;
}

void ampiParent::prepareCtv(void) {
  thread=threads[thisIndex].ckLocal();
  if (thread==NULL) CkAbort("AMPIParent cannot find its thread!\n");
  CtvAccessOther(thread->getThread(),ampiPtr) = this;
  STARTUP_DEBUG("ampiParent> found TCharm")
}

void ampiParent::init(){
  CkAssert(groups.size() == 0);
  groups.push_back(new groupStruct);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(thisIndex)){
    char fname[128];
    sprintf(fname, "%s.%d", msgLogFilename,thisIndex);
#if CMK_PROJECTIONS_USE_ZLIB && 0
    fMsgLog = gzopen(fname,"wb");
    toPUPer = new PUP::tozDisk(fMsgLog);
#else
    fMsgLog = fopen(fname,"wb");
    CkAssert(fMsgLog != NULL);
    toPUPer = new PUP::toDisk(fMsgLog);
#endif
  }else if(msgLogRead){
    char fname[128];
    sprintf(fname, "%s.%d", msgLogFilename,msgLogRank);
#if CMK_PROJECTIONS_USE_ZLIB && 0
    fMsgLog = gzopen(fname,"rb");
    fromPUPer = new PUP::fromzDisk(fMsgLog);
#else
    fMsgLog = fopen(fname,"rb");
    CkAssert(fMsgLog != NULL);
    fromPUPer = new PUP::fromDisk(fMsgLog);
#endif
    CkPrintf("AMPI> opened message log file: %s for replay\n", fname);
  }
#endif
}

void ampiParent::finalize(){
#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(thisIndex)){
    delete toPUPer;
#if CMK_PROJECTIONS_USE_ZLIB && 0
    gzclose(fMsgLog);
#else
    fclose(fMsgLog);
#endif
  }else if(msgLogRead){
    delete fromPUPer;
#if CMK_PROJECTIONS_USE_ZLIB && 0
    gzclose(fMsgLog);
#else
    fclose(fMsgLog);
#endif
  }
#endif
}

void ampiParent::setUserAboutToMigrateFn(MPI_MigrateFn f) {
  userAboutToMigrateFn = f;
}

void ampiParent::setUserJustMigratedFn(MPI_MigrateFn f) {
  userJustMigratedFn = f;
}

void ampiParent::ckAboutToMigrate(void) {
  if (userAboutToMigrateFn) {
    (*userAboutToMigrateFn)();
  }
}

void ampiParent::ckJustMigrated(void) {
  ArrayElement1D::ckJustMigrated();
  prepareCtv();
  if (userJustMigratedFn) {
    (*userJustMigratedFn)();
  }
}

void ampiParent::ckJustRestored(void) {
  FUNCCALL_DEBUG(CkPrintf("Call just restored from ampiParent[%d] with ampiInitCallDone %d\n", thisIndex, ampiInitCallDone);)
  ArrayElement1D::ckJustRestored();
  prepareCtv();
}

ampiParent::~ampiParent() {
  STARTUP_DEBUG("ampiParent> destructor called");
  finalize();
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
    vector<int> _indices;
    _indices.push_back(thisIndex);
    selfStruct = ampiCommStruct(MPI_COMM_SELF,s.getProxy(),1,_indices);
    selfStruct.setName("MPI_COMM_SELF");
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
  const char *dname;
  ampiParent *ampiPtr;
  ckptClientStruct(const char *s, ampiParent *a): dname(s), ampiPtr(a) {}
};

static void checkpointClient(void *param,void *msg)
{
  ckptClientStruct *client = (ckptClientStruct*)param;
  const char *dname = client->dname;
  ampiParent *ampiPtr = client->ampiPtr;
  ampiPtr->Checkpoint(strlen(dname), dname);
  delete client;
}

void ampiParent::startCheckpoint(const char* dname){
  if (thisIndex==0) {
    ckptClientStruct *clientData = new ckptClientStruct(dname, this);
    CkCallback *cb = new CkCallback(checkpointClient, clientData);
    thisProxy.ckSetReductionClient(cb);
  }
  contribute();

  thread->stop();

#if CMK_BIGSIM_CHARM
  TRACE_BG_ADD_TAG("CHECKPOINT_RESUME");
#endif
}

void ampiParent::Checkpoint(int len, const char* dname){
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

int ampiParent::createKeyval(MPI_Comm_copy_attr_function *copy_fn, MPI_Comm_delete_attr_function *delete_fn,
                             int *keyval, void* extra_state){
  KeyvalNode* newnode = new KeyvalNode(copy_fn, delete_fn, extra_state);
  int idx = kvlist.size();
  kvlist.resize(idx+1);
  kvlist[idx] = newnode;
  *keyval = idx;
  return 0;
}

int ampiParent::freeKeyval(int *keyval){
#if AMPI_ERROR_CHECKING
  if(*keyval<0 || *keyval >= kvlist.size() || !kvlist[*keyval])
    return MPI_ERR_KEYVAL;
#endif
  delete kvlist[*keyval];
  kvlist[*keyval] = NULL;
  *keyval = MPI_KEYVAL_INVALID;
  return MPI_SUCCESS;
}

int ampiParent::setUserKeyval(MPI_Comm comm, int keyval, void *attribute_val){
#if AMPI_ERROR_CHECKING
  if(keyval<0 || keyval >= kvlist.size() || (kvlist[keyval]==NULL))
    return MPI_ERR_KEYVAL;
#endif
  ampiCommStruct &cs = *(ampiCommStruct *)&comm2CommStruct(comm);
  // Enlarge the keyval list:
  if(cs.getKeyvals().size()<=keyval) cs.getKeyvals().resize(keyval+1, NULL);
  cs.getKeyvals()[keyval]=attribute_val;
  return MPI_SUCCESS;
}

int ampiParent::setWinAttr(MPI_Win win, int keyval, void* attribute_val){
  if(kv_set_builtin(keyval,attribute_val))
    return MPI_SUCCESS;
  MPI_Comm comm = (getAmpiParent()->getWinStruct(win)).comm;
  return setUserKeyval(comm, keyval, attribute_val);
}

int ampiParent::setCommAttr(MPI_Comm comm, int keyval, void* attribute_val){
  if(kv_set_builtin(keyval,attribute_val))
    return MPI_SUCCESS;
  return setUserKeyval(comm, keyval, attribute_val);
}

bool ampiParent::kv_set_builtin(int keyval, void* attribute_val) {
  switch(keyval) {
    case MPI_TAG_UB:            /*immutable*/ return false;
    case MPI_HOST:              /*immutable*/ return false;
    case MPI_IO:                /*immutable*/ return false;
    case MPI_WTIME_IS_GLOBAL:   /*immutable*/ return false;
    case MPI_APPNUM:            /*immutable*/ return false;
    case MPI_UNIVERSE_SIZE:     (CkpvAccess(bikvs).universe_size)     = *((int*)attribute_val);      return true;
    case MPI_WIN_BASE:          (CkpvAccess(bikvs).win_base)          = attribute_val;               return true;
    case MPI_WIN_SIZE:          (CkpvAccess(bikvs).win_size)          = *((MPI_Aint*)attribute_val); return true;
    case MPI_WIN_DISP_UNIT:     (CkpvAccess(bikvs).win_disp_unit)     = *((int*)attribute_val);      return true;
    case MPI_WIN_CREATE_FLAVOR: (CkpvAccess(bikvs).win_create_flavor) = *((int*)attribute_val);      return true;
    case MPI_WIN_MODEL:         (CkpvAccess(bikvs).win_model)         = *((int*)attribute_val);      return true;
    case AMPI_MY_PE:            /*immutable*/ return false;
    case AMPI_NUM_PES:          /*immutable*/ return false;
    case AMPI_MY_NODE:          /*immutable*/ return false;
    case AMPI_NUM_NODES:        /*immutable*/ return false;
    default: return false;
  };
}

bool ampiParent::kv_get_builtin(int keyval) {
  switch(keyval) {
    case MPI_TAG_UB:            kv_builtin_storage = &(CkpvAccess(bikvs).tag_ub);            return true;
    case MPI_HOST:              kv_builtin_storage = &(CkpvAccess(bikvs).host);              return true;
    case MPI_IO:                kv_builtin_storage = &(CkpvAccess(bikvs).io);                return true;
    case MPI_WTIME_IS_GLOBAL:   kv_builtin_storage = &(CkpvAccess(bikvs).wtime_is_global);   return true;
    case MPI_APPNUM:            kv_builtin_storage = &(CkpvAccess(bikvs).appnum);            return true;
    case MPI_UNIVERSE_SIZE:     kv_builtin_storage = &(CkpvAccess(bikvs).universe_size);     return true;
    case MPI_WIN_BASE:          win_base_storage   = &(CkpvAccess(bikvs).win_base);          return true;
    case MPI_WIN_SIZE:          win_size_storage   = &(CkpvAccess(bikvs).win_size);          return true;
    case MPI_WIN_DISP_UNIT:     kv_builtin_storage = &(CkpvAccess(bikvs).win_disp_unit);     return true;
    case MPI_WIN_CREATE_FLAVOR: kv_builtin_storage = &(CkpvAccess(bikvs).win_create_flavor); return true;
    case MPI_WIN_MODEL:         kv_builtin_storage = &(CkpvAccess(bikvs).win_model);         return true;
    case AMPI_MY_PE:     CkpvAccess(bikvs).ampi_tmp = CkMyPe();     kv_builtin_storage = &(CkpvAccess(bikvs).ampi_tmp); return true;
    case AMPI_NUM_PES:   CkpvAccess(bikvs).ampi_tmp = CkNumPes();   kv_builtin_storage = &(CkpvAccess(bikvs).ampi_tmp); return true;
    case AMPI_MY_NODE:   CkpvAccess(bikvs).ampi_tmp = CkMyNode();   kv_builtin_storage = &(CkpvAccess(bikvs).ampi_tmp); return true;
    case AMPI_NUM_NODES: CkpvAccess(bikvs).ampi_tmp = CkNumNodes(); kv_builtin_storage = &(CkpvAccess(bikvs).ampi_tmp); return true;
    default: return false;
  };
}

bool ampiParent::getBuiltinKeyval(int keyval, void *attribute_val) {
  if (kv_get_builtin(keyval)){
    /* All builtin keyvals are ints except MPI_WIN_BASE, which is a pointer
     * to the window's base address in C but an integer representation of
     * the base address in Fortran.
     * Also, MPI_WIN_SIZE is an MPI_Aint. */
    if (keyval == MPI_WIN_BASE)
      *((void**)attribute_val) = *win_base_storage;
    else if (keyval == MPI_WIN_SIZE)
      *(MPI_Aint**)attribute_val = win_size_storage;
    else
      *(int **)attribute_val = kv_builtin_storage;
    return true;
  }
  return false;
}

bool ampiParent::getUserKeyval(MPI_Comm comm, int keyval, void *attribute_val, int *flag) {
  *flag = false;
  if (keyval<0 || keyval >= kvlist.size() || (kvlist[keyval]==NULL))
    return false;
  ampiCommStruct &cs=*(ampiCommStruct *)&comm2CommStruct(comm);
  if (keyval>=cs.getKeyvals().size())
    return true; /* we don't have a value yet */
  if (cs.getKeyvals()[keyval]==NULL)
    return true; /* we had a value, but now it's NULL */
  /* Otherwise, we have a good value */
  *flag = true;
  *(void **)attribute_val = cs.getKeyvals()[keyval];
  return true;
}

int ampiParent::getCommAttr(MPI_Comm comm, int keyval, void *attribute_val, int *flag) {
  *flag = false;
  if (getBuiltinKeyval(keyval, attribute_val)) {
    *flag = true;
    return MPI_SUCCESS;
  }
  if (getUserKeyval(comm, keyval, attribute_val, flag))
    return MPI_SUCCESS;
  return MPI_ERR_KEYVAL;
}

int ampiParent::getWinAttr(MPI_Win win, int keyval, void *attribute_val, int *flag) {
  *flag = false;
  if (getBuiltinKeyval(keyval, attribute_val)) {
    *flag = true;
    return MPI_SUCCESS;
  }
  MPI_Comm comm = (getAmpiParent()->getWinStruct(win)).comm;
  if (getUserKeyval(comm, keyval, attribute_val, flag))
    return MPI_SUCCESS;
  return MPI_ERR_KEYVAL;
}

int ampiParent::deleteCommAttr(MPI_Comm comm, int keyval){
  /* no way to delete an attribute: just overwrite it with NULL */
  return setUserKeyval(comm, keyval, NULL);
}

int ampiParent::deleteWinAttr(MPI_Win win, int keyval){
  /* no way to delete an attribute: just overwrite it with NULL */
  MPI_Comm comm = (getAmpiParent()->getWinStruct(win)).comm;
  return setUserKeyval(comm, keyval, NULL);
}

//----------------------- ampi -------------------------
void ampi::init(void) {
  parent=NULL;
  thread=NULL;
  msgs=NULL;
  posted_ireqs=NULL;
  resumeOnRecv=false;
  resumeOnColl=false;
  blockingReq=NULL;
  AsyncEvacuate(false);
}

ampi::ampi()
{
  /* this constructor only exists so we can create an empty array during split */
  CkAbort("Default ampi constructor should never be called");
}

ampi::ampi(CkArrayID parent_,const ampiCommStruct &s):parentProxy(parent_)
{
  init();

  myComm=s; myComm.setArrayID(thisArrayID);
  myRank=myComm.getRankForIndex(thisIndex);

  findParent(false);

  msgs = CmmNew();
  posted_ireqs = CmmNew();

  seqEntries=parent->ckGetArraySize();
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

void ampi::ckJustRestored(void)
{
  FUNCCALL_DEBUG(CkPrintf("Call just restored from ampi[%d]\n", thisIndex);)
  findParent(true);
  ArrayElement1D::ckJustRestored();
}

void ampi::findParent(bool forMigration) {
  STARTUP_DEBUG("ampi> finding my parent")
  parent=parentProxy[thisIndex].ckLocal();
  if (parent==NULL) CkAbort("AMPI can't find its parent!");
  thread=parent->registerAmpi(this,myComm,forMigration);
  if (thread==NULL) CkAbort("AMPI can't find its thread!");
}

//The following method should be called on the first element of the
//ampi array
void ampi::allInitDone(){
  FUNCCALL_DEBUG(CkPrintf("All mpi_init have been called!\n");)
  thisProxy.setInitDoneFlag();
}

void ampi::setInitDoneFlag(){
  parent->ampiInitCallDone=1;
  parent->getTCharmThread()->start();
}

static void cmm_pup_ampi_message(pup_er p,void **msg) {
  CkPupMessage(*(PUP::er *)p,msg,1);
  if (pup_isDeleting(p)) delete (AmpiMsg *)*msg;
}

static void cmm_pup_posted_ireq(pup_er p,void **msg) {
  pup_int(p, (int *)msg);
}

void ampi::pup(PUP::er &p)
{
  p|parentProxy;
  p|myComm;
  p|myRank;
  p|tmpVec;
  p|remoteProxy;
  p|resumeOnRecv;
  p|resumeOnColl;

  // pup blockingReq
  char nonnull;
  if (!p.isUnpacking()) {
    if (blockingReq) {
      nonnull = blockingReq->getType();
    } else {
      nonnull = 0;
    }
  }
  p(nonnull);
  if (nonnull != 0) {
    if (p.isUnpacking()) {
      switch (nonnull) {
        case MPI_PERS_REQ:
          blockingReq = new PersReq;
          break;
        case MPI_I_REQ:
          blockingReq = new IReq;
          break;
        case MPI_REDN_REQ:
          blockingReq = new RednReq;
          break;
        case MPI_GATHER_REQ:
          blockingReq = new GatherReq;
          break;
        case MPI_GATHERV_REQ:
          blockingReq = new GathervReq;
          break;
        case MPI_SEND_REQ:
          blockingReq = new SendReq;
          break;
        case MPI_SSEND_REQ:
          blockingReq = new SsendReq;
          break;
        case MPI_IATA_REQ:
          blockingReq = new IATAReq;
          break;
      }
    }
    blockingReq->pup(p);
  } else {
    blockingReq = NULL;
  }
  if (p.isDeleting()) {
    delete blockingReq; blockingReq = NULL;
  }

  msgs=CmmPup((pup_er)&p,msgs,cmm_pup_ampi_message);

  posted_ireqs = CmmPup((pup_er)&p, posted_ireqs, cmm_pup_posted_ireq);

  p|seqEntries;
  p|oorder;
}

ampi::~ampi()
{
  if (CkInRestarting() || _BgOutOfCoreFlag==1) {
    // in restarting, we need to flush messages
    int tags[3];
    MPI_Status sts;
    tags[0] = tags[1] = tags[2] = CmmWildCard;
    AmpiMsg *msg = (AmpiMsg *) CmmGet(msgs, 3, tags, (int*)&sts);
    while (msg) {
      delete msg;
      msg = (AmpiMsg *) CmmGet(msgs, 3, tags, (int*)&sts);
    }
  }

  delete blockingReq; blockingReq = NULL;
  CmmFree(msgs);
  CmmFreeAll(posted_ireqs);
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
#if CMK_BIGSIM_CHARM
  void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#endif
  if (type == MPI_CART) {
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
#if CMK_BIGSIM_CHARM
  _TRACE_BG_SET_INFO(NULL, "SPLIT_RESUME", NULL, 0);
#endif
}

CDECL
int compareAmpiSplitKey(const void *a_, const void *b_) {
  const ampiSplitKey *a=(const ampiSplitKey *)a_;
  const ampiSplitKey *b=(const ampiSplitKey *)b_;
  if (a->color!=b->color) return a->color-b->color;
  if (a->key!=b->key) return a->key-b->key;
  return a->rank-b->rank;
}

CProxy_ampi ampi::createNewChildAmpiSync() {
  CkArrayOptions opts;
  opts.bindTo(parentProxy);
  opts.setNumInitial(0);
  CkArrayID unusedAID;
  ampiCommStruct unusedComm;
  CkCallback cb(CkCallback::resumeThread);
  CProxy_ampi::ckNew(unusedAID, unusedComm, opts, cb);
  CkArrayCreatedMsg *newAmpiMsg = static_cast<CkArrayCreatedMsg*>(cb.thread_delay());
  CProxy_ampi newAmpi = newAmpiMsg->aid;
  delete newAmpiMsg;
  newAmpi.doneInserting(); //<- Meaning, I need to do my own creation race resolution
  return newAmpi;
}

void ampi::splitPhase1(CkReductionMsg *msg)
{
  //Order the keys, which orders the ranks properly:
  int nKeys=msg->getSize()/sizeof(ampiSplitKey);
  ampiSplitKey *keys=(ampiSplitKey *)msg->getData();
  if (nKeys!=myComm.getSize()) CkAbort("ampi::splitReduce expected a split contribution from every rank!");
  qsort(keys,nKeys,sizeof(ampiSplitKey),compareAmpiSplitKey);

  MPI_Comm newComm = -1;
  for(int i=0;i<nKeys;i++){
    if(keys[i].nextSplitComm>newComm)
      newComm = keys[i].nextSplitComm;
  }

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

      lastAmpi = createNewChildAmpiSync();

      vector<int> indices; //Maps rank to array indices for new array
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
  CkCallback cb(CkReductionTarget(ampi,commCreatePhase1),CkArrayIndex1D(rootIdx),myComm.getProxy());
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

void ampi::insertNewChildAmpiElements(MPI_Comm nextComm, CProxy_ampi newAmpi) {
  ampiCommStruct newCommStruct = ampiCommStruct(nextComm, newAmpi, tmpVec.size(), tmpVec);
  for (int i = 0; i < tmpVec.size(); ++i)
    newAmpi[tmpVec[i]].insert(parentProxy, newCommStruct);
}

void ampi::commCreatePhase1(MPI_Comm nextGroupComm){
  CProxy_ampi newAmpi = createNewChildAmpiSync();
  insertNewChildAmpiElements(nextGroupComm, newAmpi);
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
  CkCallback cb(CkReductionTarget(ampi,commCreatePhase1),CkArrayIndex1D(rootIdx),myComm.getProxy());

  MPI_Comm nextcart = parent->getNextCart();
  contribute(sizeof(nextcart), &nextcart,CkReduction::max_int,cb);

  if(getPosOp(thisIndex,vec)>=0){
    thread->suspend(); //Resumed by ampiParent::cartChildRegister
    MPI_Comm retcomm = parent->getNextCart()-1;
    *newcomm = retcomm;
  }else
    *newcomm = MPI_COMM_NULL;
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
  CkCallback cb(CkReductionTarget(ampi,commCreatePhase1),CkArrayIndex1D(rootIdx),
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
  CkCallback cb(CkReductionTarget(ampi, intercommCreatePhase1),CkArrayIndex1D(root),myComm.getProxy());
  MPI_Comm nextinter = parent->getNextInter();
  contribute(sizeof(nextinter), &nextinter,CkReduction::max_int,cb);

  thread->suspend(); //Resumed by ampiParent::interChildRegister
  MPI_Comm newcomm=parent->getNextInter()-1;
  *ncomm=newcomm;
}

void ampi::intercommCreatePhase1(MPI_Comm nextInterComm){

  CProxy_ampi newAmpi = createNewChildAmpiSync();

  groupStruct lgroup = myComm.getIndices();
  ampiCommStruct newCommstruct = ampiCommStruct(nextInterComm,newAmpi,lgroup.size(),lgroup,tmpVec);
  for(int i=0;i<lgroup.size();i++){
    int newIdx=lgroup[i];
    newAmpi[newIdx].insert(parentProxy,newCommstruct);
  }

  parentProxy[0].ExchangeProxy(newAmpi);
}

void ampiParent::interChildRegister(const ampiCommStruct &s) {
  int idx=s.getComm()-MPI_COMM_FIRST_INTER;
  if (interComm.size()<=idx) interComm.resize(idx+1);
  interComm[idx]=new ampiCommStruct(s);
  // don't resume the thread yet, till parent set remote proxy
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
  CkCallback cb(CkReductionTarget(ampi, intercommMergePhase1),CkArrayIndex1D(rootIdx),myComm.getProxy());
  MPI_Comm nextintra = parent->getNextIntra();
  contribute(sizeof(nextintra), &nextintra,CkReduction::max_int,cb);

  thread->suspend(); //Resumed by ampiParent::interChildRegister
  MPI_Comm newcomm=parent->getNextIntra()-1;
  *ncomm=newcomm;
}

void ampi::intercommMergePhase1(MPI_Comm nextIntraComm){
  // gets called on two roots, first root creates the comm
  if(tmpVec.size()==0) return;
  CProxy_ampi newAmpi = createNewChildAmpiSync();
  insertNewChildAmpiElements(nextIntraComm, newAmpi);
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
    if (worldDex>=_mpi_nworlds)
      CkAbort("Bad world communicator passed to universeComm2CommStruct");
    return mpi_worlds[worldDex];
  }
  CkAbort("Bad communicator passed to universeComm2CommStruct");
  return mpi_worlds[0]; // meaningless return
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

ampi* ampi::blockOnRecv(void){
  resumeOnRecv = true;
  // In case this thread is migrated while suspended,
  // save myComm to get the ampi instance back. Then
  // return "dis" in case the caller needs it.
  MPI_Comm comm = myComm.getComm();
  thread->suspend();
  ampi *dis = getAmpiInstance(comm);
  dis->resumeOnRecv = false;
  return dis;
}

ampi* ampi::blockOnColl(void){
  resumeOnColl = true;
  MPI_Comm comm = myComm.getComm();
  thread->suspend();
  ampi *dis = getAmpiInstance(comm);
  dis->resumeOnColl = false;
  return dis;
}

// block on (All)Reduce or (All)Gather(v)
ampi* ampi::blockOnRedn(AmpiRequest *req){

  blockingReq = req;

#if CMK_TRACE_ENABLED && CMK_PROJECTOR
  _LOG_E_END_AMPI_PROCESSING(thisIndex)
#endif
#if CMK_BIGSIM_CHARM
  void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#if CMK_TRACE_IN_CHARM
  if(CpvAccess(traceOn)) traceSuspend();
#endif
#endif

  ampi* dis = blockOnColl();

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  CpvAccess(_currentObj) = dis;
#endif
#if CMK_TRACE_ENABLED && CMK_PROJECTOR
  _LOG_E_BEGIN_AMPI_PROCESSING(thisIndex, dis->blockingReq->src, dis->blockingReq->count)
#endif
#if CMK_BIGSIM_CHARM
#if CMK_TRACE_IN_CHARM
  if(CpvAccess(traceOn)) CthTraceResume(dis->thread->getThread());
#endif
  TRACE_BG_AMPI_BREAK(dis->thread->getThread(), "RECV_RESUME", NULL, 0, 0);
  if (dis->blockingReq->eventPe == CkMyPe()) _TRACE_BG_ADD_BACKWARD_DEP(dis->blockingReq->event);
#endif

  delete dis->blockingReq; dis->blockingReq = NULL;
  return dis;
}

void ampi::ssend_ack(int sreq_idx){
  if (sreq_idx == 1)
    thread->resume();           // MPI_Ssend
  else {
    sreq_idx -= 2;              // start from 2
    AmpiRequestList *reqs = &(parent->ampiReqs);
    SsendReq *sreq = (SsendReq *)(*reqs)[sreq_idx];
    sreq->statusIreq = true;
    if (resumeOnRecv) {
      thread->resume();
    }
  }
}

void ampi::generic(AmpiMsg* msg)
{
  MSG_ORDER_DEBUG(
    CkPrintf("AMPI vp %d arrival: tag=%d, src=%d, comm=%d  (from %d, seq %d) resumeOnRecv %d\n",
             thisIndex,msg->tag,msg->srcRank,msg->comm, msg->srcIdx, msg->seq,resumeOnRecv);
  )
#if CMK_BIGSIM_CHARM
  TRACE_BG_ADD_TAG("AMPI_generic");
  msg->event = NULL;
#endif

  int sync = UsrToEnv(msg)->getRef();
  int srcIdx;
  if (sync)  srcIdx = msg->srcIdx;

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

  // msg may be free'ed from calling inorder()
  if (sync>0) {         // send an ack to sender
    CProxy_ampi pa(thisArrayID);
    pa[srcIdx].ssend_ack(sync);
  }

  if(resumeOnRecv){
    thread->resume();
  }
}

inline static AmpiRequestList *getReqs(void);

void ampi::inorder(AmpiMsg* msg)
{
  MSG_ORDER_DEBUG(
    CkPrintf("AMPI vp %d inorder: tag=%d, src=%d, comm=%d  (from %d, seq %d)\n",
             thisIndex,msg->tag,msg->srcRank,msg->comm, msg->srcIdx, msg->seq);
  )

  // check posted recvs
  int tags[3];
  MPI_Status sts;
  tags[0] = msg->tag; tags[1] = msg->srcRank; tags[2] = msg->comm;

#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&msg->event); // store current log
  msg->eventPe = CkMyPe();
#endif

  //in case ampi has not initialized and posted_ireqs are only inserted
  //at AMPI_Irecv (MPI_Irecv)
  AmpiRequestList *reqL = &(parent->ampiReqs);
  //When storing the req index, it's 1-based. The reason is stated in the comments
  //in the ampi::irecv function.
  int ireqIdx = (int)((long)CmmGet(posted_ireqs, 3, tags, (int*)&sts));
  IReq *ireq = NULL;
  if(reqL->size()>0 && ireqIdx>0)
    ireq = (IReq *)(*reqL)[ireqIdx-1];
  if (ireq) { // receive posted
    ireq->receive(this, msg);
  } else {
    CmmPut(msgs, 3, tags, msg);
  }
}

AmpiMsg *ampi::getMessage(int t, int s, MPI_Comm comm, int *sts) const
{
  int tags[3];
  tags[0] = t; tags[1] = s; tags[2] = comm;
  AmpiMsg *msg = (AmpiMsg *) CmmGet(msgs, 3, tags, sts);
  return msg;
}

void handle_MPI_BOTTOM(void* &buf, MPI_Datatype type)
{
  if (buf == MPI_BOTTOM) {
    buf = (void*)getDDT()->getType(type)->getLB();
    getDDT()->getType(type)->setAbsolute(true);
  }
}

void handle_MPI_BOTTOM(void* &buf1, MPI_Datatype type1, void* &buf2, MPI_Datatype type2)
{
  if (buf1 == MPI_BOTTOM) {
    buf1 = (void*)getDDT()->getType(type1)->getLB();
    getDDT()->getType(type1)->setAbsolute(true);
  }
  if (buf2 == MPI_BOTTOM) {
    buf2 = (void*)getDDT()->getType(type2)->getLB();
    getDDT()->getType(type2)->setAbsolute(true);
  }
}

AmpiMsg *ampi::makeAmpiMsg(int destIdx,int t,int sRank,const void *buf,int count,
                           MPI_Datatype type,MPI_Comm destcomm, int sync)
{
  CkDDT_DataType *ddt = getDDT()->getType(type);
  int len = ddt->getSize(count);
  int sIdx=thisIndex;
  int seq = -1;
  if (destIdx>=0 && destcomm<=MPI_COMM_WORLD && t<=MPI_ATA_SEQ_TAG) //Not cross-module: set seqno
    seq = oorder.nextOutgoing(destIdx);
  AmpiMsg *msg = new (len, 0) AmpiMsg(seq, t, sIdx, sRank, len, destcomm);
  if (sync) UsrToEnv(msg)->setRef(sync);
  ddt->serialize((char*)buf, (char*)msg->data, count, 1);
  return msg;
}

void ampi::send(int t, int sRank, const void* buf, int count, MPI_Datatype type,
                int rank, MPI_Comm destcomm, int sync)
{
#if CMK_TRACE_IN_CHARM
  TRACE_BG_AMPI_BREAK(thread->getThread(), "AMPI_SEND", NULL, 0, 1);
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  MPI_Comm disComm = myComm.getComm();
  ampi *dis = getAmpiInstance(disComm);
  CpvAccess(_currentObj) = dis;
#endif

  const ampiCommStruct &dest=comm2CommStruct(destcomm);
  delesend(t,sRank,buf,count,type,rank,destcomm,dest.getProxy(),sync);

#if CMK_TRACE_IN_CHARM
  TRACE_BG_AMPI_BREAK(thread->getThread(), "AMPI_SEND_END", NULL, 0, 1);
#endif

  if (sync == 1) {
    // waiting for receiver side
    resumeOnRecv = false;            // so no one else awakes it
    block();
  }
}

void ampi::sendraw(int t, int sRank, void* buf, int len, CkArrayID aid, int idx)
{
  AmpiMsg *msg = new (len, 0) AmpiMsg(-1, t, -1, sRank, len, MPI_COMM_WORLD);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa[idx].generic(msg);
}

void ampi::delesend(int t, int sRank, const void* buf, int count, MPI_Datatype type,  int rank,
                    MPI_Comm destcomm, CProxy_ampi arrproxy, int sync)
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

  arrproxy[destIdx].generic(makeAmpiMsg(destIdx,t,sRank,buf,count,type,destcomm,sync));
}

void ampi::processAmpiMsg(AmpiMsg *msg, void* buf, MPI_Datatype type, int count)
{
  CkDDT_DataType *ddt = getDDT()->getType(type);
  int len = ddt->getSize(count);

  if(msg->length < len){ // only at rare case shall we reset count by using divide
    count = msg->length/(ddt->getSize(1));
  }

  ddt->serialize((char*)buf, (char*)msg->data, count, (-1));
}

void ampi::processRednMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type, int count)
{
  // The first sizeof(AmpiOpHeader) bytes in the redn msg data are reserved
  // for an AmpiOpHeader if our custom AmpiReducer type was used.
  int szhdr = (msg->getReducer() == AmpiReducer) ? sizeof(AmpiOpHeader) : 0;
  getDDT()->getType(type)->serialize((char*)buf, (char*)msg->getData()+szhdr, count, (-1));
}

void ampi::processNoncommutativeRednMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type, int count, MPI_User_function* func)
{
  CkReduction::tupleElement* results = NULL;
  int numReductions = 0;
  msg->toTuple(&results, &numReductions);

  // Contributions are unordered and consist of a (srcRank, data) tuple
  CkReduction::setElement *currentSrc  = (CkReduction::setElement*)results[0].data;
  CkReduction::setElement *currentData = (CkReduction::setElement*)results[1].data;
  CkDDT_DataType *ddt  = getDDT()->getType(type);
  int contributionSize = ddt->getSize(count);
  int commSize = getSize(getComm());

  // Store pointers to each contribution's data at index 'srcRank' in contributionData
  vector<void *> contributionData(commSize);
  for (int i=0; i<commSize; i++) {
    CkAssert(currentSrc && currentData);
    int srcRank = *((int*)currentSrc->data);
    CkAssert(currentData->dataSize == contributionSize);
    contributionData[srcRank] = currentData->data;
    currentSrc  = currentSrc->next();
    currentData = currentData->next();
  }

  // Copy rank 0's contribution into buf first
  memcpy(buf, contributionData[0], contributionSize);

  // Invoke the MPI_User_function on the contributions in 'rank' order
  for (int i=1; i<commSize; i++) {
    (*func)(contributionData[i], buf, &count, &type);
  }
}

void ampi::processGatherMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type, int recvCount)
{
  CkReduction::tupleElement* results = NULL;
  int numReductions = 0;
  msg->toTuple(&results, &numReductions);

  // Re-order the gather data based on the rank of the contributor
  CkReduction::setElement *currentSrc  = (CkReduction::setElement*)results[0].data;
  CkReduction::setElement *currentData = (CkReduction::setElement*)results[1].data;
  CkDDT_DataType *ddt    = getDDT()->getType(type);
  int contributionSize   = ddt->getSize(recvCount);
  int contributionExtent = ddt->getExtent()*recvCount;

  for (int i=0; i<getSize(getComm()); i++) {
    CkAssert(currentSrc && currentData);
    int srcRank = *((int*)currentSrc->data);
    CkAssert(currentData->dataSize == contributionSize);
    ddt->serialize(&(((char*)buf)[srcRank*contributionExtent]), currentData->data, recvCount, (-1));
    currentSrc  = currentSrc->next();
    currentData = currentData->next();
  }
}

void ampi::processGathervMsg(CkReductionMsg *msg, void* buf, MPI_Datatype type,
                             int* recvCounts, int* displs)
{
  CkReduction::tupleElement* results = NULL;
  int numReductions = 0;
  msg->toTuple(&results, &numReductions);

  // Re-order the gather data based on the rank of the contributor
  CkReduction::setElement *currentSrc  = (CkReduction::setElement*)results[0].data;
  CkReduction::setElement *currentData = (CkReduction::setElement*)results[1].data;
  CkDDT_DataType *ddt    = getDDT()->getType(type);
  int contributionSize   = ddt->getSize();
  int contributionExtent = ddt->getExtent();

  for (int i=0; i<getSize(getComm()); i++) {
    CkAssert(currentSrc && currentData);
    int srcRank = *((int*)currentSrc->data);
    CkAssert(currentData->dataSize == contributionSize*recvCounts[srcRank]);
    ddt->serialize(&((char*)buf)[displs[srcRank]*contributionExtent], currentData->data, recvCounts[srcRank], (-1));
    currentSrc  = currentSrc->next();
    currentData = currentData->next();
  }
}

int ampi::recv(int t, int s, void* buf, int count, MPI_Datatype type, MPI_Comm comm, MPI_Status *sts)
{
  MPI_Comm disComm = myComm.getComm();
  if(s==MPI_PROC_NULL) {
    sts->MPI_SOURCE = MPI_PROC_NULL;
    sts->MPI_TAG = MPI_ANY_TAG;
    sts->MPI_LENGTH = 0;
    return 0;
  }
#if CMK_TRACE_ENABLED && CMK_PROJECTOR
  _LOG_E_END_AMPI_PROCESSING(thisIndex)
#endif
#if CMK_BIGSIM_CHARM
   void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#if CMK_TRACE_IN_CHARM
  if(CpvAccess(traceOn)) traceSuspend();
#endif
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

  ampi *dis = getAmpiInstance(disComm);
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmGet(dis->msgs, 3, tags, (int*)sts);
    if (msg) break;
    // "dis" is updated in case an ampi thread is migrated while waiting for a message
    dis = dis->blockOnRecv();
  }

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  CpvAccess(_currentObj) = dis;
  MSG_ORDER_DEBUG( printf("[%d] AMPI thread rescheduled  to Index %d buf %p src %d\n",CkMyPe(),dis->thisIndex,buf,s); )
#endif

  if(sts)
    sts->MPI_LENGTH = msg->length;
  dis->processAmpiMsg(msg, buf, type, count);

#if CMK_TRACE_ENABLED && CMK_PROJECTOR
  _LOG_E_BEGIN_AMPI_PROCESSING(thisIndex,s,count)
#endif

#if CMK_BIGSIM_CHARM
#if CMK_TRACE_IN_CHARM
  //Due to the reason mentioned the in the while loop above, we need to 
  //use "dis" as "this" in the case of migration (or out-of-core execution in BigSim)
  if(CpvAccess(traceOn)) CthTraceResume(dis->thread->getThread());
#endif
  TRACE_BG_AMPI_BREAK(thread->getThread(), "RECV_RESUME", NULL, 0, 0);
  if (msg->eventPe == CkMyPe()) _TRACE_BG_ADD_BACKWARD_DEP(msg->event);
#endif

  delete msg;
  return 0;
}

void ampi::probe(int t, int s, MPI_Comm comm, MPI_Status *sts)
{
  int tags[3];
#if CMK_BIGSIM_CHARM
  void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#endif

  ampi *dis = getAmpiInstance(comm);
  AmpiMsg *msg = 0;
  while(1) {
    tags[0] = t; tags[1] = s; tags[2] = comm;
    msg = (AmpiMsg *) CmmProbe(dis->msgs, 3, tags, (int*)sts);
    if (msg) break;
    // "dis" is updated in case an ampi thread is migrated while waiting for a message
    dis = dis->blockOnRecv();
  }

  if(sts)
    sts->MPI_LENGTH = msg->length;

#if CMK_BIGSIM_CHARM
  _TRACE_BG_SET_INFO((char *)msg, "PROBE_RESUME",  &curLog, 1);
#endif
}

int ampi::iprobe(int t, int s, MPI_Comm comm, MPI_Status *sts)
{
  int tags[3];
  AmpiMsg *msg = 0;
  tags[0] = t; tags[1] = s; tags[2] = comm;
  msg = (AmpiMsg *) CmmProbe(msgs, 3, tags, (int*)sts);
  if (msg) {
    if(sts)
      sts->MPI_LENGTH = msg->length;
    return 1;
  }
#if CMK_BIGSIM_CHARM
  void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#endif
  thread->schedule();
#if CMK_BIGSIM_CHARM
  _TRACE_BG_SET_INFO(NULL, "IPROBE_RESUME",  &curLog, 1);
#endif
  return 0;
}


const int MPI_BCAST_COMM=MPI_COMM_WORLD+1000;

void ampi::bcast(int root, void* buf, int count, MPI_Datatype type, MPI_Comm destcomm)
{
  const ampiCommStruct &dest=comm2CommStruct(destcomm);
  int rootIdx=dest.getIndexForRank(root);
  if(rootIdx==thisIndex) {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = this;
#endif
    thisProxy.generic(makeAmpiMsg(-1,MPI_BCAST_TAG,0, buf,count,type, MPI_BCAST_COMM));
  }
  if(-1==recv(MPI_BCAST_TAG,0, buf,count,type, MPI_BCAST_COMM)) CkAbort("AMPI> Error in broadcast");
}

void ampi::ibcast(int root, void* buf, int count, MPI_Datatype type, MPI_Comm destcomm, MPI_Request* request)
{
  const ampiCommStruct &dest=comm2CommStruct(destcomm);
  int rootIdx=dest.getIndexForRank(root);
  if(rootIdx==thisIndex){
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = this;
#endif
    thisProxy.generic(makeAmpiMsg(-1, MPI_BCAST_TAG, 0, buf, count, type, MPI_BCAST_COMM));
  }

  // use an IReq to non-block the caller and get a request ptr
  *request = postReq(new IReq(buf, count, type, rootIdx, MPI_BCAST_TAG, MPI_BCAST_COMM));
}

void ampi::bcastraw(void* buf, int len, CkArrayID aid)
{
  AmpiMsg *msg = new (len, 0) AmpiMsg(-1, MPI_BCAST_TAG, -1, 0, len, MPI_COMM_WORLD);
  memcpy(msg->data, buf, len);
  CProxy_ampi pa(aid);
  pa.generic(msg);
}

AmpiMsg* ampi::Alltoall_RemoteIget(MPI_Aint disp, int cnt, MPI_Datatype type, int tag)
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

int MPI_comm_null_copy_fn(MPI_Comm comm, int keyval, void *extra_state,
                          void *attr_in, void *attr_out, int *flag){
  (*flag) = 0;
  return (MPI_SUCCESS);
}

int MPI_comm_dup_fn(MPI_Comm comm, int keyval, void *extra_state,
                    void *attr_in, void *attr_out, int *flag){
  (*(void **)attr_out) = attr_in;
  (*flag) = 1;
  return (MPI_SUCCESS);
}

int MPI_comm_null_delete_fn(MPI_Comm comm, int keyval, void *attr, void *extra_state){
  return (MPI_SUCCESS);
}

int MPI_type_null_copy_fn(MPI_Datatype type, int keyval, void *extra_state,
                          void *attr_in, void *attr_out, int *flag){
  (*flag) = 0;
  return (MPI_SUCCESS);
}

int MPI_type_dup_fn(MPI_Datatype type, int keyval, void *extra_state,
                    void *attr_in, void *attr_out, int *flag){
  (*(void **)attr_out) = attr_in;
  (*flag) = 1;
  return (MPI_SUCCESS);
}

int MPI_type_null_delete_fn(MPI_Datatype type, int keyval, void *attr, void *extra_state){
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
#if CMK_ERROR_CHECKING
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

void AmpiRequest::print(){
  CkPrintf("In AmpiRequest: buf=%p, count=%d, type=%d, src=%d, tag=%d, comm=%d, isvalid=%d\n", buf, count, type, src, tag, comm, isvalid);
}

void PersReq::print(){
  AmpiRequest::print();
  CkPrintf("In PersReq: sndrcv=%d\n", sndrcv);
}

void IReq::print(){
  AmpiRequest::print();
  CkPrintf("In IReq: this=%p, status=%d, length=%d\n", this, statusIreq, length);
}

void RednReq::print(){
  AmpiRequest::print();
  CkPrintf("In RednReq: this=%p, status=%d\n", this, statusIreq);
}

void GatherReq::print(){
  AmpiRequest::print();
  CkPrintf("In GatherReq: this=%p, status=%d\n", this, statusIreq);
}

void GathervReq::print(){
  AmpiRequest::print();
  CkPrintf("In GathervReq: this=%p, status=%d\n", this, statusIreq);
}

void IATAReq::print(){ //not complete for myreqs
  AmpiRequest::print();
  CkPrintf("In IATAReq: elmcount=%d, idx=%d\n", elmcount, idx);
}

void SendReq::print(){
  AmpiRequest::print();
  CkPrintf("In SendReq: this=%p, status=%d\n", this, statusIreq);
}

void SsendReq::print(){
  AmpiRequest::print();
  CkPrintf("In SsendReq: this=%p, status=%d\n", this, statusIreq);
}

void AmpiRequestList::pup(PUP::er &p) {
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
          case MPI_PERS_REQ:
            block[i] = new PersReq;
            break;
          case MPI_I_REQ:
            block[i] = new IReq;
            break;
          case MPI_REDN_REQ:
            block[i] = new RednReq;
            break;
          case MPI_GATHER_REQ:
            block[i] = new GatherReq;
            break;
          case MPI_GATHERV_REQ:
            block[i] = new GathervReq;
            break;
          case MPI_SEND_REQ:
            block[i] = new SendReq;
            break;
          case MPI_SSEND_REQ:
            block[i] = new SsendReq;
            break;
          case MPI_IATA_REQ:
            block[i] = new IATAReq;
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
}

//------------------ External Interface -----------------
ampiParent *getAmpiParent(void) {
  ampiParent *p = CtvAccess(ampiPtr);
#if CMK_ERROR_CHECKING
  if (p==NULL) CkAbort("Cannot call MPI routines before AMPI is initialized.\n");
#endif
  return p;
}

ampi *getAmpiInstance(MPI_Comm comm) {
  ampi *ptr=getAmpiParent()->comm2ampi(comm);
#if CMK_ERROR_CHECKING
  if (ptr==NULL) CkAbort("AMPI's getAmpiInstance> null pointer\n");
#endif
  return ptr;
}

inline static AmpiRequestList *getReqs(void) {
  return &(getAmpiParent()->ampiReqs);
}

inline void checkComm(MPI_Comm comm){
#if AMPI_ERROR_CHECKING
  getAmpiParent()->checkComm(comm);
#endif
}

inline void checkRequest(MPI_Request req){
#if AMPI_ERROR_CHECKING
  getReqs()->checkRequest(req);
#endif
}

inline void checkRequests(int n, MPI_Request* reqs){
#if AMPI_ERROR_CHECKING
  AmpiRequestList* reqlist = getReqs();
  for(int i=0;i<n;i++)
    reqlist->checkRequest(reqs[i]);
#endif
}

int testRequest(MPI_Request *reqIdx, int *flag, MPI_Status *sts){
  MPI_Status tempStatus;
  if(!sts) sts = &tempStatus;

  if(*reqIdx==MPI_REQUEST_NULL){
    *flag = 1;
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  checkRequest(*reqIdx);
  AmpiRequestList* reqList = getReqs();
  AmpiRequest& req = *(*reqList)[*reqIdx];
  if(1 == (*flag = req.itest(sts))){
    req.complete(sts);
    if(req.getType() != MPI_PERS_REQ) { // only free non-blocking request
      reqList->free(*reqIdx);
      *reqIdx = MPI_REQUEST_NULL;
    }
  }
  return MPI_SUCCESS;
}

int testRequestNoFree(MPI_Request *reqIdx, int *flag, MPI_Status *sts){
  MPI_Status tempStatus;
  if(!sts) sts = &tempStatus;

  if(*reqIdx==MPI_REQUEST_NULL){
    *flag = 1;
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  checkRequest(*reqIdx);
  AmpiRequestList* reqList = getReqs();
  AmpiRequest& req = *(*reqList)[*reqIdx];
  *flag = req.itest(sts);
  if(*flag)
    req.complete(sts);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Is_thread_main(int *flag)
{
  AMPIAPI("AMPI_Is_thread_main");
  *flag=1;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Query_thread(int *provided)
{
  AMPIAPI("AMPI_Query_thread");
  *provided = MPI_THREAD_SINGLE;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Init_thread(int *p_argc, char*** p_argv, int required, int *provided)
{
  AMPIAPI("AMPI_Init_thread");
  *provided = MPI_THREAD_SINGLE;
  AMPI_Init(p_argc, p_argv);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Init(int *p_argc, char*** p_argv)
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
    CkAbort("MPI_Init> AMPI has not been initialized! Possibly due to AMPI requiring '#include \"mpi.h\" be in the same file as main() in C/C++ programs and \'program main\' be renamed to \'subroutine mpi_main\' in Fortran programs!");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Initialized(int *isInit)
{
  if (nodeinit_has_been_called) {
    AMPIAPI("AMPI_Initialized");     /* in case charm init not called */
    *isInit=CtvAccess(ampiInitDone);
  }
  else /* !nodeinit_has_been_called */ {
    *isInit=nodeinit_has_been_called;
  }
  return MPI_SUCCESS;
}

CDECL
int AMPI_Finalized(int *isFinalized)
{
  AMPIAPI("AMPI_Finalized");     /* in case charm init not called */
  *isFinalized=CtvAccess(ampiFinalized);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_rank(MPI_Comm comm, int *rank)
{
  //AMPIAPI("AMPI_Comm_rank");

#if AMPI_ERROR_CHECKING
  int ret = checkCommunicator("AMPI_Comm_rank", comm);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char*)rank, sizeof(int));
    return MPI_SUCCESS;
  }
#endif

  *rank = getAmpiInstance(comm)->getRank(comm);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char*)rank, sizeof(int));
  }
#endif
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_size(MPI_Comm comm, int *size)
{
  //AMPIAPI("AMPI_Comm_size");

#if AMPI_ERROR_CHECKING
  int ret = checkCommunicator("AMPI_Comm_size", comm);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char*)size, sizeof(int));
    return MPI_SUCCESS;
  }
#endif

  *size = getAmpiInstance(comm)->getSize(comm);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char*)size, sizeof(int));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_compare(MPI_Comm comm1,MPI_Comm comm2, int *result)
{
  AMPIAPI("AMPI_Comm_compare");

#if AMPI_ERROR_CHECKING
  int ret;
  ret = checkCommunicator("AMPI_Comm_compare", comm1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = checkCommunicator("AMPI_Comm_compare", comm2);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm1==comm2) *result=MPI_IDENT;
  else{
    int congruent=1;
    vector<int> ind1, ind2;
    ind1 = getAmpiInstance(comm1)->getIndices();
    ind2 = getAmpiInstance(comm2)->getIndices();
    if(ind1.size()==ind2.size()){
      for(int i=0;i<ind1.size();i++){
        int equal=0;
        for(int j=0;j<ind2.size();j++){
          if(ind1[i]==ind2[j]){
            equal=1;
            if(i!=j) congruent=0;
          }
        }
        if(!equal){
          *result=MPI_UNEQUAL;
          return MPI_SUCCESS;
        }
      }
    }
    if(congruent==1) *result=MPI_CONGRUENT;
    else *result=MPI_SIMILAR;
  }
  return MPI_SUCCESS;
}

CDECL
void AMPI_Exit(int /*exitCode*/)
{
  AMPIAPI("AMPI_Exit");
  TCHARM_Done();
}

FDECL
void FTN_NAME(MPI_EXIT,mpi_exit)(int *exitCode)
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
  CtvAccess(ampiFinalized)=1;

#if CMK_BIGSIM_CHARM && CMK_TRACE_IN_CHARM
  if(CpvAccess(traceOn)) traceSuspend();
#endif

  AMPI_Exit(0);
  return MPI_SUCCESS;
}

MPI_Request ampi::postReq(AmpiRequest* newreq, AmpiReqSts status/*=AMPI_REQ_PENDING*/)
{
  MPI_Request request;
  if (status == AMPI_REQ_COMPLETED) {
    newreq->statusIreq = true;
    request = getReqs()->insert(newreq);
  }
  else {
    request = getReqs()->insert(newreq);
    int tags[3] = { newreq->tag, newreq->src, newreq->comm };
    CmmPut(posted_ireqs, 3, tags, (void *)(CmiIntPtr)(request+1));
  }
  return request;
}

CDECL
int AMPI_Send(void *msg, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
  AMPIAPI("AMPI_Send");

  handle_MPI_BOTTOM(msg, type);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Send", comm, 1, count, 1, type, 1, tag, 1, dest, 1, msg, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

#if AMPIMSGLOG
  if(msgLogRead){
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  ptr->send(tag, ptr->getRank(comm), msg, count, type, dest, comm);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ssend(void *msg, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm)
{
  AMPIAPI("AMPI_Ssend");

  handle_MPI_BOTTOM(msg, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Ssend", comm, 1, count, 1, type, 1, tag, 1, dest, 1, msg, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

#if AMPIMSGLOG
  if(msgLogRead){
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  ptr->send(tag, ptr->getRank(comm), msg, count, type, dest, comm, 1);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Issend(void *buf, int count, MPI_Datatype type, int dest,
                int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Issend");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Issend", comm, 1, count, 1, type, 1, tag, 1, dest, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char *)request, sizeof(MPI_Request));
    return MPI_SUCCESS;
  }
#endif

  USER_CALL_DEBUG("AMPI_Issend("<<type<<","<<dest<<","<<tag<<","<<comm<<")");
  ampi *ptr = getAmpiInstance(comm);
  AmpiRequestList* reqs = getReqs();
  SsendReq *newreq = new SsendReq(comm);
  *request = reqs->insert(newreq);
  // 1:  blocking now  - used by MPI_Ssend
  // >=2:  the index of the requests - used by MPI_Issend
  ptr->send(tag, ptr->getRank(comm), buf, count, type, dest, comm, *request+2);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char *)request, sizeof(MPI_Request));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Recv(void *msg, int count, MPI_Datatype type, int src, int tag,
              MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("AMPI_Recv");

  MPI_Status tempStatus;
  if(!status) status = &tempStatus;

  handle_MPI_BOTTOM(msg, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Recv", comm, 1, count, 1, type, 1, tag, 1, src, 1, msg, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)msg, (pptr->pupBytes));
    PUParray(*(pptr->fromPUPer), (char *)status, sizeof(MPI_Status));
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  if(-1==ptr->recv(tag,src,msg,count,type,comm,status)) CkAbort("AMPI> Error in MPI_Recv");

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(type) * count;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)msg, (pptr->pupBytes));
    PUParray(*(pptr->toPUPer), (char *)status, sizeof(MPI_Status));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Probe(int src, int tag, MPI_Comm comm, MPI_Status *status)
{
  AMPIAPI("AMPI_Probe");

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Probe", comm, 1, 0, 0, 0, 0, tag, 1, src, 1, 0, 0);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  MPI_Status tempStatus;
  if(!status) status = &tempStatus;

  ampi *ptr = getAmpiInstance(comm);
  ptr->probe(tag, src, comm, status);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Iprobe(int src,int tag,MPI_Comm comm,int *flag,MPI_Status *status)
{
  AMPIAPI("AMPI_Iprobe");

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Iprobe", comm, 1, 0, 0, 0, 0, tag, 1, src, 1, 0, 0);
  if(ret != MPI_SUCCESS)
    return ret;
#endif
  MPI_Status tempStatus;
  if(!status) status = &tempStatus;

  ampi *ptr = getAmpiInstance(comm);
  *flag = ptr->iprobe(tag, src, comm, status);
  return MPI_SUCCESS;
}

void ampi::sendrecv(void *sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                    void *rbuf, int rcount, MPI_Datatype rtype, int src, int rtag,
                    MPI_Comm comm, MPI_Status *sts)
{
  send(stag, getRank(comm), sbuf, scount, stype, dest, comm);

  if(-1==recv(rtag, src, rbuf, rcount, rtype, comm, sts))
    CkAbort("AMPI> Error in MPI_Sendrecv!\n");
}

CDECL
int AMPI_Sendrecv(void *sbuf, int scount, MPI_Datatype stype, int dest,
                  int stag, void *rbuf, int rcount, MPI_Datatype rtype,
                  int src, int rtag, MPI_Comm comm, MPI_Status *sts)
{
  AMPIAPI("AMPI_Sendrecv");

  handle_MPI_BOTTOM(sbuf, stype, rbuf, rtype);

#if AMPI_ERROR_CHECKING
  if(sbuf == MPI_IN_PLACE || rbuf == MPI_IN_PLACE)
    CkAbort("MPI_sendrecv does not accept MPI_IN_PLACE; use MPI_Sendrecv_replace instead.");
  int ret;
  ret = errorCheck("AMPI_Sendrecv", comm, 1, scount, 1, stype, 1, stag, 1, dest, 1, sbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Sendrecv", comm, 1, rcount, 1, rtype, 1, rtag, 1, src, 1, rbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  MPI_Status tempStatus;
  if(!sts) sts = &tempStatus;
  ampi *ptr = getAmpiInstance(comm);

  ptr->sendrecv(sbuf, scount, stype, dest, stag,
                rbuf, rcount, rtype, src, rtag,
                comm, sts);

  return MPI_SUCCESS;
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

void ampi::barrier()
{
  CkCallback barrierCB(CkReductionTarget(ampi, barrierResult), getProxy());
  contribute(barrierCB);
  thread->suspend(); //Resumed by ampi::barrierResult
}

void ampi::barrierResult(void)
{
  MSG_ORDER_DEBUG(CkPrintf("[%d] barrierResult called\n", thisIndex));
  thread->resume();
}

CDECL
int AMPI_Barrier(MPI_Comm comm)
{
  AMPIAPI("AMPI_Barrier");

#if AMPI_ERROR_CHECKING
  int ret = checkCommunicator("AMPI_Barrier", comm);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return MPI_SUCCESS;
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Barrier for Inter-communicators!");

#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_LOG(MPI_BARRIER, 0);
#endif

  ampi *ptr = getAmpiInstance(comm);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Barrier called on comm %d\n", ptr->thisIndex, comm));

  ptr->barrier();

  return MPI_SUCCESS;
}

void ampi::ibarrier(MPI_Request *request)
{
  CkCallback ibarrierCB(CkReductionTarget(ampi, ibarrierResult), getProxy());
  contribute(ibarrierCB);

  // use an IReq to non-block the caller and get a request ptr
  *request = postReq(new IReq(NULL, 0, MPI_INT, AMPI_COLL_SOURCE, MPI_ATA_TAG, AMPI_COLL_COMM));
}

void ampi::ibarrierResult(void)
{
  MSG_ORDER_DEBUG(CkPrintf("[%d] ibarrierResult called\n", thisIndex));
  ampi::sendraw(MPI_ATA_TAG, AMPI_COLL_SOURCE, NULL, 0, thisArrayID, thisIndex);
}

CDECL
int AMPI_Ibarrier(MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ibarrier");

#if AMPI_ERROR_CHECKING
  int ret = checkCommunicator("AMPI_Ibarrier", comm);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(NULL, 0, MPI_INT, AMPI_COLL_SOURCE, MPI_ATA_TAG, AMPI_COLL_COMM),
                            AMPI_REQ_COMPLETED);
    return MPI_SUCCESS;
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Ibarrier for Inter-communicators!");

#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_LOG(MPI_BARRIER, 0);
#endif

  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Ibarrier called on comm %d\n", ptr->thisIndex, comm));

  ptr->ibarrier(request);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Bcast");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Bcast", comm, 1, count, 1, type, 1, 0, 0, root, 1, buf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return MPI_SUCCESS;
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Bcast for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)buf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  ampi* ptr = getAmpiInstance(comm);
  ptr->bcast(root, buf, count, type,comm);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)) {
    (pptr->pupBytes) = getDDT()->getSize(type) * count;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)buf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ibcast(void *buf, int count, MPI_Datatype type, int root,
                MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ibcast");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Ibcast", comm, 1, count, 1, type, 1, 0, 0, root, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi* ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(buf, count, type, root, MPI_BCAST_TAG, MPI_BCAST_COMM),
                            AMPI_REQ_COMPLETED);
    return MPI_SUCCESS;
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Ibcast for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)buf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  ptr->ibcast(root, buf, count, type, comm, request);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)) {
    (pptr->pupBytes) = getDDT()->getSize(type) * count;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)buf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

// This routine is called with the results of an (All)Reduce or (All)Gather(v)
void ampi::rednResult(CkReductionMsg *msg)
{
  MSG_ORDER_DEBUG(CkPrintf("[%d] rednResult called on comm %d\n", thisIndex, myComm.getComm()));

  if (blockingReq == NULL) {
    CkAbort("AMPI> recv'ed a blocking reduction unexpectedly!\n");
  }

#if CMK_BIGSIM_CHARM
  TRACE_BG_ADD_TAG("AMPI_generic");
  msg->event = NULL;
  _TRACE_BG_TLINE_END(&msg->event); // store current log
  msg->eventPe = CkMyPe();
#endif

  blockingReq->receive(this, msg);

  if (resumeOnColl) {
    thread->resume();
  }
  // [nokeep] entry method, so do not delete msg
}

// This routine is called with the results of an I(all)reduce or I(all)gather(v)
void ampi::irednResult(CkReductionMsg *msg)
{
  MSG_ORDER_DEBUG(CkPrintf("[%d] irednResult called on comm %d\n", thisIndex, myComm.getComm()));

  MPI_Status sts;
  int tags[3] = { MPI_REDN_TAG, AMPI_COLL_SOURCE, myComm.getComm() };
  AmpiRequestList *reqL = &(parent->ampiReqs);
  int rednReqIdx = (int)((long)CmmGet(posted_ireqs, 3, tags, (int*)&sts));
  AmpiRequest *rednReq = NULL;
  if(reqL->size()>0 && rednReqIdx>0)
    rednReq = (AmpiRequest *)(*reqL)[rednReqIdx-1];
  if (rednReq == NULL)
    CkAbort("AMPI> recv'ed a non-blocking reduction unexpectedly!\n");

#if CMK_BIGSIM_CHARM
  TRACE_BG_ADD_TAG("AMPI_generic");
  msg->event = NULL;
  _TRACE_BG_TLINE_END(&msg->event); // store current log
  msg->eventPe = CkMyPe();
#endif
#if AMPIMSGLOG
  if(msgLogRead){
    PUParray(*(getAmpiParent()->fromPUPer), (char *)rednReq, sizeof(int));
    return;
  }
#endif

  rednReq->receive(this, msg);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(getAmpiParent()->thisIndex)){
    PUParray(*(getAmpiParent()->toPUPer), (char *)reqnReq, sizeof(int));
  }
#endif

  if (resumeOnColl) {
    thread->resume();
  }
  // [nokeep] entry method, so do not delete msg
}

static CkReductionMsg *makeRednMsg(CkDDT_DataType *ddt,const void *inbuf,int count,int type,int rank,MPI_Op op)
{
  CkReductionMsg *msg;
  ampiParent *parent = getAmpiParent();
  int szdata = ddt->getSize(count);
  CkReduction::reducerType reducer = getBuiltinReducerType(type, op);

  if (reducer != CkReduction::invalid) {
    // MPI predefined op matches a Charm++ builtin reducer type
    AMPI_DEBUG("[%d] In makeRednMsg, using Charm++ built-in reducer type for a predefined op\n", thisIndex);
    msg = CkReductionMsg::buildNew(szdata, NULL, reducer);
    ddt->serialize((char*)inbuf, (char*)msg->getData(), count, 1);
  }
  else if (parent->opIsCommutative(op)) {
    // Either an MPI predefined reducer operation with no Charm++ builtin
    // reducer type equivalent, or a commutative user-defined reducer operation
    AMPI_DEBUG("[%d] In makeRednMsg, using custom AmpiReducer type for a commutative op\n", thisIndex);
    AmpiOpHeader newhdr = parent->op2AmpiOpHeader(op, type, count);
    int szhdr = sizeof(AmpiOpHeader);
    msg = CkReductionMsg::buildNew(szdata+szhdr, NULL, AmpiReducer);
    memcpy(msg->getData(), &newhdr, szhdr);
    ddt->serialize((char*)inbuf, (char*)msg->getData()+szhdr, count, 1);
  }
  else {
    // Non-commutative user-defined reducer operation
    AMPI_DEBUG("[%d] In makeRednMsg, using a non-commutative user-defined operation\n", thisIndex);
    const int tupleSize = 2;
    CkReduction::tupleElement tupleRedn[tupleSize];
    tupleRedn[0] = CkReduction::tupleElement(sizeof(int), &rank, CkReduction::set);
    if (!ddt->isContig()) {
      vector<char> sbuf(szdata);
      ddt->serialize((char*)inbuf, &sbuf[0], count, 1);
      tupleRedn[1] = CkReduction::tupleElement(szdata, &sbuf[0], CkReduction::set);
    }
    else {
      tupleRedn[1] = CkReduction::tupleElement(szdata, (void*)inbuf, CkReduction::set);
    }
    msg = CkReductionMsg::buildFromTuple(tupleRedn, tupleSize);
  }
  return msg;
}

// Copy the MPI datatype "type" from inbuf to outbuf
static int copyDatatype(MPI_Comm comm,MPI_Datatype type,int count,const void *inbuf,void *outbuf) {
  ampi *ptr = getAmpiInstance(comm);
  CkDDT_DataType *ddt = ptr->getDDT()->getType(type);
  int len = ddt->getSize(count);

  if (ddt->isContig()) {
    memcpy(outbuf, inbuf, len);
  } else {
    // ddts don't have "copy", so fake it by serializing into a temp buffer, then
    //  deserializing into the output.
    vector<char> serialized(len);
    ddt->serialize((char*)inbuf, &serialized[0], count, 1);
    ddt->serialize((char*)outbuf, &serialized[0], count, -1);
  }

  return MPI_SUCCESS;
}

static void handle_MPI_IN_PLACE(void* &inbuf, void* &outbuf)
{
  if (inbuf == MPI_IN_PLACE) inbuf = outbuf;
  if (outbuf == MPI_IN_PLACE) outbuf = inbuf;
  CkAssert(inbuf != MPI_IN_PLACE && outbuf != MPI_IN_PLACE);
}

void applyOp(MPI_Datatype datatype, MPI_Op op, int count, void* invec, void* inoutvec)
{
  // inoutvec[i] = invec[i] op inoutvec[i]
  MPI_User_function *func = getAmpiParent()->op2User_function(op);
  (func)(invec, inoutvec, &count, &datatype);
}

#define SYNCHRONOUS_REDUCE                           0

CDECL
int AMPI_Reduce(void *inbuf, void *outbuf, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Reduce");

  handle_MPI_BOTTOM(inbuf, type, outbuf, type);
  handle_MPI_IN_PLACE(inbuf, outbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Reduce", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Reduce", comm, 1, count, 1, type, 1, 0, 0, root, 1, inbuf, 1,
                       outbuf, getAmpiInstance(comm)->getRank(comm) == root);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,type,count,inbuf,outbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Reduce for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)outbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rootIdx=ptr->comm2CommStruct(comm).getIndexForRank(root);

  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,ptr->getRank(comm),op);

  CkCallback reduceCB(CkIndex_ampi::rednResult(0),CkArrayIndex1D(rootIdx),ptr->getProxy());
  msg->setCallback(reduceCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Reduce called on comm %d root %d \n",ptr->thisIndex,comm,rootIdx));
  ptr->contribute(msg);

  if (ptr->thisIndex == rootIdx){
    ptr = ptr->blockOnRedn(new RednReq(outbuf, count, type, comm, op));

#if SYNCHRONOUS_REDUCE
    AmpiMsg *msg = new (0, 0) AmpiMsg(-1, MPI_REDN_TAG, -1, rootIdx, 0, comm);
    CProxy_ampi pa(ptr->getProxy());
    pa.generic(msg);
#endif
  }
#if SYNCHRONOUS_REDUCE
  ptr->recv(MPI_REDN_TAG, AMPI_COLL_SOURCE, NULL, 0, type, comm);
#endif

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(type) * count;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)outbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Allreduce(void *inbuf, void *outbuf, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("AMPI_Allreduce");

  handle_MPI_BOTTOM(inbuf, type, outbuf, type);
  handle_MPI_IN_PLACE(inbuf, outbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Allreduce", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Allreduce", comm, 1, count, 1, type, 1, 0, 0, 0, 0, inbuf, 1, outbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,type,count,inbuf,outbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Allreduce for Inter-communicators!");

#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_LOG(MPI_ALLREDUCE, getAmpiInstance(comm)->getDDT()->getType(type)->getSize(count));
#endif

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)outbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type), inbuf, count, type, ptr->getRank(comm), op);
  CkCallback allreduceCB(CkIndex_ampi::rednResult(0),ptr->getProxy());
  msg->setCallback(allreduceCB);
  ptr->contribute(msg);

  ptr->blockOnRedn(new RednReq(outbuf, count, type, comm, op));

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(type) * count;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)outbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Iallreduce(void *inbuf, void *outbuf, int count, MPI_Datatype type, MPI_Op op,
                    MPI_Comm comm, MPI_Request* request)
{
  AMPIAPI("AMPI_Iallreduce");

  handle_MPI_BOTTOM(inbuf, type, outbuf, type);
  handle_MPI_IN_PLACE(inbuf, outbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Iallreduce", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Iallreduce", comm, 1, count, 1, type, 1, 0, 0, 0, 0, inbuf, 1, outbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new RednReq(outbuf,count,type,comm,op), AMPI_REQ_COMPLETED);
    return copyDatatype(comm,type,count,inbuf,outbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Iallreduce for Inter-communicators!");

  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),inbuf,count,type,ptr->getRank(comm),op);
  CkCallback allreduceCB(CkIndex_ampi::irednResult(0),ptr->getProxy());
  msg->setCallback(allreduceCB);
  ptr->contribute(msg);

  // use a RednReq to non-block the caller and get a request ptr
  *request = ptr->postReq(new RednReq(outbuf,count,type,comm,op));

  return MPI_SUCCESS;
}

CDECL
int AMPI_Reduce_local(void *inbuf, void *outbuf, int count, MPI_Datatype type, MPI_Op op)
{
  AMPIAPI("AMPI_Reduce_local");

  handle_MPI_BOTTOM(inbuf, type, outbuf, type);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Reduce_local", MPI_ERR_OP);
  if(inbuf == MPI_IN_PLACE || outbuf == MPI_IN_PLACE)
    CkAbort("MPI_Reduce_local does not accept MPI_IN_PLACE!");
  int ret = errorCheck("AMPI_Reduce_local", MPI_COMM_SELF, 1, count, 1, type, 1, 0, 0, 0, 1, inbuf, 1, outbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  applyOp(type, op, count, inbuf, outbuf);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Reduce_scatter_block(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                              MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("AMPI_Reduce_scatter_block");

  handle_MPI_BOTTOM(sendbuf, datatype, recvbuf, datatype);
  handle_MPI_IN_PLACE(sendbuf, recvbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Reduce_scatter_block", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Reduce_scatter_block", comm, 1, 0, 0, datatype, 1, 0, 0, 0, 0, sendbuf, 1, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm, datatype, count, sendbuf, recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Reduce_scatter_block for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  vector<char> tmpbuf(ptr->getDDT()->getType(datatype)->getSize(count)*size);

  AMPI_Reduce(sendbuf, &tmpbuf[0], count*size, datatype, op, AMPI_COLL_SOURCE, comm);
  AMPI_Scatter(&tmpbuf[0], count, datatype, recvbuf, count, datatype, AMPI_COLL_SOURCE, comm);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Reduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts, MPI_Datatype datatype,
                        MPI_Op op, MPI_Comm comm)
{
  AMPIAPI("AMPI_Reduce_scatter");

  handle_MPI_BOTTOM(sendbuf, datatype, recvbuf, datatype);
  handle_MPI_IN_PLACE(sendbuf, recvbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Reduce_scatter", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Reduce_scatter", comm, 1, 0, 0, datatype, 1, 0, 0, 0, 0, sendbuf, 1, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,datatype,recvcounts[0],sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Reduce_scatter for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int count=0;
  vector<int> displs(size);
  int len;

  //under construction
  for(int i=0;i<size;i++){
    displs[i] = count;
    count+= recvcounts[i];
  }
  vector<char> tmpbuf(ptr->getDDT()->getType(datatype)->getSize(count));
  AMPI_Reduce(sendbuf, &tmpbuf[0], count, datatype, op, AMPI_COLL_SOURCE, comm);
  AMPI_Scatterv(&tmpbuf[0], recvcounts, &displs[0], datatype,
      recvbuf, recvcounts[ptr->getRank(comm)], datatype, AMPI_COLL_SOURCE, comm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Scan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
              MPI_Op op, MPI_Comm comm ){
  AMPIAPI("AMPI_Scan");

  handle_MPI_BOTTOM(sendbuf, datatype, recvbuf, datatype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Scan", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Scan", comm, 1, count, 1, datatype, 1, 0, 0, 0, 0, sendbuf, 1, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  MPI_Status sts;
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int blklen = ptr->getDDT()->getType(datatype)->getSize(count);
  int rank = ptr->getRank(comm);
  int mask = 0x1;
  int dst;
  vector<char> tmp_buf(blklen);
  vector<char> partial_scan(blklen);

  memcpy(recvbuf, sendbuf, blklen);
  memcpy(&partial_scan[0], sendbuf, blklen);
  while(mask < size){
    dst = rank^mask;
    if(dst < size){
      ptr->sendrecv(&partial_scan[0], count, datatype, dst, MPI_SCAN_TAG,
                    &tmp_buf[0], count, datatype, dst, MPI_SCAN_TAG, comm, &sts);
      if(rank > dst){
        applyOp(datatype, op, count, &tmp_buf[0], &partial_scan[0]);
        applyOp(datatype, op, count, &tmp_buf[0], recvbuf);
      }else {
        applyOp(datatype, op, count, &partial_scan[0], &tmp_buf[0]);
        memcpy(&partial_scan[0],&tmp_buf[0],blklen);
      }
    }
    mask <<= 1;
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Exscan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, MPI_Comm comm){
  AMPIAPI("AMPI_Exscan");

  handle_MPI_BOTTOM(sendbuf, datatype, recvbuf, datatype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Exscan", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Excan", comm, 1, count, 1, datatype, 1, 0, 0, 0, 0, sendbuf, 1, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  MPI_Status sts;
  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  int blklen = ptr->getDDT()->getType(datatype)->getSize(count);
  int rank = ptr->getRank(comm);
  int mask = 0x1;
  int dst, flag;
  vector<char> tmp_buf(blklen);
  vector<char> partial_scan(blklen);

  memcpy(recvbuf, sendbuf, blklen);
  memcpy(&partial_scan[0], sendbuf, blklen);
  flag = 0;
  mask = 0x1;
  while(mask < size){
    dst = rank^mask;
    if(dst < size){
      ptr->sendrecv(&partial_scan[0], count, datatype, dst, MPI_EXSCAN_TAG,
                    &tmp_buf[0], count, datatype, dst, MPI_EXSCAN_TAG, comm, &sts);
      if(rank > dst){
        applyOp(datatype, op, count, &tmp_buf[0], &partial_scan[0]);
        if(rank != 0){
          if(flag == 0){
            memcpy(recvbuf,&tmp_buf[0],blklen);
            flag = 1;
          }
          else{
            applyOp(datatype, op, count, &tmp_buf[0], recvbuf);
          }
        }
      }
      else{
        applyOp(datatype, op, count, &partial_scan[0], &tmp_buf[0]);
        memcpy(&partial_scan[0],&tmp_buf[0],blklen);
      }
      mask <<= 1;
    }
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op){
  AMPIAPI("AMPI_Op_create");
  *op = getAmpiParent()->createOp(function, commute);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Op_free(MPI_Op *op){
  AMPIAPI("AMPI_Op_free");
  *op = MPI_OP_NULL;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Op_commutative(MPI_Op op, int *commute){
  AMPIAPI("AMPI_Op_commutative");
  *commute = (int)getAmpiParent()->opIsCommutative(op);
  return MPI_SUCCESS;
}

CDECL
double AMPI_Wtime(void)
{
  //AMPIAPI("AMPI_Wtime");

#if AMPIMSGLOG
  double ret=TCHARM_Wall_timer();
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|ret;
    return ret;
  }

  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (*(pptr->toPUPer))|ret;
  }
#endif

#if CMK_BIGSIM_CHARM
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
  if(sndrcv == 1 || sndrcv == 3) { // send or ssend request
    ampi *ptr=getAmpiInstance(comm);
    ptr->send(tag, ptr->getRank(comm), buf, count, type, src, comm, sndrcv==3?1:0);
  }
  return 0;
}

CDECL
int AMPI_Start(MPI_Request *request)
{
  AMPIAPI("AMPI_Start");
  checkRequest(*request);
  AmpiRequestList *reqs = getReqs();
  if(-1==(*reqs)[*request]->start()) {
    CkAbort("MPI_Start could be used only on persistent communication requests!");
  }
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
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
  for (i=0; i<n-1; i++) {
    for (j=0; j<n-1-i; j++) {
      if (arr[idx[j+1]] < arr[idx[j]])
        swapInt(idx[j+1],idx[j]);
    }
  }
}

CkVec<CkVec<int> > *vecIndex(int count, int* arr){
  CkAssert(count!=0);
  vector<int> newidx(count);
  int flag;
  sortedIndex(count,arr,&newidx[0]);
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
    if(-1==getAmpiInstance(comm)->recv(tag, src, buf, count, type, comm, sts))
      CkAbort("AMPI> Error in persistent request wait");
#if CMK_BIGSIM_CHARM
    _TRACE_BG_TLINE_END(&event);
#endif
  }
  return 0;
}

int IReq::wait(MPI_Status *sts){
  //Copy "this" to a local variable in the case that "this" pointer
  //is updated during the out-of-core emulation.

  // optimization for Irecv
  // generic() writes directly to the buffer, so the only thing we
  // do here is to wait
  ampi *dis = getAmpiInstance(comm);

  while (statusIreq == false) {
    // "dis" is updated in case an ampi thread is migrated while waiting for a message
    dis->resumeOnRecv = true;
    dis->block();
    dis = getAmpiInstance(comm);

#if CMK_BIGSIM_CHARM
    //Because of the out-of-core emulation, this pointer is changed after in-out
    //memory operation. So we need to return from this function and do the while loop
    //in the outer function call.
    if(_BgInOutOfCoreMode)
      return -1;
#endif
  } // end of while
  dis->resumeOnRecv = false;

  AMPI_DEBUG("IReq::wait has resumed\n");

  if(sts) {
    AMPI_DEBUG("Setting sts->MPI_TAG to this->tag=%d in IReq::wait  this=%p\n", (int)this->tag, this);
    sts->MPI_TAG = tag;
    sts->MPI_SOURCE = src;
    sts->MPI_COMM = comm;
    sts->MPI_LENGTH = length;
  }

  return 0;
}

int RednReq::wait(MPI_Status *sts){
  //Copy "this" to a local variable in the case that "this" pointer
  //is updated during the out-of-core emulation.

  // ampi::irednResult writes directly to the buffer, so the only thing we
  // do here is to wait
  ampi *dis = getAmpiInstance(comm);

  while (!statusIreq) {
    dis->resumeOnColl = true;
    dis->block();
    dis = getAmpiInstance(comm);

#if CMK_BIGSIM_CHARM
    //Because of the out-of-core emulation, this pointer is changed after in-out
    //memory operation. So we need to return from this function and do the while loop
    //in the outer function call.
    if (_BgInOutOfCoreMode)
      return -1;
#endif
  }
  dis->resumeOnColl = false;

  AMPI_DEBUG("RednReq::wait has resumed\n");

  if (sts) {
    sts->MPI_TAG = tag;
    sts->MPI_SOURCE = src;
    sts->MPI_COMM = comm;
  }
  return 0;
}

int GatherReq::wait(MPI_Status *sts){
  //Copy "this" to a local variable in the case that "this" pointer
  //is updated during the out-of-core emulation.

  // ampi::irednResult writes directly to the buffer, so the only thing we
  // do here is to wait
  ampi *dis = getAmpiInstance(comm);

  while (!statusIreq) {
    dis->resumeOnColl = true;
    dis->block();
    dis = getAmpiInstance(comm);

#if CMK_BIGSIM_CHARM
    //Because of the out-of-core emulation, this pointer is changed after in-out
    //memory operation. So we need to return from this function and do the while loop
    //in the outer function call.
    if (_BgInOutOfCoreMode)
      return -1;
#endif
  }
  dis->resumeOnColl = false;

  AMPI_DEBUG("GatherReq::wait has resumed\n");

  if (sts) {
    sts->MPI_TAG = tag;
    sts->MPI_SOURCE = src;
    sts->MPI_COMM = comm;
  }
  return 0;
}

int GathervReq::wait(MPI_Status *sts){
  //Copy "this" to a local variable in the case that "this" pointer
  //is updated during the out-of-core emulation.

  // ampi::irednResult writes directly to the buffer, so the only thing we
  // do here is to wait
  ampi *dis = getAmpiInstance(comm);

  while (!statusIreq) {
    dis->resumeOnColl = true;
    dis->block();
    dis = getAmpiInstance(comm);

#if CMK_BIGSIM_CHARM
    //Because of the out-of-core emulation, this pointer is changed after in-out
    //memory operation. So we need to return from this function and do the while loop
    //in the outer function call.
    if (_BgInOutOfCoreMode)
      return -1;
#endif
  }
  dis->resumeOnColl = false;

  AMPI_DEBUG("GathervReq::wait has resumed\n");

  if (sts) {
    sts->MPI_TAG = tag;
    sts->MPI_SOURCE = src;
    sts->MPI_COMM = comm;
  }
  return 0;
}

int SendReq::wait(MPI_Status *sts){
  ampi *dis = getAmpiInstance(comm);
  while (!statusIreq) {
    dis->resumeOnRecv = true;
    dis->block();
    // "dis" is updated in case an ampi thread is migrated while waiting for a message
    dis = getAmpiInstance(comm);
  }
  dis->resumeOnRecv = false;
  AMPI_DEBUG("SendReq::wait has resumed\n");
  if (sts) {
    sts->MPI_COMM = comm;
  }
  return 0;
}

int SsendReq::wait(MPI_Status *sts){
  ampi *dis = getAmpiInstance(comm);
  while (!statusIreq) {
    // "dis" is updated in case an ampi thread is migrated while waiting for a message
    dis = dis->blockOnRecv();
  }
  if (sts) {
    sts->MPI_COMM = comm;
  }
  return 0;
}

int IATAReq::wait(MPI_Status *sts){
  int i;
  for(i=0;i<elmcount;i++){
    if(-1==getAmpiInstance(myreqs[i].comm)->recv(myreqs[i].tag, myreqs[i].src, myreqs[i].buf,
                                                 myreqs[i].count, myreqs[i].type,
                                                 myreqs[i].comm, sts))
      CkAbort("AMPI> Error in ialltoall request wait");
#if CMK_BIGSIM_CHARM
    _TRACE_BG_TLINE_END(&myreqs[i].event);
#endif
  }
#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_BREAK(getAmpiInstance(MPI_COMM_WORLD)->getThread(), "IATAReq_wait", NULL, 0, 1);
  for (i=0; i<elmcount; i++)
    _TRACE_BG_ADD_BACKWARD_DEP(myreqs[i].event);
  _TRACE_BG_TLINE_END(&event);
#endif
  return 0;
}

CDECL
int AMPI_Wait(MPI_Request *request, MPI_Status *sts)
{
  AMPIAPI("AMPI_Wait");

  MPI_Status tempStatus;
  if(!sts) sts = &tempStatus;

  if(*request == MPI_REQUEST_NULL){
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  checkRequest(*request);
  AmpiRequestList* reqs = getReqs();

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)((*reqs)[*request]->buf), (pptr->pupBytes));
    PUParray(*(pptr->fromPUPer), (char *)sts, sizeof(MPI_Status));
    return MPI_SUCCESS;
  }
#endif

#if CMK_BIGSIM_CHARM
  void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#endif

  AMPI_DEBUG("AMPI_Wait request=%d (*reqs)[*request]=%p (*reqs)[*request]->tag=%d\n",
             *request, (*reqs)[*request], (int)((*reqs)[*request]->tag));
  AMPI_DEBUG("MPI_Wait: request=%d, reqs.size=%d, &reqs=%d\n",
             *request, reqs->size(), reqs);
  int waitResult = -1;
  do{
    AmpiRequest *waitReq = (*reqs)[*request];
    waitResult = waitReq->wait(sts);
#if CMK_BIGSIM_CHARM
    if(_BgInOutOfCoreMode){
      reqs = getReqs();
    }
#endif
  }while(waitResult==-1);

  AMPI_DEBUG("AMPI_Wait after calling wait, request=%d (*reqs)[*request]=%p (*reqs)[*request]->tag=%d\n",
             *request, (*reqs)[*request], (int)((*reqs)[*request]->tag));

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize((*reqs)[*request]->type) * ((*reqs)[*request]->count);
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)((*reqs)[*request]->buf), (pptr->pupBytes));
    PUParray(*(pptr->toPUPer), (char *)sts, sizeof(MPI_Status));
  }
#endif

#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_WAIT(reqs); // setup forward and backward dependence
#endif

  if((*reqs)[*request]->getType() != MPI_PERS_REQ) { // only free non-blocking request
    reqs->free(*request);
    *request = MPI_REQUEST_NULL;
  }

  AMPI_DEBUG("End of AMPI_Wait\n");

  return MPI_SUCCESS;
}

CDECL
int AMPI_Waitall(int count, MPI_Request request[], MPI_Status sts[])
{
  AMPIAPI("AMPI_Waitall");
  if(count==0) return MPI_SUCCESS;
  checkRequests(count,request);
  int i,j,oldPe;

  MPI_Status tempStatus;

  AmpiRequestList* reqs = getReqs();
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    for(i=0;i<reqvec->size();i++){
      for(j=0;j<((*reqvec)[i]).size();j++){
        if(request[((*reqvec)[i])[j]] == MPI_REQUEST_NULL){
          stsempty(sts[((*reqvec)[i])[j]]);
          continue;
        }
        AmpiRequest *waitReq = ((*reqs)[request[((*reqvec)[i])[j]]]);
        (*(pptr->fromPUPer))|(pptr->pupBytes);
        PUParray(*(pptr->fromPUPer), (char *)(waitReq->buf), (pptr->pupBytes));
        PUParray(*(pptr->fromPUPer), (char *)(&sts[((*reqvec)[i])[j]]), sizeof(MPI_Status));
      }
    }
    return MPI_SUCCESS;
  }
#endif

#if CMK_BIGSIM_CHARM
  void *curLog; // store current log in timeline
  _TRACE_BG_TLINE_END(&curLog);
#endif
  for(i=0;i<reqvec->size();i++){
    for(j=0;j<((*reqvec)[i]).size();j++){
      if(request[((*reqvec)[i])[j]] == MPI_REQUEST_NULL){
        if(sts)
          stsempty(sts[((*reqvec)[i])[j]]);
        continue;
      }
      oldPe = CkMyPe();

      int waitResult = -1;
      do{
        AmpiRequest *waitReq = ((*reqs)[request[((*reqvec)[i])[j]]]);
        waitResult = waitReq->wait(sts ? &sts[((*reqvec)[i])[j]] : &tempStatus);

#if CMK_BIGSIM_CHARM
        if(_BgInOutOfCoreMode){
          reqs = getReqs();
          reqvec = vecIndex(count, request);
        }
#endif
#if AMPIMSGLOG
        if(msgLogWrite && record_msglog(pptr->thisIndex)){
          (pptr->pupBytes) = getDDT()->getSize(waitReq->type) * (waitReq->count);
          (*(pptr->toPUPer))|(pptr->pupBytes);
          PUParray(*(pptr->toPUPer), (char *)(waitReq->buf), (pptr->pupBytes));
          PUParray(*(pptr->toPUPer), (char *)(&sts[((*reqvec)[i])[j]]), sizeof(MPI_Status));
        }
#endif

      }while(waitResult==-1);

#if 1
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
      //for fault evacuation
      if(oldPe != CkMyPe()){
#endif
        reqs = getReqs();
        reqvec  = vecIndex(count,request);
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
      }
#endif
#endif
    }
  }
#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_WAITALL(reqs); // setup forward and backward dependence
#endif
  // free memory of requests
  for(i=0;i<count;i++){
    if(request[i] == MPI_REQUEST_NULL)
      continue;
    if((*reqs)[request[i]]->getType() != MPI_PERS_REQ) { // only free non-blocking request
      reqs->free(request[i]);
      request[i] = MPI_REQUEST_NULL;
    }
  }
  delete reqvec;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Waitany(int count, MPI_Request *request, int *idx, MPI_Status *sts)
{
  AMPIAPI("AMPI_Waitany");

  USER_CALL_DEBUG("AMPI_Waitany("<<count<<")");
  if(count == 0) return MPI_SUCCESS;
  checkRequests(count,request);
  MPI_Status tempStatus;
  if(!sts) sts = &tempStatus;

  if(areInactiveReqs(count,request)){
    *idx=MPI_UNDEFINED;
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  int flag=0;
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  while(count>0){ /* keep looping until some request finishes: */
    for(int i=0;i<reqvec->size();i++){
      testRequest(&request[((*reqvec)[i])[0]], &flag, sts);
      if(flag == 1 && sts->MPI_COMM != 0){ // to skip MPI_REQUEST_NULL
        *idx = ((*reqvec)[i])[0];
        USER_CALL_DEBUG("AMPI_Waitany returning "<<*idx);
        return MPI_SUCCESS;
      }
    }
    /* no requests have finished yet-- block until one does */
    getAmpiInstance(MPI_COMM_WORLD)->blockOnRecv();
  }
  *idx = MPI_UNDEFINED;
  USER_CALL_DEBUG("AMPI_Waitany returning UNDEFINED");
  delete reqvec;
  return MPI_SUCCESS;
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
      testRequest(&array_of_requests[((*reqvec)[i])[0]], &flag, &sts);
      if(flag == 1){
        array_of_indices[(*outcount)]=((*reqvec)[i])[0];
        if(sts.MPI_COMM != 0){
          realflag=1; // there is real(non null) request
          (*outcount)++;
          if(array_of_statuses){
            array_of_statuses[(*outcount)]=sts;
          }
        }
      }
    }
    if(realflag && *outcount>0)
      break;
    else
      getAmpiInstance(MPI_COMM_WORLD)->blockOnRecv();
  }
  delete reqvec;
  return MPI_SUCCESS;
}

bool PersReq::test(MPI_Status *sts){
  if(sndrcv == 2) // recv request
    return getAmpiInstance(comm)->iprobe(tag, src, comm, sts);
  else            // send request
    return true;
}

bool PersReq::itest(MPI_Status *sts){
  return test(sts);
}

bool IReq::test(MPI_Status *sts){
  if (statusIreq && sts) {
    sts->MPI_LENGTH = length;
  }
  else {
    getAmpiInstance(comm)->yield();
  }
  return statusIreq;
}

bool IReq::itest(MPI_Status *sts){
  if (statusIreq && sts) {
    sts->MPI_LENGTH = length;
  }
  return statusIreq;
}

bool RednReq::test(MPI_Status *sts){
  if (!statusIreq) {
    getAmpiInstance(comm)->yield();
  }
  return statusIreq;
}

bool RednReq::itest(MPI_Status *sts){
  return statusIreq;
}

bool GatherReq::test(MPI_Status *sts){
  if (!statusIreq) {
    getAmpiInstance(comm)->yield();
  }
  return statusIreq;
}

bool GatherReq::itest(MPI_Status *sts){
  return statusIreq;
}

bool GathervReq::test(MPI_Status *sts){
  if (!statusIreq) {
    getAmpiInstance(comm)->yield();
  }
  return statusIreq;
}

bool GathervReq::itest(MPI_Status *sts){
  return statusIreq;
}

bool SendReq::test(MPI_Status *sts){
  if (!statusIreq) {
    getAmpiInstance(comm)->yield();
  }
  return statusIreq;
}

bool SendReq::itest(MPI_Status *sts){
  return statusIreq;
}

bool SsendReq::test(MPI_Status *sts){
  if (!statusIreq) {
    getAmpiInstance(comm)->yield();
  }
  return statusIreq;
}

bool SsendReq::itest(MPI_Status *sts){
  return statusIreq;
}

bool IATAReq::test(MPI_Status *sts){
  for(int i=0;i<elmcount;i++){
    if(false==myreqs[i].itest(sts)){
      getAmpiInstance(comm)->yield();
      return false;
    }
  }
  return true;
}

bool IATAReq::itest(MPI_Status *sts){
  for(int i=0;i<elmcount;i++){
    if(false==myreqs[i].itest(sts))
      return false;
  }
  return true;
}

void PersReq::complete(MPI_Status *sts){
  if(-1==getAmpiInstance(comm)->recv(tag, src, buf, count, type, comm, sts))
    CkAbort("AMPI> Error in persistent request complete");
}

void IReq::complete(MPI_Status *sts){
  wait(sts);
}

void RednReq::complete(MPI_Status *sts){
  wait(sts);
}

void GatherReq::complete(MPI_Status *sts){
  wait(sts);
}

void GathervReq::complete(MPI_Status *sts){
  wait(sts);
}

void SendReq::complete(MPI_Status *sts){
  wait(sts);
}

void SsendReq::complete(MPI_Status *sts){
  wait(sts);
}

void IATAReq::complete(MPI_Status *sts){
  for(int i=0;i<elmcount;i++){
    if(-1==getAmpiInstance(myreqs[i].comm)->recv(myreqs[i].tag, myreqs[i].src, myreqs[i].buf,
                                                 myreqs[i].count, myreqs[i].type,
                                                 myreqs[i].comm, sts))
      CkAbort("AMPI> Error in ialltoall request complete");
  }
}

void IReq::receive(ampi *ptr, AmpiMsg *msg)
{
  ptr->processAmpiMsg(msg, buf, type, count);
  statusIreq = true;
  length = msg->length;
  this->tag = msg->tag; // Although not required, we also extract tag from msg
  src = msg->srcRank;   // Although not required, we also extract src from msg
  comm = msg->comm;
  AMPI_DEBUG("Setting this->tag to %d in IReq::receive this=%p\n", (int)this->tag, this);
#if CMK_BIGSIM_CHARM
  event = msg->event;
  eventPe = msg->eventPe;
#endif
  delete msg;
}

void RednReq::receive(ampi *ptr, CkReductionMsg *msg)
{
  if (ptr->opIsCommutative(op)) {
    ptr->processRednMsg(msg, buf, type, count);
  } else {
    MPI_User_function* func = ptr->op2User_function(op);
    ptr->processNoncommutativeRednMsg(msg, buf, type, count, func);
  }
  statusIreq = true;
  comm = ptr->getComm();
#if CMK_BIGSIM_CHARM
  event = msg->event;
  eventPe = msg->eventPe;
#endif
  // ampi::rednResult is a [nokeep] entry method, so do not delete msg
}

void GatherReq::receive(ampi *ptr, CkReductionMsg *msg)
{
  ptr->processGatherMsg(msg, buf, type, count);
  statusIreq = true;
  comm = ptr->getComm();
#if CMK_BIGSIM_CHARM
  event = msg->event;
  eventPe = msg->eventPe;
#endif
  // ampi::rednResult is a [nokeep] entry method, so do not delete msg
}

void GathervReq::receive(ampi *ptr, CkReductionMsg *msg)
{
  ptr->processGathervMsg(msg, buf, type, &recvCounts[0], &displs[0]);
  statusIreq = true;
  comm = ptr->getComm();
#if CMK_BIGSIM_CHARM
  event = msg->event;
  eventPe = msg->eventPe;
#endif
  // ampi::rednResult is a [nokeep] entry method, so do not delete msg
}

CDECL
int AMPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *sts)
{
  AMPIAPI("AMPI_Request_get_status");
  testRequestNoFree(&request, flag, sts);
  if(*flag != 1)
    AMPI_Yield(MPI_COMM_WORLD);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Test(MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("AMPI_Test");
  testRequest(request, flag, sts);
  if(*flag != 1)
    AMPI_Yield(MPI_COMM_WORLD);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Testany(int count, MPI_Request *request, int *index, int *flag, MPI_Status *sts){
  AMPIAPI("AMPI_Testany");
  checkRequests(count,request);

  MPI_Status tempStatus;
  if(!sts) sts = &tempStatus;

  if(areInactiveReqs(count,request)){
    *flag=1;
    *index=MPI_UNDEFINED;
    stsempty(*sts);
    return MPI_SUCCESS;
  }
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  *flag=0;
  for(int i=0;i<reqvec->size();i++){
    testRequest(&request[((*reqvec)[i])[0]], flag, sts);
    if(*flag==1 && sts->MPI_COMM!=0){ // skip MPI_REQUEST_NULL
      *index = ((*reqvec)[i])[0];
      return MPI_SUCCESS;
    }
  }
  *index = MPI_UNDEFINED;
  delete reqvec;
  AMPI_Yield(MPI_COMM_WORLD);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Testall(int count, MPI_Request *request, int *flag, MPI_Status *sts)
{
  AMPIAPI("AMPI_Testall");
  if(count==0){
    *flag = 1;
    return MPI_SUCCESS;
  }
  checkRequests(count,request);
  int i,j,tmpflag;
  AmpiRequestList* reqs = getReqs();
  CkVec<CkVec<int> > *reqvec = vecIndex(count,request);
  *flag = 1;
  for(i=0;i<reqvec->size();i++){
    for(j=0;j<((*reqvec)[i]).size();j++){
      if(request[((*reqvec)[i])[j]] == MPI_REQUEST_NULL)
        continue;
      tmpflag = (*reqs)[request[((*reqvec)[i])[j]]]->itest(&sts[((*reqvec)[i])[j]]);
      *flag *= tmpflag;
    }
  }
  delete reqvec;
  if(*flag)
    AMPI_Waitall(count,request,sts);
  else
    AMPI_Yield(MPI_COMM_WORLD);
  return MPI_SUCCESS;
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
    testRequest(&array_of_requests[((*reqvec)[i])[0]], &flag, &sts);
    if(flag == 1){
      array_of_indices[(*outcount)]=((*reqvec)[i])[0];
      (*outcount)++;
      if(array_of_statuses){
        array_of_statuses[(*outcount)]=sts;
      }
    }
  }
  delete reqvec;
  if(*outcount==0)
    AMPI_Yield(MPI_COMM_WORLD);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Request_free(MPI_Request *request){
  AMPIAPI("AMPI_Request_free");
  if(*request==MPI_REQUEST_NULL) return MPI_SUCCESS;
  checkRequest(*request);
  AmpiRequestList* reqs = getReqs();
  reqs->free(*request);
  *request = MPI_REQUEST_NULL;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Cancel(MPI_Request *request){
  AMPIAPI("AMPI_Cancel");
  return AMPI_Request_free(request);
}

CDECL
int AMPI_Test_cancelled(MPI_Status* status, int* flag) {
  AMPIAPI("AMPI_Test_cancelled");
  /* FIXME: always returns success */
  *flag = 1;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Status_set_cancelled(MPI_Status *status, int flag){
  AMPIAPI("AMPI_Status_set_cancelled");
  /* AMPI_Test_cancelled always returns true */
  return MPI_SUCCESS;
}

CDECL
int AMPI_Recv_init(void *buf, int count, MPI_Datatype type, int src, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("AMPI_Recv_init");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Recv_init", comm, 1, count, 1, type, 1, tag, 1, src, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *req = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  AmpiRequestList* reqs = getReqs();
  PersReq *newreq = new PersReq(buf,count,type,src,tag,comm,2);
  *req = reqs->insert(newreq);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Send_init(void *buf, int count, MPI_Datatype type, int dest, int tag,
                   MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("AMPI_Send_init");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Send_init", comm, 1, count, 1, type, 1, tag, 1, dest, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *req = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  AmpiRequestList* reqs = getReqs();
  PersReq *newreq = new PersReq(buf,count,type,dest,tag,comm,1);
  *req = reqs->insert(newreq);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Ssend_init(void *buf, int count, MPI_Datatype type, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req)
{
  AMPIAPI("AMPI_Ssend_init");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Ssend_init", comm, 1, count, 1, type, 1, tag, 1, dest, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *req = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  AmpiRequestList* reqs = getReqs();
  PersReq *newreq = new PersReq(buf,count,type,dest,tag,comm,3);
  *req = reqs->insert(newreq);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_contiguous(int count, MPI_Datatype oldtype,
                         MPI_Datatype *newtype)
{
  AMPIAPI("AMPI_Type_contiguous");
  getDDT()->newContiguous(count, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_vector(int count, int blocklength, int stride,
                     MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_vector");
  getDDT()->newVector(count, blocklength, stride, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride,
                             MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_create_hvector");
  getDDT()->newHVector(count, blocklength, stride, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_hvector(int count, int blocklength, MPI_Aint stride,
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_hvector");
  return AMPI_Type_create_hvector(count, blocklength, stride, oldtype, newtype);
}

CDECL
int AMPI_Type_indexed(int count, int* arrBlength, int* arrDisp,
                      MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_indexed");
  /*CkDDT_Indexed's arrDisp has type MPI_Aint* (not int*). */
  vector<MPI_Aint> arrDispAint(count);
  for(int i=0; i<count; i++)
    arrDispAint[i] = (MPI_Aint)(arrDisp[i]);
  getDDT()->newIndexed(count, arrBlength, &arrDispAint[0], oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_create_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp,
                              MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_create_hindexed");
  getDDT()->newHIndexed(count, arrBlength, arrDisp, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_hindexed(int count, int* arrBlength, MPI_Aint* arrDisp,
                       MPI_Datatype oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_hindexed");
  return AMPI_Type_create_hindexed(count, arrBlength, arrDisp, oldtype, newtype);
}

CDECL
int AMPI_Type_create_indexed_block(int count, int Blength, MPI_Aint *arr,
                                   MPI_Datatype oldtype, MPI_Datatype *newtype)
{
  AMPIAPI("AMPI_Type_create_indexed_block");
  getDDT()->newIndexedBlock(count,Blength, arr, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_create_hindexed_block(int count, int Blength, MPI_Aint *arr,
                                    MPI_Datatype oldtype, MPI_Datatype *newtype)
{
  AMPIAPI("AMPI_Type_create_hindexed_block");
  getDDT()->newHIndexedBlock(count,Blength, arr, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_create_struct(int count, int* arrBlength, MPI_Aint* arrDisp,
                            MPI_Datatype* oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_create_struct");
  getDDT()->newStruct(count, arrBlength, arrDisp, oldtype, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_struct(int count, int* arrBlength, MPI_Aint* arrDisp,
                     MPI_Datatype* oldtype, MPI_Datatype*  newtype)
{
  AMPIAPI("AMPI_Type_struct");
  return AMPI_Type_create_struct(count, arrBlength, arrDisp, oldtype, newtype);
}

CDECL
int AMPI_Type_commit(MPI_Datatype *datatype)
{
  AMPIAPI("AMPI_Type_commit");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_free(MPI_Datatype *datatype)
{
  AMPIAPI("AMPI_Type_free");
  getDDT()->freeType(datatype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent)
{
  AMPIAPI("AMPI_Type_get_extent");
  *lb = getDDT()->getLB(datatype);
  *extent = getDDT()->getExtent(datatype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent)
{
  AMPIAPI("AMPI_Type_extent");
  MPI_Aint tmpLB;
  return AMPI_Type_get_extent(datatype, &tmpLB, extent);
}

CDECL
int AMPI_Type_size(MPI_Datatype datatype, int *size)
{
  AMPIAPI("AMPI_Type_size");
  *size=getDDT()->getSize(datatype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_set_name(MPI_Datatype datatype, const char *name)
{
  AMPIAPI("AMPI_Type_set_name");
  getDDT()->setName(datatype, name);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_get_name(MPI_Datatype datatype, char *name, int *resultlen)
{
  AMPIAPI("AMPI_Type_get_name");
  getDDT()->getName(datatype, name, resultlen);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype)
{
  AMPIAPI("AMPI_Type_create_resized");
  getDDT()->createResized(oldtype, lb, extent, newtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Isend(void *buf, int count, MPI_Datatype type, int dest,
               int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Isend");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Isend", comm, 1, count, 1, type, 1, tag, 1, dest, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char *)request, sizeof(MPI_Request));
    return MPI_SUCCESS;
  }
#endif

  USER_CALL_DEBUG("AMPI_Isend("<<type<<","<<dest<<","<<tag<<","<<comm<<")");
  ampi *ptr = getAmpiInstance(comm);
  ptr->send(tag, ptr->getRank(comm), buf, count, type, dest, comm);
  *request = ptr->postReq(new SendReq(comm), AMPI_REQ_COMPLETED);

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char *)request, sizeof(MPI_Request));
  }
#endif

  return MPI_SUCCESS;
}

void ampi::irecv(void *buf, int count, MPI_Datatype type, int src,
                 int tag, MPI_Comm comm, MPI_Request *request)
{
  if (src==MPI_PROC_NULL) {
    *request = MPI_REQUEST_NULL;
    return;
  }
  AmpiRequestList* reqs = getReqs();
  IReq *newreq = new IReq(buf, count, type, src, tag, comm);
  *request = reqs->insert(newreq);

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char *)request, sizeof(MPI_Request));
    return MPI_SUCCESS;
  }
#endif

  AmpiMsg *msg = NULL;
  msg = getMessage(tag, src, comm, &newreq->tag);
  // if msg has already arrived, do the receive right away
  if (msg) {
    newreq->receive(this, msg);
  }
  // ... otherwise post the receive
  else {
    int tags[3];
    tags[0] = tag; tags[1] = src; tags[2] = comm;

    //just insert the index of the newreq in the ampiParent::ampiReqs
    //to posted_ireqs. Such change is due to the need for Out-of-core Emulation
    //in BigSim. Before this change, posted_ireqs and ampiReqs both hold pointers to
    //AmpiRequest instances. After going through the Pupping routines, both will have
    //pointers to different AmpiRequest instances and no longer refer to the same AmpiRequest
    //instance. Therefore, to keep both always accessing the same AmpiRequest instance,
    //posted_ireqs stores the index (an integer) to ampiReqs.
    //The index is 1-based rather 0-based because when pulling entries from posted_ireqs,
    //if not found, a "0" (i.e. NULL) is returned, this confuses the indexing of ampiReqs.
    CmmPut(posted_ireqs, 3, tags, (void *)(CmiIntPtr)((*request)+1));
  }

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char *)request, sizeof(MPI_Request));
  }
#endif
}

CDECL
int AMPI_Irecv(void *buf, int count, MPI_Datatype type, int src,
               int tag, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Irecv");

  handle_MPI_BOTTOM(buf, type);

#if AMPI_ERROR_CHECKING
  int ret = errorCheck("AMPI_Irecv", comm, 1, count, 1, type, 1, tag, 1, src, 1, buf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  USER_CALL_DEBUG("AMPI_Irecv("<<type<<","<<src<<","<<tag<<","<<comm<<")");
  ampi *ptr = getAmpiInstance(comm);

  ptr->irecv(buf, count, type, src, tag, comm, request);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ireduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op op,
                 int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ireduce");

  handle_MPI_BOTTOM(sendbuf, type, recvbuf, type);
  handle_MPI_IN_PLACE(sendbuf, recvbuf);

#if AMPI_ERROR_CHECKING
  if(op == MPI_OP_NULL)
    return ampiErrhandler("AMPI_Ireduce", MPI_ERR_OP);
  int ret = errorCheck("AMPI_Ireduce", comm, 1, count, 1, type, 1, 0, 0, root, 1, sendbuf, 1,
                       recvbuf, getAmpiInstance(comm)->getRank(comm) == root);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new RednReq(recvbuf, count, type, comm, op), AMPI_REQ_COMPLETED);
    return copyDatatype(comm,type,count,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Ireduce for Inter-communicators!");

  CkReductionMsg *msg=makeRednMsg(ptr->getDDT()->getType(type),sendbuf,count,type,ptr->getRank(comm),op);
  int rootIdx=ptr->comm2CommStruct(comm).getIndexForRank(root);

  CkCallback reduceCB(CkIndex_ampi::irednResult(0),CkArrayIndex1D(rootIdx),ptr->getProxy());
  msg->setCallback(reduceCB);
  ptr->contribute(msg);

  if (ptr->thisIndex == rootIdx){
    // use a RednReq to non-block the caller and get a request ptr
    *request = ptr->postReq(new RednReq(recvbuf,count,type,comm,op));
  }

  return MPI_SUCCESS;
}

static CkReductionMsg *makeGatherMsg(const void *inbuf, int count, MPI_Datatype type, int rank)
{
  CkDDT_DataType* ddt = getDDT()->getType(type);
  int szdata = ddt->getSize(count);
  const int tupleSize = 2;
  CkReduction::tupleElement tupleRedn[tupleSize];
  tupleRedn[0] = CkReduction::tupleElement(sizeof(int), &rank, CkReduction::set);

  if (ddt->isContig()) {
    tupleRedn[1] = CkReduction::tupleElement(szdata, (void*)inbuf, CkReduction::set);
  } else {
    vector<char> sbuf(szdata);
    ddt->serialize((char*)inbuf, &sbuf[0], count, 1);
    tupleRedn[1] = CkReduction::tupleElement(szdata, &sbuf[0], CkReduction::set);
  }

  return CkReductionMsg::buildFromTuple(tupleRedn, tupleSize);
}

CDECL
int AMPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm)
{
  AMPIAPI("AMPI_Allgather");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Allgather", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Allgather", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Allgather for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  int rank = ptr->getRank(comm);
  int sendSize = ptr->getDDT()->getType(sendtype)->getSize(sendcount);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback allgatherCB(CkIndex_ampi::rednResult(0), ptr->getProxy());
  msg->setCallback(allgatherCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Allgather called on comm %d\n", ptr->thisIndex, comm));
  ptr->contribute(msg);

  ptr->blockOnRedn(new GatherReq(recvbuf, recvcount, recvtype, comm));

  return MPI_SUCCESS;
}

CDECL
int AMPI_Iallgather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request* request)
{
  AMPIAPI("AMPI_Iallgather");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Iallgather", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Iallgather", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new GatherReq(recvbuf, recvcount, recvtype, comm), AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Iallgather for Inter-communicators!");

  int rank = ptr->getRank(comm);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback allgatherCB(CkIndex_ampi::irednResult(0), ptr->getProxy());
  msg->setCallback(allgatherCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Iallgather called on comm %d\n", ptr->thisIndex, comm));
  ptr->contribute(msg);

  // use a RednReq to non-block the caller and get a request ptr
  *request = ptr->postReq(new GatherReq(recvbuf, recvcount, recvtype, comm));

  return MPI_SUCCESS;
}

CDECL
int AMPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int *recvcounts, int *displs,
                    MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Allgatherv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Allgatherv", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Allgatherv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Allgatherv for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  int rank = ptr->getRank(comm);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback allgathervCB(CkIndex_ampi::rednResult(0), ptr->getProxy());
  msg->setCallback(allgathervCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Allgatherv called on comm %d\n", ptr->thisIndex, comm));
  ptr->contribute(msg);

  ptr->blockOnRedn(new GathervReq(recvbuf, ptr->getSize(comm), recvtype, comm, recvcounts, displs));

  return MPI_SUCCESS;
}

CDECL
int AMPI_Iallgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int *recvcounts, int *displs,
                     MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Iallgatherv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Iallgatherv", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Iallgatherv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank = ptr->getRank(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new GathervReq(recvbuf, rank, recvtype, comm, recvcounts, displs),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Iallgatherv for Inter-communicators!");

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback allgathervCB(CkIndex_ampi::irednResult(0), ptr->getProxy());
  msg->setCallback(allgathervCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Iallgatherv called on comm %d\n", ptr->thisIndex, comm));
  ptr->contribute(msg);

  // use a GathervReq to non-block the caller and get a request ptr
  *request = ptr->postReq(new GathervReq(recvbuf, ptr->getSize(comm), recvtype,
                                         comm, recvcounts, displs));

  return MPI_SUCCESS;
}

CDECL
int AMPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Gather");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Gather", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Gather", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
    if(ret != MPI_SUCCESS)
      return ret;
  }
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Gather for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)recvbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rootIdx = ptr->comm2CommStruct(comm).getIndexForRank(root);
  int rank = ptr->getRank(comm);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback gatherCB(CkIndex_ampi::rednResult(0), CkArrayIndex1D(rootIdx), ptr->getProxy());
  msg->setCallback(gatherCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Gather called on comm %d root %d \n", ptr->thisIndex, comm, rootIdx));
  ptr->contribute(msg);

  if(rank==root) {
    ptr->blockOnRedn(new GatherReq(recvbuf, recvcount, recvtype, comm));
  }

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcount * size;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)recvbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Igather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Igather");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Igather", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Igather", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
    if(ret != MPI_SUCCESS){
      *request = MPI_REQUEST_NULL;
      return ret;
    }
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new GatherReq(recvbuf, recvcount, recvtype, comm), AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Igather for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)recvbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  int rootIdx = ptr->comm2CommStruct(comm).getIndexForRank(root);
  int rank = ptr->getRank(comm);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback gatherCB(CkIndex_ampi::irednResult(0), CkArrayIndex1D(rootIdx), ptr->getProxy());
  msg->setCallback(gatherCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Igather called on comm %d root %d \n", ptr->thisIndex, comm, rootIdx));
  ptr->contribute(msg);

  if(rank==root) {
    // use a GatherReq to non-block the caller and get a request ptr
    *request = ptr->postReq(new GatherReq(recvbuf, recvcount, recvtype, comm));
  }
  else {
    *request = MPI_REQUEST_NULL;
  }

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcount * size;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)recvbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int *recvcounts, int *displs,
                 MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Gatherv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Gatherv", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Gatherv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
    if(ret != MPI_SUCCESS)
      return ret;
  }
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Gatherv for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    int commsize;
    int itemsize = getDDT()->getSize(recvtype);
    (*(pptr->fromPUPer))|commsize;
    for(int i=0;i<commsize;i++){
      (*(pptr->fromPUPer))|(pptr->pupBytes);
      PUParray(*(pptr->fromPUPer), (char *)(((char*)recvbuf)+(itemsize*displs[i])), (pptr->pupBytes));
    }
    return MPI_SUCCESS;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rootIdx = ptr->comm2CommStruct(comm).getIndexForRank(root);
  int rank = ptr->getRank(comm);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback gathervCB(CkIndex_ampi::rednResult(0), CkArrayIndex1D(rootIdx), ptr->getProxy());
  msg->setCallback(gathervCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Gatherv called on comm %d root %d \n", ptr->thisIndex, comm, rootIdx));
  ptr->contribute(msg);

  if(rank==root) {
    ptr->blockOnRedn(new GathervReq(recvbuf, ptr->getSize(comm), recvtype, comm, recvcounts, displs));
  }

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    for(int i=0;i<size;i++){
      (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcounts[i];
      (*(pptr->toPUPer))|(pptr->pupBytes);
      PUParray(*(pptr->toPUPer), (char *)(((char*)recvbuf)+(itemsize*displs[i])), (pptr->pupBytes));
    }
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Igatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int *recvcounts, int *displs,
                  MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Igatherv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Igatherv", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Igatherv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
    if(ret != MPI_SUCCESS){
      *request = MPI_REQUEST_NULL;
      return ret;
    }
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank = ptr->getRank(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new GathervReq(recvbuf, rank, recvtype, comm, recvcounts, displs),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Igatherv for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    int commsize;
    int itemsize = getDDT()->getSize(recvtype);
    (*(pptr->fromPUPer))|commsize;
    for(int i=0;i<commsize;i++){
      (*(pptr->fromPUPer))|(pptr->pupBytes);
      PUParray(*(pptr->fromPUPer), (char *)(((char*)recvbuf)+(itemsize*displs[i])), (pptr->pupBytes));
    }
    return MPI_SUCCESS;
  }
#endif

  int rootIdx = ptr->comm2CommStruct(comm).getIndexForRank(root);

  CkReductionMsg* msg = makeGatherMsg(sendbuf, sendcount, sendtype, rank);
  CkCallback gathervCB(CkIndex_ampi::irednResult(0), CkArrayIndex1D(rootIdx), ptr->getProxy());
  msg->setCallback(gathervCB);
  MSG_ORDER_DEBUG(CkPrintf("[%d] AMPI_Igatherv called on comm %d root %d \n", ptr->thisIndex, comm, rootIdx));
  ptr->contribute(msg);

  if(rank==root) {
    // use a GathervReq to non-block the caller and get a request ptr
    *request = ptr->postReq(new GathervReq(recvbuf, ptr->getSize(comm), recvtype,
                                           comm, recvcounts, displs));
  }
  else {
    *request = MPI_REQUEST_NULL;
  }

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    for(int i=0;i<size;i++){
      (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcounts[i];
      (*(pptr->toPUPer))|(pptr->pupBytes);
      PUParray(*(pptr->toPUPer), (char *)(((char*)recvbuf)+(itemsize*displs[i])), (pptr->pupBytes));
    }
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Scatter");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Scatter", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
    if(ret != MPI_SUCCESS)
      return ret;
  }
  ret = errorCheck("AMPI_Scatter", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Scatter for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)recvbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

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
  }

  if(-1==ptr->recv(MPI_SCATTER_TAG, root, recvbuf, recvcount, recvtype, comm))
    CkAbort("AMPI> Error in MPI_Scatter recv");

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcount;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)recvbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Iscatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Iscatter");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Iscatter", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
    if(ret != MPI_SUCCESS){
      *request = MPI_REQUEST_NULL;
      return ret;
    }
  }
  ret = errorCheck("AMPI_Iscatter", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,root,MPI_SCATTER_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Iscatter for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)recvbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  int size = ptr->getSize(comm);
  int i;

  if(ptr->getRank(comm)==root) {
    CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
    int itemsize = dttype->getSize(sendcount) ;
    for(i=0;i<size;i++) {
      ptr->send(MPI_SCATTER_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i),
                sendcount, sendtype, i, comm);
    }
  }

  // use an IReq to non-block the caller and get a request ptr
  *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,root,MPI_SCATTER_TAG,comm));

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcount;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)recvbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root, MPI_Comm comm)
{
  AMPIAPI("AMPI_Scatterv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf, recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Scatterv", comm, 1, 0, 0, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
    if(ret != MPI_SUCCESS)
      return ret;
  }
  ret = errorCheck("AMPI_Scatterv", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcounts[0],sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Scatterv for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)recvbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

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
  }

  if(-1==ptr->recv(MPI_SCATTER_TAG, root, recvbuf, recvcount, recvtype, comm))
    CkAbort("AMPI> Error in MPI_Scatterv recv");

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcount;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)recvbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Iscatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Iscatterv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  if (getAmpiInstance(comm)->getRank(comm) == root) {
    ret = errorCheck("AMPI_Iscatterv", comm, 1, 0, 0, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
    if(ret != MPI_SUCCESS){
      *request = MPI_REQUEST_NULL;
      return ret;
    }
  }
  ret = errorCheck("AMPI_Iscatterv", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,root,MPI_SCATTER_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcounts[0],sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Iscatterv for Inter-communicators!");

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    (*(pptr->fromPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->fromPUPer), (char *)recvbuf, (pptr->pupBytes));
    return MPI_SUCCESS;
  }
#endif

  int size = ptr->getSize(comm);
  int i;

  if(ptr->getRank(comm) == root) {
    CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
    int itemsize = dttype->getSize() ;
    for(i=0;i<size;i++) {
      ptr->send(MPI_SCATTER_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*displs[i]),
                sendcounts[i], sendtype, i, comm);
    }
  }

  // use an IReq to non-block the caller and get a request ptr
  *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,root,MPI_SCATTER_TAG,comm));

#if AMPIMSGLOG
  if(msgLogWrite && record_msglog(pptr->thisIndex)){
    (pptr->pupBytes) = getDDT()->getSize(recvtype) * recvcount;
    (*(pptr->toPUPer))|(pptr->pupBytes);
    PUParray(*(pptr->toPUPer), (char *)recvbuf, (pptr->pupBytes));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoall");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Alltoall", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Alltoall", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("AMPI does not implement MPI_IN_PLACE for MPI_Alltoall!");
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Alltoall for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  CkDDT_DataType *dttype;
  int itemsize;
  int i;

  dttype = ptr->getDDT()->getType(sendtype) ;
  itemsize = dttype->getSize(sendcount) ;
  int rank = ptr->getRank(comm);
  int comm_size = size;
  MPI_Status status;

#if CMK_BIGSIM_CHARM
  TRACE_BG_AMPI_LOG(MPI_ALLTOALL, itemsize);
#endif

  if( itemsize <= AMPI_ALLTOALL_SHORT_MSG ){
    /* Short message. Use recursive doubling. Each process sends all
       its data at each step along with all data it received in
       previous steps. */

    /* need to allocate temporary buffer of size
       sendbuf_extent*comm_size */

    int sendtype_extent = getDDT()->getExtent(sendtype);
    int recvtype_extent = getDDT()->getExtent(recvtype);
    int sendbuf_extent = sendcount * comm_size * sendtype_extent;

    vector<char> tmp_buf(sendbuf_extent*comm_size);

    /* copy local sendbuf into tmp_buf at location indexed by rank */
    int curr_cnt = sendcount*comm_size;
    copyDatatype(comm, sendtype, curr_cnt, sendbuf,
                 (&tmp_buf[0] + rank*sendbuf_extent));

    int mask = 0x1;
    int dst,tree_root,dst_tree_root,my_tree_root;
    int last_recv_cnt,nprocs_completed;
    int j,k,tmp_mask;
    i = 0;
    while (mask < comm_size) {
      dst = rank ^ mask;

      dst_tree_root = dst >> i;
      dst_tree_root <<= i;

      my_tree_root = rank >> i;
      my_tree_root <<= i;

      if (dst < comm_size) {
        ptr->sendrecv((&tmp_buf[0] + my_tree_root*sendbuf_extent),
                      curr_cnt, sendtype, dst, MPI_ATA_SEQ_TAG,
                      (&tmp_buf[0] + dst_tree_root*sendbuf_extent),
                      sendcount*comm_size*mask, sendtype, dst,
                      MPI_ATA_SEQ_TAG, comm, &status);

        /* in case of non-power-of-two nodes, less data may be
           received than specified */
        AMPI_Get_count(&status, sendtype, &last_recv_cnt);
        curr_cnt += last_recv_cnt;
      }

      /* if some processes in this process's subtree in this step
         did not have any destination process to communicate with
         because of non-power-of-two, we need to send them the
         result. We use a logarithmic recursive-halfing algorithm
         for this. */

      if (dst_tree_root + mask > comm_size) {
        nprocs_completed = comm_size - my_tree_root - mask;
        /* nprocs_completed is the number of processes in this
           subtree that have all the data. Send data to others
           in a tree fashion. First find root of current tree
           that is being divided into two. k is the number of
           least-significant bits in this process's rank that
           must be zeroed out to find the rank of the root */
        j = mask;
        k = 0;
        while (j) {
          j >>= 1;
          k++;
        }
        k--;

        tmp_mask = mask >> 1;
        while (tmp_mask) {
          dst = rank ^ tmp_mask;

          tree_root = rank >> k;
          tree_root <<= k;

          /* send only if this proc has data and destination
             doesn't have data. at any step, multiple processes
             can send if they have the data */
          if ((dst > rank) &&
              (rank < tree_root + nprocs_completed)
              && (dst >= tree_root + nprocs_completed)) {
            /* send the data received in this step above */
            ptr->send(MPI_ATA_SEQ_TAG, ptr->getRank(comm),
                      (&tmp_buf[0] + dst_tree_root * sendbuf_extent),
                      last_recv_cnt, sendtype, dst, comm);
          }
          /* recv only if this proc. doesn't have data and sender
             has data */
          else if ((dst < rank) &&
              (dst < tree_root + nprocs_completed) &&
              (rank >= tree_root + nprocs_completed)) {
            if(-1==ptr->recv(MPI_ATA_SEQ_TAG, dst, &tmp_buf[0] + dst_tree_root*sendbuf_extent,
                             sendcount*comm_size*mask, sendtype, comm, &status))
              CkAbort("AMPI> Error in MPI_Alltoall");
            AMPI_Get_count(&status, sendtype, &last_recv_cnt);
            curr_cnt += last_recv_cnt;
          }
          tmp_mask >>= 1;
          k--;
        }
      }

      mask <<= 1;
      i++;
    }

    /* now copy everyone's contribution from tmp_buf to recvbuf */
    for (int p=0; p<comm_size; p++) {
      copyDatatype(comm,sendtype,sendcount,
                   (&tmp_buf[0] + p*sendbuf_extent + rank*sendcount*sendtype_extent),
                   ((char*)recvbuf + p*recvcount*recvtype_extent));
    }

  }else if ( itemsize <= AMPI_ALLTOALL_MEDIUM_MSG ) {
    for(i=0;i<size;i++) {
      int dst = (rank+i) % size;
      ptr->send(MPI_ATA_TAG, rank, ((char*)sendbuf)+(itemsize*dst), sendcount,
                sendtype, dst, comm);
    }
    dttype = ptr->getDDT()->getType(recvtype) ;
    itemsize = dttype->getSize(recvcount) ;
    for(i=0;i<size;i++) {
      int dst = (rank+i) % size;
      if(-1==ptr->recv(MPI_ATA_TAG, dst, ((char*)recvbuf)+(itemsize*dst), recvcount,
                       recvtype, comm))
        CkAbort("AMPI> Error in MPI_Alltoall");
    }
  } else { // large messages
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
     ptr->sendrecv(((char *)sendbuf + dst*itemsize), sendcount, sendtype, dst, MPI_ATA_TAG,
                   ((char *)recvbuf + src*itemsize), recvcount, recvtype, src, MPI_ATA_TAG,
                   comm, &status);
    } // end of large message
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Alltoall_iget(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoall_iget");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Alltoall_iget", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Alltoall_iget", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  if(sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("AMPI does not implement MPI_IN_PLACE for MPI_Alltoall_iget!");
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Alltoall_iget for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  CProxy_ampi pa(ptr->ckGetArrayID());
  int size = ptr->getSize(comm);
  CkDDT_DataType *dttype;
  int itemsize;
  int recvdisp;
  int myrank;
  int i;
  // Set flags for others to get
  ptr->setA2AIgetFlag((void*)sendbuf);
  MPI_Comm_rank(comm,&myrank);
  recvdisp = myrank*recvcount;

  ptr->barrier();
  // post receives
  vector<MPI_Request> reqs(size);
  for(i=0;i<size;i++) {
    reqs[i] = pa[i].Alltoall_RemoteIget(recvdisp, recvcount, recvtype, MPI_ATA_TAG);
  }

  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize(recvcount) ;
  AmpiMsg *msg;
  for(i=0;i<size;i++) {
    msg = (AmpiMsg*)CkWaitReleaseFuture(reqs[i]);
    memcpy((char*)recvbuf+(itemsize*i), msg->data,itemsize);
    delete msg;
  }

  ptr->barrier();

  // Reset flags
  ptr->resetA2AIgetFlag();

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ialltoall");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Ialltoall", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ialltoall", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,ptr->getRank(comm),MPI_ATA_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcount,sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Ialltoall for Inter-communicators!");

  int size = ptr->getSize(comm);
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype);
  int itemsize = dttype->getSize(sendcount);
  int i;
  for(i=0;i<size;i++) {
    ptr->send(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+(itemsize*i), sendcount,
              sendtype, i, comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(size);
  for(i=0;i<size;i++){
    if(newreq->addReq(((char*)recvbuf)+(itemsize*i),recvcount,recvtype,i,MPI_ATA_TAG,comm)!=(i+1))
      CkAbort("MPI_Ialltoall: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);
  AMPI_DEBUG("MPI_Ialltoall: request=%d, reqs.size=%d, &reqs=%d\n",*request,reqs->size(),reqs);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                   MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                   int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoallv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Alltoallv", comm, 1, 0, 0, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Alltoallv", comm, 1, 0, 0, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtype,sendcounts[0],sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Alltoallv for Inter-communicators!");

  ampi *ptr = getAmpiInstance(comm);
  int size = ptr->getSize(comm);
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
  int itemsize = dttype->getSize() ;
  int i;
  for(i=0;i<size;i++)  {
    ptr->send(MPI_ATA_TAG,ptr->getRank(comm),((char*)sendbuf)+(itemsize*sdispls[i]),sendcounts[i],
              sendtype, i, comm);
  }
  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize() ;

  for(i=0;i<size;i++) {
    if(-1==ptr->recv(MPI_ATA_TAG,i,((char*)recvbuf)+(itemsize*rdispls[i]),recvcounts[i],recvtype, comm))
      CkAbort("AMPI> Error in MPI_Alltoallv");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ialltoallv(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
                    void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                    MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ialltoallv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Ialltoallv", comm, 1, 0, 0, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ialltoallv", comm, 1, 0, 0, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(recvbuf,recvcounts[0],recvtype,ptr->getRank(comm),MPI_ATA_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtype,sendcounts[0],sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Ialltoallv for Inter-communicators!");

  int size = ptr->getSize(comm);
  CkDDT_DataType* dttype = ptr->getDDT()->getType(sendtype) ;
  int itemsize = dttype->getSize() ;
  int i;
  for(i=0;i<size;i++)  {
    ptr->send(MPI_ATA_TAG,ptr->getRank(comm),((char*)sendbuf)+(itemsize*sdispls[i]),sendcounts[i],
              sendtype, i, comm);
  }

  dttype = ptr->getDDT()->getType(recvtype) ;
  itemsize = dttype->getSize() ;

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(size);
  for(i=0;i<size;i++){
    if(newreq->addReq((void*)(((char*)recvbuf)+(itemsize*rdispls[i])),recvcounts[i],recvtype,i,MPI_ATA_TAG,comm)!=(i+1))
      CkAbort("MPI_Ialltoallv: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);
  AMPI_DEBUG("MPI_Ialltoallv: request=%d, reqs.size=%d, &reqs=%d\n",*request,reqs->size(),reqs);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Alltoallw(void *sendbuf, int *sendcounts, int *sdispls,
                   MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts,
                   int *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm)
{
  AMPIAPI("AMPI_Alltoallw");

  handle_MPI_BOTTOM(sendbuf, sendtypes[0], recvbuf, recvtypes[0]);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Alltoallw", comm, 1, 0, 0, sendtypes[0], 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Alltoallw", comm, 1, 0, 0, recvtypes[0], 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if(comm==MPI_COMM_SELF)
    return copyDatatype(comm,sendtypes[0],sendcounts[0],sendbuf,recvbuf);
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Alltoallw for Inter-communicators!");

  /* displs are in terms of bytes for Alltoallw (unlike Alltoallv) */
  ampi *ptr = getAmpiInstance(comm);
  int i, size = ptr->getSize(comm);
  for(i=0;i<size;i++){
    ptr->send(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+sdispls[i],
              sendcounts[i], sendtypes[i], i, comm);
  }

  for(i=0;i<size;i++){
    if(-1==ptr->recv(MPI_ATA_TAG, i, ((char*)recvbuf)+rdispls[i], recvcounts[i],
                     recvtypes[i], comm))
      CkAbort("MPI_Alltoallw failed in recv\n");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ialltoallw(void *sendbuf, int *sendcounts, int *sdispls,
                    MPI_Datatype *sendtypes, void *recvbuf, int *recvcounts,
                    int *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm,
                    MPI_Request *request)
{
  AMPIAPI("AMPI_Ialltoallw");

  handle_MPI_BOTTOM(sendbuf, sendtypes[0], recvbuf, recvtypes[0]);
  handle_MPI_IN_PLACE(sendbuf,recvbuf);

#if AMPI_ERROR_CHECKING
  int ret;
  ret = errorCheck("AMPI_Ialltoallw", comm, 1, 0, 0, sendtypes[0], 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ialltoallw", comm, 1, 0, 0, recvtypes[0], 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);

  if(comm==MPI_COMM_SELF){
    *request = ptr->postReq(new IReq(recvbuf,recvcounts[0],recvtypes[0],ptr->getRank(comm),MPI_ATA_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm,sendtypes[0],sendcounts[0],sendbuf,recvbuf);
  }
  if(getAmpiParent()->isInter(comm))
    CkAbort("AMPI does not implement MPI_Ialltoallw for Inter-communicators!");

  /* displs are in terms of bytes for Alltoallw (unlike Alltoallv) */
  int i, size = ptr->getSize(comm);
  for(i=0;i<size;i++){
    ptr->send(MPI_ATA_TAG, ptr->getRank(comm), ((char*)sendbuf)+sdispls[i],
              sendcounts[i], sendtypes[i], i, comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(size);
  for(i=0;i<size;i++){
    if(newreq->addReq((void*)(((char*)recvbuf)+rdispls[i]), recvcounts[i],
                      recvtypes[i], i, MPI_ATA_TAG, comm) != (i+1))
      CkAbort("MPI_Ialltoallw: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Neighbor_alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                           void* recvbuf, int recvcount, MPI_Datatype recvtype,
                           MPI_Comm comm)
{
  AMPIAPI("AMPI_Neighbor_alltoall");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Neighbor_alltoall does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Neighbor_alltoall is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Neighbor_alltoall", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Neighbor_alltoall", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if (comm == MPI_COMM_SELF)
    return copyDatatype(comm, sendtype, sendcount, sendbuf, recvbuf);

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  int itemsize = getDDT()->getType(sendtype)->getSize(sendcount);
  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, (void*)((char*)sendbuf+(itemsize*i)),
              sendcount, sendtype, neighbors[i], comm);
  }
  for (int j=0; j<num_neighbors; j++) {
    if (-1==ptr->recv(MPI_NBOR_TAG, neighbors[j], (void*)(((char*)recvbuf)+(itemsize*j)),
                      recvcount, recvtype, comm))
      CkAbort("AMPI> Error in MPI_Neighbor_alltoall recv");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ineighbor_alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype,
                            MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ineighbor_alltoall");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Ineighbor_alltoall does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Ineighbor_alltoall is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Ineighbor_alltoall", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ineighbor_alltoall", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  if (comm == MPI_COMM_SELF) {
    *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,rank_in_comm,MPI_NBOR_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm, sendtype, sendcount, sendbuf, recvbuf);
  }

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  int itemsize = getDDT()->getType(sendtype)->getSize(sendcount);
  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, (void*)((char*)sendbuf+(itemsize*i)),
              sendcount, sendtype, neighbors[i], comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(num_neighbors);
  for (int j=0; j<num_neighbors; j++) {
    if(newreq->addReq(((char*)recvbuf)+(itemsize*j), recvcount, recvtype,
                      neighbors[j], MPI_NBOR_TAG, comm)!=(j+1))
      CkAbort("MPI_Ineighbor_alltoall: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Neighbor_alltoallv(void* sendbuf, int *sendcounts, int *sdispls,
                            MPI_Datatype sendtype, void* recvbuf, int *recvcounts,
                            int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Neighbor_alltoallv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Neighbor_alltoallv does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Neighbor_alltoallv is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Neighbor_alltoallv", comm, 1, sendcounts[0], 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Neighbor_alltoallv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if (comm == MPI_COMM_SELF)
    return copyDatatype(comm, sendtype, sendcounts[0], sendbuf, recvbuf);

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  int itemsize = getDDT()->getType(sendtype)->getSize();
  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, (void*)((char*)sendbuf+(itemsize*sdispls[i])),
              sendcounts[i], sendtype, neighbors[i], comm);
  }
  for (int j=0; j<num_neighbors; j++) {
    if (-1==ptr->recv(MPI_NBOR_TAG, neighbors[j], (void*)(((char*)recvbuf)+(itemsize*rdispls[j])),
                      recvcounts[j], recvtype, comm))
      CkAbort("AMPI> Error in MPI_Neighbor_alltoallv recv");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ineighbor_alltoallv(void* sendbuf, int *sendcounts, int *sdispls,
                             MPI_Datatype sendtype, void* recvbuf, int *recvcounts,
                             int *rdispls, MPI_Datatype recvtype, MPI_Comm comm,
                             MPI_Request *request)
{
  AMPIAPI("AMPI_Ineighbor_alltoallv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Ineighbor_alltoallv does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Ineighbor_alltoallv is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Ineighbor_alltoallv", comm, 1, sendcounts[0], 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ineighbor_alltoallv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  if (comm == MPI_COMM_SELF) {
    *request = ptr->postReq(new IReq(recvbuf,recvcounts[0],recvtype,rank_in_comm,MPI_NBOR_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm, sendtype, sendcounts[0], sendbuf, recvbuf);
  }

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  int itemsize = getDDT()->getType(sendtype)->getSize();
  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, (void*)((char*)sendbuf+(itemsize*sdispls[i])),
              sendcounts[i], sendtype, neighbors[i], comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(num_neighbors);
  for (int j=0; j<num_neighbors; j++) {
    if(newreq->addReq(((char*)recvbuf)+(itemsize*rdispls[j]), recvcounts[j], recvtype,
                      neighbors[j], MPI_NBOR_TAG, comm)!=(j+1))
      CkAbort("MPI_Ineighbor_alltoallv: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Neighbor_alltoallw(void* sendbuf, int *sendcounts, MPI_Aint *sdispls,
                            MPI_Datatype *sendtypes, void* recvbuf, int *recvcounts,
                            MPI_Aint *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm)
{
  AMPIAPI("AMPI_Neighbor_alltoallw");

  handle_MPI_BOTTOM(sendbuf, sendtypes[0], recvbuf, recvtypes[0]);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Neighbor_alltoallw does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Neighbor_alltoallw is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Neighbor_alltoallw", comm, 1, sendcounts[0], 1, sendtypes[0], 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Neighbor_alltoallw", comm, 1, recvcounts[0], 1, recvtypes[0], 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if (comm == MPI_COMM_SELF)
    return copyDatatype(comm, sendtypes[0], sendcounts[0], sendbuf, recvbuf);

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, (void*)((char*)sendbuf+sdispls[i]),
              sendcounts[i], sendtypes[i], neighbors[i], comm);
  }
  for (int j=0; j<num_neighbors; j++) {
    if (-1==ptr->recv(MPI_NBOR_TAG, neighbors[j], (void*)((char*)recvbuf+rdispls[j]),
                      recvcounts[j], recvtypes[j], comm))
      CkAbort("AMPI> Error in MPI_Neighbor_alltoallv recv");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ineighbor_alltoallw(void* sendbuf, int *sendcounts, MPI_Aint *sdispls,
                             MPI_Datatype *sendtypes, void* recvbuf, int *recvcounts,
                             MPI_Aint *rdispls, MPI_Datatype *recvtypes, MPI_Comm comm,
                             MPI_Request *request)
{
  AMPIAPI("AMPI_Ineighbor_alltoallw");

  handle_MPI_BOTTOM(sendbuf, sendtypes[0], recvbuf, recvtypes[0]);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Ineighbor_alltoallw does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Ineighbor_alltoallw is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Ineighbor_alltoallw", comm, 1, sendcounts[0], 1, sendtypes[0], 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ineighbor_alltoallw", comm, 1, recvcounts[0], 1, recvtypes[0], 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  if (comm == MPI_COMM_SELF) {
    *request = ptr->postReq(new IReq(recvbuf,recvcounts[0],recvtypes[0],rank_in_comm,MPI_NBOR_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm, sendtypes[0], sendcounts[0], sendbuf, recvbuf);
  }

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, (void*)((char*)sendbuf+sdispls[i]),
              sendcounts[i], sendtypes[i], neighbors[i], comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(num_neighbors);
  for (int j=0; j<num_neighbors; j++) {
    if(newreq->addReq((char*)recvbuf+rdispls[j], recvcounts[j], recvtypes[j],
                      neighbors[j], MPI_NBOR_TAG, comm)!=(j+1))
      CkAbort("MPI_Ineighbor_alltoallw: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Neighbor_allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype,
                            MPI_Comm comm)
{
  AMPIAPI("AMPI_Neighbor_allgather");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Neighbor_allgather does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Neighbor_allgather is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Neighbor_allgather", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Neighbor_allgather", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if (comm == MPI_COMM_SELF)
    return copyDatatype(comm, sendtype, sendcount, sendbuf, recvbuf);

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, sendbuf, sendcount, sendtype, neighbors[i], comm);
  }
  int itemsize = getDDT()->getType(recvtype)->getSize(recvcount);
  for (int j=0; j<num_neighbors; j++) {
    if (-1==ptr->recv(MPI_NBOR_TAG, neighbors[j], (void*)(((char*)recvbuf)+(itemsize*j)),
                      recvcount, recvtype, comm))
      CkAbort("AMPI> Error in MPI_Neighbor_allgather recv");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ineighbor_allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ineighbor_allgather");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Ineighbor_allgather does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Ineighbor_allgather is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Ineighbor_allgather", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ineighbor_allgather", comm, 1, recvcount, 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  if (comm == MPI_COMM_SELF) {
    *request = ptr->postReq(new IReq(recvbuf,recvcount,recvtype,rank_in_comm,MPI_NBOR_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm, sendtype, sendcount, sendbuf, recvbuf);
  }

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, sendbuf, sendcount, sendtype, neighbors[i], comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(num_neighbors);
  int itemsize = getDDT()->getType(recvtype)->getSize(recvcount);
  for (int j=0; j<num_neighbors; j++) {
    if(newreq->addReq(((char*)recvbuf)+(itemsize*j), recvcount, recvtype,
                      neighbors[j], MPI_NBOR_TAG, comm)!=(j+1))
      CkAbort("MPI_Ineighbor_allgather: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Neighbor_allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, int *recvcounts, int *displs,
                             MPI_Datatype recvtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Neighbor_allgatherv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Neighbor_allgatherv does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Neighbor_allgatherv is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Neighbor_allgatherv", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
  ret = errorCheck("AMPI_Neighbor_allgatherv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS)
    return ret;
#endif

  if (comm == MPI_COMM_SELF)
    return copyDatatype(comm, sendtype, sendcount, sendbuf, recvbuf);

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, sendbuf, sendcount, sendtype, neighbors[i], comm);
  }
  int itemsize = getDDT()->getType(recvtype)->getSize();
  for (int j=0; j<num_neighbors; j++) {
    if (-1==ptr->recv(MPI_NBOR_TAG, neighbors[j], (void*)(((char*)recvbuf)+(itemsize*displs[j])),
                      recvcounts[j], recvtype, comm))
      CkAbort("AMPI> Error in MPI_Neighbor_allgatherv recv");
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Ineighbor_allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, int* recvcounts, int* displs,
                              MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
  AMPIAPI("AMPI_Ineighbor_allgatherv");

  handle_MPI_BOTTOM(sendbuf, sendtype, recvbuf, recvtype);

#if AMPI_ERROR_CHECKING
  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE)
    CkAbort("MPI_Ineighbor_allgatherv does not accept MPI_IN_PLACE!");
  if (getAmpiParent()->isInter(comm))
    CkAbort("MPI_Ineighbor_allgatherv is not defined for Inter-communicators!");
  int ret;
  ret = errorCheck("AMPI_Ineighbor_allgatherv", comm, 1, sendcount, 1, sendtype, 1, 0, 0, 0, 0, sendbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
  ret = errorCheck("AMPI_Ineighbor_allgatherv", comm, 1, recvcounts[0], 1, recvtype, 1, 0, 0, 0, 0, recvbuf, 1);
  if(ret != MPI_SUCCESS){
    *request = MPI_REQUEST_NULL;
    return ret;
  }
#endif

  ampi *ptr = getAmpiInstance(comm);
  int rank_in_comm = ptr->getRank(comm);

  if (comm == MPI_COMM_SELF) {
    *request = ptr->postReq(new IReq(recvbuf,recvcounts[0],recvtype,rank_in_comm,MPI_NBOR_TAG,comm),
                            AMPI_REQ_COMPLETED);
    return copyDatatype(comm, sendtype, sendcount, sendbuf, recvbuf);
  }

  const vector<int>& neighbors = ptr->getNeighbors();
  int num_neighbors = neighbors.size();

  for (int i=0; i<num_neighbors; i++) {
    ptr->send(MPI_NBOR_TAG, rank_in_comm, sendbuf, sendcount, sendtype, neighbors[i], comm);
  }

  // use an IATAReq to non-block the caller and get a request ptr
  AmpiRequestList* reqs = getReqs();
  IATAReq *newreq = new IATAReq(num_neighbors);
  int itemsize = getDDT()->getType(recvtype)->getSize();
  for (int j=0; j<num_neighbors; j++) {
    if(newreq->addReq(((char*)recvbuf)+(itemsize*displs[j]), recvcounts[j], recvtype,
                      neighbors[j], MPI_NBOR_TAG, comm)!=(j+1))
      CkAbort("MPI_Ineighbor_allgatherv: Error adding requests into IATAReq!");
  }
  *request = reqs->insert(newreq);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
{
  AMPIAPI("AMPI_Comm_dup");

  int topol;
  ampi *ptr = getAmpiInstance(comm);
  int rank = ptr->getRank(comm);

  AMPI_Topo_test(comm, &topol);
  if (topol == MPI_CART) {
    ptr->split(0, rank, newcomm, MPI_CART);

    // duplicate cartesian topology info
    ampiCommStruct &c = getAmpiParent()->getCart(comm);
    ampiCommStruct &newc = getAmpiParent()->getCart(*newcomm);
    newc.setndims(c.getndims());
    newc.setdims(c.getdims());
    newc.setperiods(c.getperiods());
    newc.setnbors(c.getnbors());
  }
  else {
    ptr->split(0, rank, newcomm, MPI_UNDEFINED /*not MPI_CART*/);
  }
  ptr->barrier();

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char *)newcomm, sizeof(int));
    return MPI_SUCCESS;
  }
  else if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char *)newcomm, sizeof(int));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_split(MPI_Comm src, int color, int key, MPI_Comm *dest)
{
  AMPIAPI("AMPI_Comm_split");

  {
    ampi *ptr = getAmpiInstance(src);
    ptr->split(color, key, dest, MPI_UNDEFINED /*not MPI_CART*/);
    ptr->barrier(); // to prevent race condition in the new comm
  }
  if (color == MPI_UNDEFINED) *dest = MPI_COMM_NULL;

#if AMPIMSGLOG
  ampiParent* pptr = getAmpiParent();
  if(msgLogRead){
    PUParray(*(pptr->fromPUPer), (char *)dest, sizeof(int));
    return MPI_SUCCESS;
  }
  else if(msgLogWrite && record_msglog(pptr->thisIndex)){
    PUParray(*(pptr->toPUPer), (char *)dest, sizeof(int));
  }
#endif

  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_split_type(MPI_Comm src, int split_type, int key, MPI_Info info, MPI_Comm *dest)
{
  AMPIAPI("AMPI_Comm_split_type");

  if (src == MPI_COMM_SELF && split_type == MPI_UNDEFINED) {
    *dest = MPI_COMM_NULL;
    return MPI_SUCCESS;
  }

  int color = MPI_UNDEFINED;

  if (split_type == MPI_COMM_TYPE_SHARED || split_type == AMPI_COMM_TYPE_HOST) {
    color = CmiPhysicalNodeID(CkMyPe());
  }
  else if (split_type == AMPI_COMM_TYPE_PROCESS) {
    color = CkMyNode();
  }
  else if (split_type == AMPI_COMM_TYPE_WTH) {
    color = CkMyPe();
  }

  if (color == MPI_UNDEFINED) {
    *dest = MPI_COMM_NULL;
    return ampiErrhandler("MPI_Comm_split_type", MPI_ERR_ARG);
  }

  return AMPI_Comm_split(src, color, key, dest);
}

CDECL
int AMPI_Comm_free(MPI_Comm *comm)
{
  AMPIAPI("AMPI_Comm_free");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_test_inter(MPI_Comm comm, int *flag){
  AMPIAPI("AMPI_Comm_test_inter");
  *flag = getAmpiParent()->isInter(comm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_remote_size(MPI_Comm comm, int *size){
  AMPIAPI("AMPI_Comm_remote_size");
  *size = getAmpiParent()->getRemoteSize(comm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group){
  AMPIAPI("AMPI_Comm_remote_group");
  *group = getAmpiParent()->getRemoteGroup(comm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Intercomm_create(MPI_Comm lcomm, int lleader, MPI_Comm rcomm, int rleader,
                          int tag, MPI_Comm *newintercomm){
  AMPIAPI("AMPI_Intercomm_create");

#if AMPI_ERROR_CHECKING
  if (getAmpiParent()->isInter(lcomm) || getAmpiParent()->isInter(rcomm))
    return ampiErrhandler("AMPI_Intercomm_create", MPI_ERR_COMM);
#endif

  ampi *lptr = getAmpiInstance(lcomm);
  ampi *rptr = getAmpiInstance(rcomm);
  int root = lptr->getIndexForRank(lleader);
  int lsize = lptr->getSize(lcomm);
  int lrank = lptr->getRank(lcomm);
  vector<int> rvec;

  if(lrank==lleader){
    int rsize;
    MPI_Status sts;
    vector<int> lvec = lptr->getIndices();

    // local leader exchanges groupStruct with remote leader
    lptr->send(tag, rptr->getRank(rcomm), &lvec[0], lvec.size(), MPI_INT, rleader, rcomm);
    rptr->probe(tag, rleader, rcomm, &sts);
    AMPI_Get_count(&sts, MPI_INT, &rsize);
    rvec.resize(rsize);
    if(-1==rptr->recv(tag, rleader, &rvec[0], rsize, MPI_INT, rcomm))
      CkAbort("AMPI> Error in MPI_Intercomm_create");

    if(rsize==0){
      AMPI_DEBUG("AMPI> In MPI_Intercomm_create, creating an empty communicator\n");
      *newintercomm = MPI_COMM_NULL;
      return MPI_SUCCESS;
    }
  }

  lptr->intercommCreate(rvec,root,newintercomm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm){
  AMPIAPI("AMPI_Intercomm_merge");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isInter(intercomm))
    return ampiErrhandler("AMPI_Intercomm_merge", MPI_ERR_COMM);
#endif

  ampi *ptr = getAmpiInstance(intercomm);
  int lroot, rroot, lrank, lhigh, rhigh, first;
  lroot = ptr->getIndexForRank(0);
  rroot = ptr->getIndexForRemoteRank(0);
  lhigh = high;
  lrank = ptr->getRank(intercomm);
  first = 0;

  if(lrank==0){
    ptr->send(MPI_ATA_TAG, ptr->getRank(intercomm), &lhigh, 1, MPI_INT, 0, intercomm);
    if(-1==ptr->recv(MPI_ATA_TAG,0,&rhigh,1,MPI_INT,intercomm))
      CkAbort("AMPI> Error in MPI_Intercomm_create");

    if((lhigh && rhigh) || (!lhigh && !rhigh)){ // same value: smaller root goes first (first=1 if local goes first)
      first = (lroot < rroot);
    }else{ // different values, then high=false goes first
      first = (lhigh == false);
    }
  }

  ptr->intercommMerge(first, newintracomm);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Abort(MPI_Comm comm, int errorcode)
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
  if (itemsize == 0) {
    *count = 0;
  } else {
    *count = sts->MPI_LENGTH/itemsize;
  }
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_lb(MPI_Datatype dtype, MPI_Aint* displacement){
  AMPIAPI("AMPI_Type_lb");
  *displacement = getDDT()->getLB(dtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_ub(MPI_Datatype dtype, MPI_Aint* displacement){
  AMPIAPI("AMPI_Type_ub");
  *displacement = getDDT()->getUB(dtype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Get_address(const void* location, MPI_Aint *address){
  AMPIAPI("AMPI_Get_address");
  *address = (MPI_Aint)location;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Address(void* location, MPI_Aint *address){
  AMPIAPI("AMPI_Address");
  return AMPI_Get_address(location, address);
}

CDECL
int AMPI_Status_set_elements(MPI_Status *sts, MPI_Datatype dtype, int count){
  AMPIAPI("AMPI_Status_set_elements");
  if(sts == MPI_STATUS_IGNORE || sts == MPI_STATUSES_IGNORE)
    return MPI_SUCCESS;
  CkDDT_DataType* dttype = getDDT()->getType(dtype);
  int basesize = dttype->getBaseSize();
  if(basesize==0) basesize = dttype->getSize();
  sts->MPI_LENGTH = basesize * count;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Get_elements(MPI_Status *sts, MPI_Datatype dtype, int *count){
  AMPIAPI("AMPI_Get_elements");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  *count = dttype->getNumElements();
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
}

CDECL
int AMPI_Unpack(void *inbuf, int insize, int *position, void *outbuf,
                int outcount, MPI_Datatype dtype, MPI_Comm comm)
{
  AMPIAPI("AMPI_Unpack");
  CkDDT_DataType* dttype = getDDT()->getType(dtype) ;
  int itemsize = dttype->getSize();
  dttype->serialize((char*)outbuf, ((char*)inbuf+(*position)), outcount, -1);
  *position += (itemsize*outcount);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int *sz)
{
  AMPIAPI("AMPI_Pack_size");
  CkDDT_DataType* dttype = getDDT()->getType(datatype) ;
  *sz = incount*dttype->getSize() ;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Get_version(int *version, int *subversion){
  AMPIAPI("AMPI_Get_version");
  *version = MPI_VERSION;
  *subversion = MPI_SUBVERSION;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Get_library_version(char *version, int *resultlen){
  AMPIAPI("AMPI_Get_library_version");
  const char *ampiNameStr = "Adaptive MPI ";
  strncpy(version, ampiNameStr, MPI_MAX_LIBRARY_VERSION_STRING);
  strncat(version, CmiCommitID, MPI_MAX_LIBRARY_VERSION_STRING - strlen(version));
  *resultlen = strlen(version);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Get_processor_name(char *name, int *resultlen){
  AMPIAPI("AMPI_Get_processor_name");
  ampiParent *ptr = getAmpiParent();
  sprintf(name,"AMPI_VP[%d]_PE[%d]",ptr->thisIndex,ptr->getMyPe());
  *resultlen = strlen(name);
  return MPI_SUCCESS;
}

/* Error handling */
#if defined(USE_STDARG)
void error_handler(MPI_Comm *, int *, ...);
#else
void error_handler ( MPI_Comm *, int * );
#endif

CDECL
int AMPI_Comm_call_errhandler(MPI_Comm comm, int errorcode){
  AMPIAPI("AMPI_Comm_call_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_create_errhandler(MPI_Comm_errhandler_fn *function, MPI_Errhandler *errhandler){
  AMPIAPI("AMPI_Comm_create_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler){
  AMPIAPI("AMPI_Comm_set_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler){
  AMPIAPI("AMPI_Comm_get_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_free_errhandler(MPI_Errhandler *errhandler){
  AMPIAPI("AMPI_Comm_free_errhandler");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Errhandler_create(MPI_Handler_function *function, MPI_Errhandler *errhandler){
  AMPIAPI("AMPI_Errhandler_create");
  return AMPI_Comm_create_errhandler(function, errhandler);
}

CDECL
int AMPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler){
  AMPIAPI("AMPI_Errhandler_set");
  return AMPI_Comm_set_errhandler(comm, errhandler);
}

CDECL
int AMPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler){
  AMPIAPI("AMPI_Errhandler_get");
  return AMPI_Comm_get_errhandler(comm, errhandler);
}

CDECL
int AMPI_Errhandler_free(MPI_Errhandler *errhandler){
  AMPIAPI("AMPI_Errhandler_free");
  return AMPI_Comm_free_errhandler(errhandler);
}

CDECL
int AMPI_Add_error_code(int errorclass, int *errorcode){
  AMPIAPI("AMPI_Add_error_code");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Add_error_class(int *errorclass){
  AMPIAPI("AMPI_Add_error_class");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Add_error_string(int errorcode, const char *errorstring){
  AMPIAPI("AMPI_Add_error_string");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Error_class(int errorcode, int *errorclass){
  AMPIAPI("AMPI_Error_class");
  *errorclass = errorcode;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Error_string(int errorcode, char *errorstring, int *resultlen)
{
  AMPIAPI("AMPI_Error_string");
  const char *r="";
  switch(errorcode) {
    case MPI_SUCCESS:
      r="MPI_SUCCESS: no errors"; break;
    case MPI_ERR_BUFFER:
      r="MPI_ERR_BUFFER: invalid buffer pointer"; break;
    case MPI_ERR_COUNT:
      r="MPI_ERR_COUNT: invalid count argument"; break;
    case MPI_ERR_TYPE:
      r="MPI_ERR_TYPE: invalid datatype"; break;
    case MPI_ERR_TAG:
      r="MPI_ERR_TAG: invalid tag"; break;
    case MPI_ERR_COMM:
      r="MPI_ERR_COMM: invalid communicator"; break;
    case MPI_ERR_RANK:
      r="MPI_ERR_RANK: invalid rank"; break;
    case MPI_ERR_REQUEST:
      r="MPI_ERR_REQUEST: invalid request (handle)"; break;
    case MPI_ERR_ROOT:
      r="MPI_ERR_ROOT: invalid root"; break;
    case MPI_ERR_GROUP:
      r="MPI_ERR_GROUP: invalid group"; break;
    case MPI_ERR_OP:
      r="MPI_ERR_OP: invalid operation"; break;
    case MPI_ERR_TOPOLOGY:
      r="MPI_ERR_TOPOLOGY: invalid communicator topology"; break;
    case MPI_ERR_DIMS:
      r="MPI_ERR_DIMS: invalid dimension argument"; break;
    case MPI_ERR_ARG:
      r="MPI_ERR_ARG: invalid argument of some other kind"; break;
    case MPI_ERR_TRUNCATE:
      r="MPI_ERR_TRUNCATE: message truncated in recieve"; break;
    case MPI_ERR_OTHER:
      r="MPI_ERR_OTHER: known error not in this list"; break;
    case MPI_ERR_INTERN:
      r="MPI_ERR_INTERN: internal MPI (implementation) error"; break;
    case MPI_ERR_IN_STATUS:
      r="MPI_ERR_IN_STATUS: error code in status"; break;
    case MPI_ERR_PENDING:
      r="MPI_ERR_PENDING: pending request"; break;
    case MPI_ERR_ACCESS:
      r="MPI_ERR_ACCESS: invalid access mode"; break;
    case MPI_ERR_AMODE:
      r="MPI_ERR_AMODE: invalid amode argument"; break;
    case MPI_ERR_ASSERT:
      r="MPI_ERR_ASSERT: invalid assert argument"; break;
    case MPI_ERR_BAD_FILE:
      r="MPI_ERR_BAD_FILE: bad file"; break;
    case MPI_ERR_BASE:
      r="MPI_ERR_BASE: invalid base"; break;
    case MPI_ERR_CONVERSION:
      r="MPI_ERR_CONVERSION: error in data conversion"; break;
    case MPI_ERR_DISP:
      r="MPI_ERR_DISP: invalid displacement"; break;
    case MPI_ERR_DUP_DATAREP:
      r="MPI_ERR_DUP_DATAREP: error duplicating data representation"; break;
    case MPI_ERR_FILE_EXISTS:
      r="MPI_ERR_FILE_EXISTS: file exists already"; break;
    case MPI_ERR_FILE_IN_USE:
      r="MPI_ERR_FILE_IN_USE: file in use already"; break;
    case MPI_ERR_FILE:
      r="MPI_ERR_FILE: invalid file"; break;
    case MPI_ERR_INFO_KEY:
      r="MPI_ERR_INFO_KEY: invalid key argument for info object"; break;
    case MPI_ERR_INFO_NOKEY:
      r="MPI_ERR_INFO_NOKEY: unknown key for info object"; break;
    case MPI_ERR_INFO_VALUE:
      r="MPI_ERR_INFO_VALUE: invalid value argument for info object"; break;
    case MPI_ERR_INFO:
      r="MPI_ERR_INFO: invalid info object"; break;
    case MPI_ERR_IO:
      r="MPI_ERR_IO: input/output error"; break;
    case MPI_ERR_KEYVAL:
      r="MPI_ERR_KEYVAL: invalid keyval"; break;
    case MPI_ERR_LOCKTYPE:
      r="MPI_ERR_LOCKTYPE: invalid locktype argument"; break;
    case MPI_ERR_NAME:
      r="MPI_ERR_NAME: invalid name argument"; break;
    case MPI_ERR_NO_MEM:
      r="MPI_ERR_NO_MEM: out of memory"; break;
    case MPI_ERR_NOT_SAME:
      r="MPI_ERR_NOT_SAME: objects are not identical"; break;
    case MPI_ERR_NO_SPACE:
      r="MPI_ERR_NO_SPACE: no space left on device"; break;
    case MPI_ERR_NO_SUCH_FILE:
      r="MPI_ERR_NO_SUCH_FILE: no such file or directory"; break;
    case MPI_ERR_PORT:
      r="MPI_ERR_PORT: invalid port"; break;
    case MPI_ERR_QUOTA:
      r="MPI_ERR_QUOTA: out of quota"; break;
    case MPI_ERR_READ_ONLY:
      r="MPI_ERR_READ_ONLY: file is read only"; break;
    case MPI_ERR_RMA_CONFLICT:
      r="MPI_ERR_RMA_CONFLICT: rma conflict during operation"; break;
    case MPI_ERR_RMA_SYNC:
      r="MPI_ERR_RMA_SYNC: error executing rma sync"; break;
    case MPI_ERR_SERVICE:
      r="MPI_ERR_SERVICE: unknown service name"; break;
    case MPI_ERR_SIZE:
      r="MPI_ERR_SIZE: invalid size argument"; break;
    case MPI_ERR_SPAWN:
      r="MPI_ERR_SPAWN: error in spawning processes"; break;
    case MPI_ERR_UNSUPPORTED_DATAREP:
      r="MPI_ERR_UNSUPPORTED_DATAREP: data representation not supported"; break;
    case MPI_ERR_UNSUPPORTED_OPERATION:
      r="MPI_ERR_UNSUPPORTED_OPERATION: operation not supported"; break;
    case MPI_ERR_WIN:
      r="MPI_ERR_WIN: invalid win argument"; break;
    default:
      r="unknown error";
      *resultlen=strlen(r);
      strcpy(errorstring,r);
      return MPI_ERR_UNKNOWN;
  };
  *resultlen=strlen(r);
  strcpy(errorstring,r);
  return MPI_SUCCESS;
}

/* Group operations */
CDECL
int AMPI_Comm_group(MPI_Comm comm, MPI_Group *group)
{
  AMPIAPI("AMPI_Comm_Group");
  *group = getAmpiParent()->comm2group(comm);
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
}

CDECL
int AMPI_Group_size(MPI_Group group, int *size)
{
  AMPIAPI("AMPI_Group_size");
  *size = (getAmpiParent()->group2vec(group)).size();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Group_rank(MPI_Group group, int *rank)
{
  AMPIAPI("AMPI_Group_rank");
  *rank = getAmpiParent()->getRank(group);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Group_translate_ranks (MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2)
{
  AMPIAPI("AMPI_Group_translate_ranks");
  ampiParent *ptr = getAmpiParent();
  groupStruct vec1, vec2;
  vec1 = ptr->group2vec(group1);
  vec2 = ptr->group2vec(group2);
  translateRanksOp(n, vec1, ranks1, vec2, ranks2);
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
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
  return MPI_SUCCESS;
}

CDECL
int AMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_range_incl");
  groupStruct vec, newvec;
  int ret;
  ampiParent *ptr = getAmpiParent();
  vec = ptr->group2vec(group);
  newvec = rangeInclOp(n,ranges,vec,&ret);
  if(ret != MPI_SUCCESS){
    *newgroup = MPI_GROUP_EMPTY;
    return ampiErrhandler("AMPI_Group_range_incl", ret);
  }else{
    *newgroup = ptr->saveGroupStruct(newvec);
    return MPI_SUCCESS;
  }
}

CDECL
int AMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
{
  AMPIAPI("AMPI_Group_range_excl");
  groupStruct vec, newvec;
  int ret;
  ampiParent *ptr = getAmpiParent();
  vec = ptr->group2vec(group);
  newvec = rangeExclOp(n,ranges,vec,&ret);
  if(ret != MPI_SUCCESS){
    *newgroup = MPI_GROUP_EMPTY;
    return ampiErrhandler("AMPI_Group_range_excl", ret);
  }else{
    *newgroup = ptr->saveGroupStruct(newvec);
    return MPI_SUCCESS;
  }
}

CDECL
int AMPI_Group_free(MPI_Group *group)
{
  AMPIAPI("AMPI_Group_free");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm)
{
  AMPIAPI("AMPI_Comm_create");
  int rank_in_group, key, color, zero;
  MPI_Group group_of_comm;

  groupStruct vec = getAmpiParent()->group2vec(group);
  if(vec.size()==0){
    AMPI_DEBUG("AMPI> In MPI_Comm_create, creating an empty communicator");
    *newcomm = MPI_COMM_NULL;
    return MPI_SUCCESS;
  }

  if(getAmpiParent()->isInter(comm)){
    /* inter-communicator: create a single new comm. */
    ampi *ptr = getAmpiInstance(comm);
    ptr->commCreate(vec, newcomm);
    ptr->barrier();
  }
  else{
    /* intra-communicator: create comm's for disjoint subgroups,
     * by calculating (color, key) and splitting comm. */
    AMPI_Group_rank(group, &rank_in_group);
    if(rank_in_group == MPI_UNDEFINED){
      color = MPI_UNDEFINED;
      key = 0;
    }
    else{
      /* use rank in 'comm' of the 0th rank in 'group'
       * as identical 'color' of all ranks in 'group' */
      AMPI_Comm_group(comm, &group_of_comm);
      zero = 0;
      AMPI_Group_translate_ranks(group, 1, &zero, group_of_comm, &color);
      key = rank_in_group;
    }
    return AMPI_Comm_split(comm, color, key, newcomm);
  }
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_set_name(MPI_Comm comm, const char *comm_name){
  AMPIAPI("AMPI_Comm_set_name");
  getAmpiInstance(comm)->setCommName(comm_name);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen){
  AMPIAPI("AMPI_Comm_get_name");
  getAmpiInstance(comm)->getCommName(comm_name, resultlen);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_set_info(MPI_Comm comm, MPI_Info info){
  AMPIAPI("AMPI_Comm_set_info");
  /* FIXME: no-op implementation */
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_get_info(MPI_Comm comm, MPI_Info *info){
  AMPIAPI("AMPI_Comm_get_info");
  /* FIXME: no-op implementation */
  *info = MPI_INFO_NULL;
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_create_keyval(MPI_Comm_copy_attr_function *copy_fn,
                            MPI_Comm_delete_attr_function *delete_fn,
                            int *keyval, void* extra_state){
  AMPIAPI("AMPI_Comm_create_keyval");
  int ret = getAmpiParent()->createKeyval(copy_fn,delete_fn,keyval,extra_state);
  return ampiErrhandler("AMPI_Comm_create_keyval", ret);
}

CDECL
int AMPI_Comm_free_keyval(int *keyval){
  AMPIAPI("AMPI_Comm_free_keyval");
  int ret = getAmpiParent()->freeKeyval(keyval);
  return ampiErrhandler("AMPI_Comm_free_keyval", ret);
}

CDECL
int AMPI_Comm_set_attr(MPI_Comm comm, int keyval, void* attribute_val){
  AMPIAPI("AMPI_Comm_set_attr");
  int ret = getAmpiParent()->setCommAttr(comm,keyval,attribute_val);
  return ampiErrhandler("AMPI_Comm_set_attr", ret);
}

CDECL
int AMPI_Comm_get_attr(MPI_Comm comm, int keyval, void *attribute_val, int *flag){
  AMPIAPI("AMPI_Comm_get_attr");
  int ret = getAmpiParent()->getCommAttr(comm,keyval,attribute_val,flag);
  return ampiErrhandler("AMPI_Comm_get_attr", ret);
}

CDECL
int AMPI_Comm_delete_attr(MPI_Comm comm, int keyval){
  AMPIAPI("AMPI_Comm_delete_attr");
  int ret = getAmpiParent()->deleteCommAttr(comm,keyval);
  return ampiErrhandler("AMPI_Comm_delete_attr", ret);
}

CDECL
int AMPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn,
                       int *keyval, void* extra_state){
  AMPIAPI("AMPI_Keyval_create");
  return AMPI_Comm_create_keyval(copy_fn, delete_fn, keyval, extra_state);
}

CDECL
int AMPI_Keyval_free(int *keyval){
  AMPIAPI("AMPI_Keyval_free");
  return AMPI_Comm_free_keyval(keyval);
}

CDECL
int AMPI_Attr_put(MPI_Comm comm, int keyval, void* attribute_val){
  AMPIAPI("AMPI_Attr_put");
  return AMPI_Comm_set_attr(comm, keyval, attribute_val);
}

CDECL
int AMPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag){
  AMPIAPI("AMPI_Attr_get");
  return AMPI_Comm_get_attr(comm, keyval, attribute_val, flag);
}

CDECL
int AMPI_Attr_delete(MPI_Comm comm, int keyval){
  AMPIAPI("AMPI_Attr_delete");
  return AMPI_Comm_delete_attr(comm, keyval);
}

CDECL
int AMPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank) {
  AMPIAPI("AMPI_Cart_map");
  return AMPI_Comm_rank(comm, newrank);
}

CDECL
int AMPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges, int *newrank) {
  AMPIAPI("AMPI_Graph_map");
  return AMPI_Comm_rank(comm, newrank);
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

  vector<int> dimsv;
  vector<int> periodsv;

  for (int i = 0; i < ndims; i++) {
    dimsv.push_back(dims[i]);
    periodsv.push_back(periods[i]);
  }

  c.setdims(dimsv);
  c.setperiods(periodsv);

  vector<int> nborsv;
  getAmpiInstance(*comm_cart)->findNeighbors(*comm_cart, newrank, nborsv);
  c.setnbors(nborsv);

  return MPI_SUCCESS;
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

  vector<int> index_;
  vector<int> edges_;

  int i;
  for (i = 0; i < nnodes; i++)
    index_.push_back(index[i]);

  c.setindex(index_);

  for (i = 0; i < index[nnodes - 1]; i++)
    edges_.push_back(edges[i]);

  c.setedges(edges_);

  vector<int> nborsv;
  getAmpiInstance(*comm_graph)->findNeighbors(*comm_graph, newrank, nborsv);
  c.setnbors(nborsv);

  return MPI_SUCCESS;
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

  return MPI_SUCCESS;
}

CDECL
int AMPI_Cartdim_get(MPI_Comm comm, int *ndims) {
  AMPIAPI("AMPI_Cartdim_get");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isCart(comm))
    return ampiErrhandler("AMPI_Cartdim_get", MPI_ERR_TOPOLOGY);
#endif

  *ndims = getAmpiParent()->getCart(comm).getndims();

  return MPI_SUCCESS;
}

CDECL
int AMPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords){
  int i, ndims;

  AMPIAPI("AMPI_Cart_get");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isCart(comm))
    return ampiErrhandler("AMPI_Cart_get", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  ndims = c.getndims();
  int rank = getAmpiInstance(comm)->getRank(comm);

  const vector<int> &dims_ = c.getdims();
  const vector<int> &periods_ = c.getperiods();

  for (i = 0; i < maxdims; i++) {
    dims[i] = dims_[i];
    periods[i] = periods_[i];
  }

  for (i = ndims - 1; i >= 0; i--) {
    if (i < maxdims)
      coords[i] = rank % dims_[i];
    rank = (int) (rank / dims_[i]);
  }

  return MPI_SUCCESS;
}

CDECL
int AMPI_Cart_rank(MPI_Comm comm, int *coords, int *rank) {
  AMPIAPI("AMPI_Cart_rank");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isCart(comm))
    return ampiErrhandler("AMPI_Cart_rank", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  int ndims = c.getndims();
  const vector<int> &dims = c.getdims();
  const vector<int> &periods = c.getperiods();

  int prod = 1;
  int r = 0;

  for (int i = ndims - 1; i >= 0; i--) {
    if ((coords[i] < 0) || (coords[i] >= dims[i])) {
      if (periods[i] != 0) {
        if (coords[i] > 0) {
          coords[i] %= dims[i];
        } else {
          while (coords[i] < 0) coords[i]+=dims[i];
        }
      }
    }
    r += prod * coords[i];
    prod *= dims[i];
  }

  *rank = r;

  return MPI_SUCCESS;
}

CDECL
int AMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords) {
  AMPIAPI("AMPI_Cart_coords");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isCart(comm))
    return ampiErrhandler("AMPI_Cart_coorts", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  int ndims = c.getndims();
  const vector<int> &dims = c.getdims();

  for (int i = ndims - 1; i >= 0; i--) {
    if (i < maxdims)
      coords[i] = rank % dims[i];
    rank = (int) (rank / dims[i]);
  }

  return MPI_SUCCESS;
}

// Offset coords[direction] by displacement, and set the rank that
// results
static void cart_clamp_coord(MPI_Comm comm, const vector<int> &dims,
                             const vector<int> &periodicity, int *coords,
                             int direction, int displacement, int *rank_out)
{
  int base_coord = coords[direction];
  coords[direction] += displacement;

  if (periodicity[direction] != 0) {
    while (coords[direction] < 0)
      coords[direction] += dims[direction];
    while (coords[direction] >= dims[direction])
      coords[direction] -= dims[direction];
  }

  if (coords[direction]<0 || coords[direction]>= dims[direction])
    *rank_out = MPI_PROC_NULL;
  else
    AMPI_Cart_rank(comm, coords, rank_out);

  coords[direction] = base_coord;
}

CDECL
int AMPI_Cart_shift(MPI_Comm comm, int direction, int disp,
                    int *rank_source, int *rank_dest) {
  AMPIAPI("AMPI_Cart_shift");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isCart(comm))
    return ampiErrhandler("AMPI_Cart_shift", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  int ndims = c.getndims();

#if AMPI_ERROR_CHECKING
  if ((direction < 0) || (direction >= ndims))
    return ampiErrhandler("AMPI_Cart_shift", MPI_ERR_DIMS);
#endif

  const vector<int> &dims = c.getdims();
  const vector<int> &periods = c.getperiods();
  vector<int> coords(ndims);

  int mype = getAmpiInstance(comm)->getRank(comm);
  AMPI_Cart_coords(comm, mype, ndims, &coords[0]);

  cart_clamp_coord(comm, dims, periods, &coords[0], direction,  disp, rank_dest);
  cart_clamp_coord(comm, dims, periods, &coords[0], direction, -disp, rank_source);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges) {
  AMPIAPI("AMPI_Graphdim_get");

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);
  *nnodes = c.getnvertices();
  const vector<int> &index = c.getindex();
  *nedges = index[(*nnodes) - 1];

  return MPI_SUCCESS;
}

CDECL
int AMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index, int *edges) {
  AMPIAPI("AMPI_Graph_get");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isGraph(comm))
    return ampiErrhandler("AMPI_Graph_get", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);
  const vector<int> &index_ = c.getindex();
  const vector<int> &edges_ = c.getedges();

  if (maxindex > index_.size())
    maxindex = index_.size();

  int i;
  for (i = 0; i < maxindex; i++)
    index[i] = index_[i];

  for (i = 0; i < maxedges; i++)
    edges[i] = edges_[i];

  return MPI_SUCCESS;
}

CDECL
int AMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors) {
  AMPIAPI("AMPI_Graph_neighbors_count");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isGraph(comm))
    return ampiErrhandler("AMPI_Graph_neighbors_count", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);
  const vector<int> &index = c.getindex();

#if AMPI_ERROR_CHECKING
  if ((rank >= index.size()) || (rank < 0))
    return ampiErrhandler("AMPI_Graph_neighbors_count", MPI_ERR_RANK);
#endif

  if (rank == 0)
    *nneighbors = index[rank];
  else
    *nneighbors = index[rank] - index[rank - 1];

  return MPI_SUCCESS;
}

CDECL
int AMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int *neighbors) {
  AMPIAPI("AMPI_Graph_neighbors");

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isGraph(comm))
    return ampiErrhandler("AMPI_Graph_neighbors", MPI_ERR_TOPOLOGY);
#endif

  ampiCommStruct &c = getAmpiParent()->getGraph(comm);
  const vector<int> &index = c.getindex();
  const vector<int> &edges = c.getedges();

  int numneighbors = (rank == 0) ? index[rank] : index[rank] - index[rank - 1];
  if (maxneighbors > numneighbors)
    maxneighbors = numneighbors;

#if AMPI_ERROR_CHECKING
  if (maxneighbors < 0)
    return ampiErrhandler("AMPI_Graph_neighbors", MPI_ERR_ARG);
  if ((rank >= index.size()) || (rank < 0))
    return ampiErrhandler("AMPI_Graph_neighbors", MPI_ERR_RANK);
#endif

  if (rank == 0) {
    for (int i = 0; i < maxneighbors; i++)
      neighbors[i] = edges[i];
  } else {
    for (int i = 0; i < maxneighbors; i++)
      neighbors[i] = edges[index[rank - 1] + i];
  }
  return MPI_SUCCESS;
}

/* Used by MPI_Cart_create & MPI_Graph_create */
void ampi::findNeighbors(MPI_Comm comm, int rank, vector<int>& neighbors) const {
  int max_neighbors = 0;
  ampiParent *ptr = getAmpiParent();
  if (ptr->isGraph(comm)) {
    AMPI_Graph_neighbors_count(comm, rank, &max_neighbors);
    neighbors.resize(max_neighbors);
    AMPI_Graph_neighbors(comm, rank, max_neighbors, &neighbors[0]);
  }
  else if (ptr->isCart(comm)) {
    int num_dims;
    AMPI_Cartdim_get(comm, &num_dims);
    max_neighbors = 2*num_dims;
    for (int i=0; i<max_neighbors; i++) {
      int src, dest;
      AMPI_Cart_shift(comm, i/2, (i%2==0)?1:-1, &src, &dest);
      if (dest != MPI_PROC_NULL)
        neighbors.push_back(dest);
    }
  }
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
    for (int k=k_up;k>=m;k--) {
      if (n%k==0) { /* k divides n-- try it as a factor */
        dims[0]=k;
        if (factors(n/k,d-1,&dims[1],k))
          return true;
      }
    }
  }
  /* If we fall out here, there were no factors available */
  return false;
}

CDECL
int AMPI_Dims_create(int nnodes, int ndims, int *dims) {
  AMPIAPI("AMPI_Dims_create");

  int i, n, d;

  n = nnodes;
  d = ndims;

  for (i = 0; i < ndims; i++) {
    if (dims[i] != 0) {
      if (n % dims[i] != 0) {
        return ampiErrhandler("AMPI_Dims_create", MPI_ERR_DIMS);
      } else {
        n = n / dims[i];
        d--;
      }
    }
  }

  if(d > 0) {
    vector<int> pdims(d);

    if (!factors(n, d, &pdims[0], 1))
      CkAbort("MPI_Dims_create: factorization failed!\n");

    int j = 0;
    for (i = 0; i < ndims; i++) {
      if (dims[i] == 0) {
        dims[i] = pdims[j];
        j++;
      }
    }
  }

  return MPI_SUCCESS;
}

/* Implemented with call to MPI_Comm_Split. Color and key are single integer
   encodings of the lost and preserved dimensions, respectively,
   of the subgraphs.
 */
CDECL
int AMPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *newcomm) {
  AMPIAPI("AMPI_Cart_sub");

  int i, ndims;
  int color = 1, key = 1;

#if AMPI_ERROR_CHECKING
  if (!getAmpiParent()->isCart(comm))
    return ampiErrhandler("AMPI_Cart_sub", MPI_ERR_TOPOLOGY);
#endif

  int rank = getAmpiInstance(comm)->getRank(comm);
  ampiCommStruct &c = getAmpiParent()->getCart(comm);
  ndims = c.getndims();
  const vector<int> &dims = c.getdims();
  int num_remain_dims = 0;

  vector<int> coords(ndims);
  AMPI_Cart_coords(comm, rank, ndims, &coords[0]);

  for (i = 0; i < ndims; i++) {
    if (remain_dims[i]) {
      /* key single integer encoding*/
      key = key * dims[i] + coords[i];
      num_remain_dims++;
    }
    else {
      /* color */
      color = color * dims[i] + coords[i];
    }
  }

  getAmpiInstance(comm)->split(color, key, newcomm, MPI_CART);

  ampiCommStruct &newc = getAmpiParent()->getCart(*newcomm);
  newc.setndims(num_remain_dims);
  vector<int> dimsv;
  const vector<int> &periods = c.getperiods();
  vector<int> periodsv;

  for (i = 0; i < ndims; i++) {
    if (remain_dims[i]) {
      dimsv.push_back(dims[i]);
      periodsv.push_back(periods[i]);
    }
  }
  newc.setdims(dimsv);
  newc.setperiods(periodsv);

  vector<int> nborsv;
  getAmpiInstance(*newcomm)->findNeighbors(*newcomm, getAmpiParent()->getRank(*newcomm), nborsv);
  newc.setnbors(nborsv);

  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_get_envelope(MPI_Datatype datatype, int *ni, int *na, int *nd, int *combiner){
  AMPIAPI("AMPI_Type_get_envelope");
  return getDDT()->getEnvelope(datatype,ni,na,nd,combiner);
}

CDECL
int AMPI_Type_get_contents(MPI_Datatype datatype, int ni, int na, int nd, int i[],
                           MPI_Aint a[], MPI_Datatype d[]){
  AMPIAPI("AMPI_Type_get_contents");
  return getDDT()->getContents(datatype,ni,na,nd,i,a,d);
}

CDECL
int AMPI_Pcontrol(const int level, ...) {
  //AMPIAPI("AMPI_Pcontrol");
  return MPI_SUCCESS;
}

/******** AMPI Extensions to the MPI standard *********/

CDECL
int AMPI_Migrate(MPI_Info hints)
{
  AMPIAPI("AMPI_Migrate");
  int nkeys, exists;
  char key[MPI_MAX_INFO_KEY], value[MPI_MAX_INFO_VAL];

  AMPI_Info_get_nkeys(hints, &nkeys);

  for (int i=0; i<nkeys; i++) {
    AMPI_Info_get_nthkey(hints, i, key);
    AMPI_Info_get(hints, key, MPI_MAX_INFO_VAL, value, &exists);
    if (!exists) {
      continue;
    }
    else if (strncmp(key, "ampi_load_balance", MPI_MAX_INFO_KEY) == 0) {

      if (strncmp(value, "sync", MPI_MAX_INFO_VAL) == 0) {
        TCHARM_Migrate();
      }
      else if (strncmp(value, "async", MPI_MAX_INFO_VAL) == 0) {
        TCHARM_Async_Migrate();
      }
      else if (strncmp(value, "false", MPI_MAX_INFO_VAL) == 0) {
        /* do nothing */
      }
      else {
        CkPrintf("WARNING: Unknown MPI_Info value (%s) given to AMPI_Migrate for key: %s\n", value, key);
      }
    }
    else if (strncmp(key, "ampi_checkpoint", MPI_MAX_INFO_KEY) == 0) {

      if (strncmp(value, "true", MPI_MAX_INFO_VAL) == 0) {
        CkAbort("AMPI> Error: Value \"true\" is not supported for AMPI_Migrate key \"ampi_checkpoint\"!\n");
      }
      else if (strncmp(value, "to_file=", strlen("to_file=")) == 0) {
        int offset = strlen("to_file=");
        int restart_dir_name_len = 0;
        AMPI_Info_get_valuelen(hints, key, &restart_dir_name_len, &exists);
        if (restart_dir_name_len > offset) {
          value[restart_dir_name_len] = '\0';
        }
        else {
          CkAbort("AMPI> Error: No checkpoint directory name given to AMPI_Migrate\n");
        }
        getAmpiInstance(MPI_COMM_WORLD)->barrier();
        getAmpiParent()->startCheckpoint(&value[offset]);
      }
      else if (strncmp(value, "in_memory", MPI_MAX_INFO_VAL) == 0) {
#if CMK_MEM_CHECKPOINT
        getAmpiInstance(MPI_COMM_WORLD)->barrier();
        getAmpiParent()->startCheckpoint("");
#else
        CkPrintf("AMPI> Error: In-memory checkpoint/restart is not enabled!\n");
        CkAbort("AMPI> Error: Recompile Charm++/AMPI with CMK_MEM_CHECKPOINT.\n");
#endif
      }
      else if (strncmp(value, "message_logging", MPI_MAX_INFO_VAL) == 0) {
#if CMK_MESSAGE_LOGGING
        TCHARM_Migrate();
#else
        CkPrintf("AMPI> Error: Message logging is not enabled!\n");
        CkAbort("AMPI> Error: Recompile Charm++/AMPI with CMK_MESSAGE_LOGGING.\n");
#endif
      }
      else if (strncmp(value, "false", MPI_MAX_INFO_VAL) == 0) {
        /* do nothing */
      }
      else {
        CkPrintf("WARNING: Unknown MPI_Info value (%s) given to AMPI_Migrate for key: %s\n", value, key);
      }
    }
    else {
      CkPrintf("WARNING: Unknown MPI_Info key given to AMPI_Migrate: %s\n", key);
    }
  }

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  ampi *currentAmpi = getAmpiInstance(MPI_COMM_WORLD);
  CpvAccess(_currentObj) = currentAmpi;
#endif

#if CMK_BIGSIM_CHARM
  TRACE_BG_ADD_TAG("AMPI_MIGRATE");
#endif
  return MPI_SUCCESS;
}

CDECL
int AMPI_Evacuate(void)
{
  //AMPIAPI("AMPI_Evacuate");
  TCHARM_Evacuate();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Migrate_to_pe(int dest)
{
  AMPIAPI("AMPI_Migrate_to_pe");
  TCHARM_Migrate_to(dest);
#if CMK_BIGSIM_CHARM
  TRACE_BG_ADD_TAG("AMPI_MIGRATE_TO_PE");
#endif
  return MPI_SUCCESS;
}

CDECL
int AMPI_Comm_set_migratable(MPI_Comm comm, int mig){
  AMPIAPI("AMPI_Comm_set_migratable");
#if CMK_LBDB_ON
  ampi *ptr=getAmpiInstance(comm);
  ptr->setMigratable(mig);
#else
  CkPrintf("WARNING: MPI_Comm_set_migratable is not supported in this build of Charm++/AMPI.\n");
#endif
  return MPI_SUCCESS;
}

CDECL
int AMPI_Load_start_measure(void)
{
  AMPIAPI("AMPI_Load_start_measure");
  ampiParent *ptr = getAmpiParent();
  ptr->start_measure();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Load_stop_measure(void)
{
  AMPIAPI("AMPI_Load_stop_measure");
  ampiParent *ptr = getAmpiParent();
  ptr->stop_measure();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Load_set_value(double value)
{
  AMPIAPI("AMPI_Load_set_value");
  ampiParent *ptr = getAmpiParent();
  ptr->setObjTime(value);
  return MPI_SUCCESS;
}

void _registerampif(void) {
  _registerampi();
}

CDECL
int AMPI_Register_main(MPI_MainFn mainFn,const char *name)
{
  AMPIAPI("AMPI_Register_main");
  if (TCHARM_Element()==0)
  { // I'm responsible for building the TCHARM threads:
    ampiCreateMain(mainFn,name,strlen(name));
  }
  return MPI_SUCCESS;
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
int AMPI_Register_pup(MPI_PupFn fn, void *data, int *idx)
{
  AMPIAPI("AMPI_Register_pup");
  *idx = TCHARM_Register(data, fn);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Register_about_to_migrate(MPI_MigrateFn fn)
{
  AMPIAPI("AMPI_Register_about_to_migrate");
  ampiParent *thisParent = getAmpiParent();
  thisParent->setUserAboutToMigrateFn(fn);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Register_just_migrated(MPI_MigrateFn fn)
{
  AMPIAPI("AMPI_Register_just_migrated");
  ampiParent *thisParent = getAmpiParent();
  thisParent->setUserJustMigratedFn(fn);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Get_pup_data(int idx, void *data)
{
  AMPIAPI("AMPI_Get_pup_data");
  data = TCHARM_Get_userdata(idx);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Type_is_contiguous(MPI_Datatype datatype, int *flag)
{
  AMPIAPI("AMPI_Type_is_contiguous");
  *flag = getDDT()->isContig(datatype);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Print(const char *str)
{
  AMPIAPI("AMPI_Print");
  ampiParent *ptr = getAmpiParent();
  CkPrintf("[%d] %s\n", ptr->thisIndex, str);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Suspend(MPI_Comm comm)
{
  AMPIAPI("AMPI_Suspend");
  getAmpiInstance(comm)->block();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Yield(MPI_Comm comm)
{
  AMPIAPI("AMPI_Yield");
  getAmpiInstance(comm)->yield();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Resume(int dest, MPI_Comm comm)
{
  AMPIAPI("AMPI_Resume");
  getAmpiInstance(comm)->getProxy()[dest].unblock();
  return MPI_SUCCESS;
}

CDECL
int AMPI_System(const char *cmd)
{
  return TCHARM_System(cmd);
}

CDECL
int AMPI_Trace_begin(void)
{
  traceBegin();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Trace_end(void)
{
  traceEnd();
  return MPI_SUCCESS;
}

int AMPI_Install_idle_timer(void)
{
#if AMPI_PRINT_IDLE
  beginHandle = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)BeginIdle,NULL);
  endHandle = CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE,(CcdVoidFn)EndIdle,NULL);
#endif
  return MPI_SUCCESS;
}

int AMPI_Uninstall_idle_timer(void)
{
#if AMPI_PRINT_IDLE
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,beginHandle);
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,endHandle);
#endif
  return MPI_SUCCESS;
}

#if CMK_BIGSIM_CHARM
extern "C" void startCFnCall(void *param,void *msg)
{
  BgSetStartEvent();
  ampi *ptr = (ampi*)param;
  ampi::bcastraw(NULL, 0, ptr->getProxy());
  delete (CkReductionMsg*)msg;
}

CDECL
int AMPI_Set_start_event(MPI_Comm comm)
{
  AMPIAPI("AMPI_Set_start_event");
  CkAssert(comm == MPI_COMM_WORLD);

  ampi *ptr = getAmpiInstance(comm);

  CkDDT_DataType *ddt_type = ptr->getDDT()->getType(MPI_INT);

  CkReductionMsg *msg=makeRednMsg(ddt_type, NULL, 0, MPI_INT, ptr->getRank(comm), MPI_SUM);
  if (CkMyPe() == 0) {
    CkCallback allreduceCB(startCFnCall, ptr);
    msg->setCallback(allreduceCB);
  }
  ptr->contribute(msg);

  /*HACK: Use recv() to block until the reduction data comes back*/
  if(-1==ptr->recv(MPI_BCAST_TAG, -1, NULL, 0, MPI_INT, MPI_COMM_WORLD))
    CkAbort("AMPI> MPI_Allreduce called with different values on different processors!");

  return MPI_SUCCESS;
}

CDECL
int AMPI_Set_end_event(void)
{
  AMPIAPI("AMPI_Set_end_event");
  return MPI_SUCCESS;
}
#endif // CMK_BIGSIM_CHARM

#if CMK_CUDA
GPUReq::GPUReq()
{
  comm = MPI_COMM_SELF;
  isvalid = true;
  AMPI_Comm_rank(comm, &src);
  buf = getAmpiInstance(comm);
}

bool GPUReq::test(MPI_Status *sts)
{
  return statusIreq;
}

bool GPUReq::itest(MPI_Status *sts)
{
  return test(sts);
}

void GPUReq::complete(MPI_Status *sts)
{
  wait(sts);
}

int GPUReq::wait(MPI_Status *sts)
{
  (void)sts;
  while (!statusIreq) {
    getAmpiInstance(comm)->block();
  }
  return 0;
}

void GPUReq::receive(ampi *ptr, AmpiMsg *msg)
{
  CkAbort("GPUReq::receive should never be called");
}

void GPUReq::setComplete()
{
  statusIreq = true;
}

class workRequestQueue;
extern workRequestQueue *wrQueue;
void enqueue(workRequestQueue *q, workRequest *wr);
extern "C++" void setWRCallback(workRequest *wr, void *cb);

void AMPI_GPU_complete(void *request, void* dummy)
{
  GPUReq *req = static_cast<GPUReq *>(request);
  req->setComplete();
  ampi *ptr = static_cast<ampi *>(req->buf);
  ptr->unblock();
}

CDECL
int AMPI_GPU_Iinvoke(workRequest *to_call, MPI_Request *request)
{
  AMPIAPI("AMPI_GPU_Iinvoke");

  AmpiRequestList* reqs = getReqs();
  GPUReq *newreq = new GPUReq();
  *request = reqs->insert(newreq);

  // A callback that completes the corresponding request
  CkCallback *cb = new CkCallback(&AMPI_GPU_complete, newreq);
  setWRCallback(to_call, cb);

  enqueue(wrQueue, to_call);
}

CDECL
int AMPI_GPU_Invoke(workRequest *to_call)
{
  AMPIAPI("AMPI_GPU_Invoke");

  MPI_Request req;
  AMPI_GPU_Iinvoke(to_call, &req);
  AMPI_Wait(&req, MPI_STATUS_IGNORE);

  return MPI_SUCCESS;
}
#endif // CMK_CUDA

#include "ampi.def.h"

