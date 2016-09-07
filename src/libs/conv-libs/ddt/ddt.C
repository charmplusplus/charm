#include "ddt.h"
#include <algorithm>
#include <limits>

using std::numeric_limits;

//Uncomment for debug print statements
#define DDTDEBUG(...) //CmiPrintf(__VA_ARGS__)


/* Serialize a contiguous chunk of memory */
static inline void serializeContig(char* userdata, char* buffer, size_t size, int dir)
{
  if (dir==1) {
    memcpy(buffer, userdata, size);
  } else if (dir==-1) {
    memcpy(userdata, buffer, size);
  } else {
    CmiAbort("CkDDT: Invalid dir given to serialize a contiguous type!\n");
  }
}

CkDDT_DataType*
CkDDT::getType(int nIndex) const
{
  if( (nIndex >= 0) && (nIndex < max_types))
    return typeTable[nIndex] ;
  else
    return 0 ;
}

void
CkDDT::pup(PUP::er &p)
{
  p(max_types);
  p(num_types);
  if(p.isUnpacking())
  {
    typeTable = new CkDDT_DataType*[max_types];
    types = new int[max_types];
  }
  p(types,max_types);
  int i;
  //unPacking
  if(p.isUnpacking())
  {
    for(i=0 ; i < max_types; i++)
    {
      switch(types[i])
      {
        case CkDDT_TYPE_NULL:
          break ;
        case CkDDT_CONTIGUOUS:
          typeTable[i] = new CkDDT_Contiguous ;
          break ;
        case CkDDT_VECTOR:
          typeTable[i] = new CkDDT_Vector ;
          break ;
        case CkDDT_HVECTOR:
          typeTable[i] = new CkDDT_HVector ;
          break ;
        case CkDDT_INDEXED:
          typeTable[i] = new CkDDT_Indexed ;
          break ;
        case CkDDT_HINDEXED:
          typeTable[i] = new CkDDT_HIndexed ;
          break ;
        case CkDDT_STRUCT:
          typeTable[i] = new CkDDT_Struct ;
          break ;
        default: //CkDDT_PRIMITIVE
	  typeTable[i] = new CkDDT_DataType;
          break ;
      }
    } //End of for loop
  } //end if p.Unpacking()

  for(i=0; i < max_types ; i++)
  {
    if(types[i] != CkDDT_TYPE_NULL)
    {
      typeTable[i]->pupType(p, this);
    }
  }
}

int
CkDDT::getNextFreeIndex(void)
{
  int  i;

  if(num_types < max_types)
    return num_types++;
  for(i=0; i<num_types; i++)
    if(typeTable[i] == 0)
      return i ;
  int newmax = max_types*2;
  CkDDT_DataType** newtable = new CkDDT_DataType*[newmax];
  int *newtype = new int[newmax];
  for(i=0;i<max_types;i++)
  {
    newtable[i] = typeTable[i];
    newtype[i] = types[i];
  }
  for(i=max_types;i<newmax;i++)
  {
    newtable[i] = 0;
    newtype[i] = CkDDT_TYPE_NULL;
  }
  delete[] typeTable;
  delete[] types;
  typeTable = newtable;
  types = newtype;
  num_types = max_types;
  max_types = newmax;
  return num_types++;
}

void
CkDDT::freeType(int* index)
{
  // FIXME: Use reference counting
/*  delete typeTable[*index];
  typeTable[*index] = 0 ;
  types[*index] = CkDDT_TYPE_NULL ;
  *index = -1 ;
*/
}

CkDDT::~CkDDT()
{
  for(int i=0; i < max_types ; i++)
  {
    if(types[i] != CkDDT_TYPE_NULL)
    {
       delete typeTable[i];
    }
  }
  delete[] typeTable ;
  delete[] types;
}


bool
CkDDT::isContig(int nIndex) const
{
  return getType(nIndex)->isContig();
}

int
CkDDT::getSize(int nIndex, int count) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return count*dttype->getSize();
}

CkDDT_Aint
CkDDT::getExtent(int nIndex) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getExtent();
}

CkDDT_Aint
CkDDT::getLB(int nIndex) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getLB();
}

CkDDT_Aint
CkDDT::getUB(int nIndex) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getUB();
}

int CkDDT::getEnvelope(int nIndex, int *ni, int *na, int *nd, int *combiner) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getEnvelope(ni, na, nd, combiner);
}
int CkDDT::getContents(int nIndex, int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getContents(ni, na, nd, i, a, d);
}

void
CkDDT::setName(int nIndex, const char *name)
{
  CkDDT_DataType* dttype = getType(nIndex);
  dttype->setName(name);
}

void
CkDDT::getName(int nIndex, char *name, int *len) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  dttype->getName(name, len);
}

void
CkDDT::createResized(CkDDT_Type oldtype, CkDDT_Aint lb, CkDDT_Aint extent, CkDDT_Type *newtype)
{
  CkDDT_DataType *dttype = getType(oldtype);
  CkDDT_DataType *type = new CkDDT_DataType(*dttype,lb,extent);
  int index = *newtype = getNextFreeIndex();
  typeTable[index] = type;
  types[index] = types[oldtype];
}

void
CkDDT::newContiguous(int count, CkDDT_Type oldType, CkDDT_Type *newType)
{
  int index = *newType = getNextFreeIndex() ;
  CkDDT_DataType *type = new CkDDT_Contiguous(count, oldType, typeTable[oldType]);
  typeTable[index] = type ;
  types[index] = CkDDT_CONTIGUOUS ;
}

void
CkDDT::newVector(int count, int blocklength, int stride,
                 CkDDT_Type oldType, CkDDT_Type* newType)
{
  int index = *newType = getNextFreeIndex() ;
  CkDDT_DataType* type = new CkDDT_Vector(count, blocklength, stride, oldType, typeTable[oldType]);
  typeTable[index] = type;
  types[index] = CkDDT_VECTOR ;           
}

void
CkDDT::newHVector(int count, int blocklength, int stride,
                  CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =
    new CkDDT_HVector(count, blocklength, stride, oldtype, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_HVECTOR ;
}

void
CkDDT::newIndexed(int count, int* arrbLength, CkDDT_Aint* arrDisp,
                  CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =
    new CkDDT_Indexed(count, arrbLength, arrDisp, oldtype, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_INDEXED ;
}

void
CkDDT::newHIndexed(int count, int* arrbLength, CkDDT_Aint* arrDisp,
                   CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =
    new CkDDT_HIndexed(count, arrbLength, arrDisp, oldtype, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_HINDEXED ;
}

void
CkDDT::newIndexedBlock(int count, int Blocklength, CkDDT_Aint *arrDisp, CkDDT_Type oldtype,
                  CkDDT_Type *newtype)
{
  int index = *newtype = getNextFreeIndex();
  CkDDT_DataType *type = new CkDDT_Indexed_Block(count, Blocklength, arrDisp, oldtype, typeTable[oldtype]);
  typeTable[index] = type;
  types[index] = CkDDT_INDEXED_BLOCK;
}

void
CkDDT::newHIndexedBlock(int count, int Blocklength, CkDDT_Aint *arrDisp, CkDDT_Type oldtype,
                  CkDDT_Type *newtype)
{
  int index = *newtype = getNextFreeIndex();
  CkDDT_DataType *type = new CkDDT_HIndexed_Block(count, Blocklength, arrDisp, oldtype, typeTable[oldtype]);
  typeTable[index] = type;
  types[index] = CkDDT_HINDEXED_BLOCK;
}

void
CkDDT::newStruct(int count, int* arrbLength, CkDDT_Aint* arrDisp,
                 CkDDT_Type *oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType **olddatatypes = new CkDDT_DataType*[count];
  for(int i=0;i<count;i++){
    olddatatypes[i] = typeTable[oldtype[i]];
  }
  CkDDT_DataType* type =
    new CkDDT_Struct(count, arrbLength, arrDisp, oldtype, olddatatypes);
  typeTable[index] = type ;
  types[index] = CkDDT_STRUCT ;
}

typedef struct { float val; int idx; } FloatInt;
typedef struct { double val; int idx; } DoubleInt;
typedef struct { long val; int idx; } LongInt;
typedef struct { int val; int idx; } IntInt;
typedef struct { short val; int idx; } ShortInt;
typedef struct { long double val; int idx; } LongdoubleInt;
typedef struct { float val; float idx; } FloatFloat;
typedef struct { double val; double idx; } DoubleDouble;

CkDDT_DataType::CkDDT_DataType(int type):datatype(type)
{
  count = 1;
  switch(datatype) {
    case CkDDT_DOUBLE:
      size = sizeof(double);
      break;
    case CkDDT_INT:
      size = sizeof(signed int);
      break;
    case CkDDT_FLOAT:
      size = sizeof(float);
      break;
    case CkDDT_CHAR:
      size = sizeof(char);
      break;
    case CkDDT_BYTE:
      size = 1 ;
      break;
    case CkDDT_PACKED:
      size = 1 ;
      break;
    case CkDDT_COMPLEX:
    case CkDDT_FLOAT_COMPLEX:
      size =  2 * sizeof(float) ;
      break;
    case CkDDT_DOUBLE_COMPLEX:
      size =  2 * sizeof(double) ;
      break;
    case CkDDT_LONG_DOUBLE_COMPLEX:
      size =  2 * sizeof(long double) ;
      break;
    case CkDDT_C_BOOL:
      /* Should be C99 _Bool instead of C++ bool, but MSVC doesn't support that */
      size = sizeof(bool) ;
      break;
    case CkDDT_LOGICAL:
      size =  sizeof(int) ;
      break;
    case CkDDT_SHORT:
      size = sizeof(signed short int);
      break ;
    case CkDDT_LONG:
      size = sizeof(signed long);
      break ;
    case CkDDT_UNSIGNED_CHAR:
      size = sizeof(unsigned char);
      break;
    case CkDDT_UNSIGNED_SHORT:
      size = sizeof(unsigned short);
      break;
    case CkDDT_UNSIGNED:
      size = sizeof(unsigned);
      break ;
    case CkDDT_UNSIGNED_LONG:
      size = sizeof(unsigned long);
      break ;
    case CkDDT_LONG_DOUBLE:
      size = sizeof(long double);
      break ;
    case CkDDT_FLOAT_INT:
      size = sizeof(FloatInt);
      break;
    case CkDDT_DOUBLE_INT:
      size = sizeof(DoubleInt);
      break;
    case CkDDT_LONG_INT:
      size = sizeof(LongInt);
      break;
    case CkDDT_2INT:
      size = sizeof(IntInt);
      break;
    case CkDDT_SHORT_INT:
      size = sizeof(ShortInt);
      break;
    case CkDDT_LONG_DOUBLE_INT:
      size = sizeof(LongdoubleInt);
      break;
    case CkDDT_2FLOAT:
      size = sizeof(FloatFloat);
      break;
    case CkDDT_2DOUBLE:
      size = sizeof(DoubleDouble);
      break;
    case CkDDT_SIGNED_CHAR:
      size = sizeof(signed char);
      break;
    case CkDDT_UNSIGNED_LONG_LONG:
      size = sizeof(unsigned long long);
      break;
    case CkDDT_WCHAR:
      size = sizeof(wchar_t);
      break;
    case CkDDT_INT8_T:
      size = sizeof(int8_t);
      break;
    case CkDDT_INT16_T:
      size = sizeof(int16_t);
      break;
    case CkDDT_INT32_T:
      size = sizeof(int32_t);
      break;
    case CkDDT_INT64_T:
      size = sizeof(int64_t);
      break;
    case CkDDT_UINT8_T:
      size = sizeof(uint8_t);
      break;
    case CkDDT_UINT16_T:
      size = sizeof(uint16_t);
      break;
    case CkDDT_UINT32_T:
      size = sizeof(uint32_t);
      break;
    case CkDDT_UINT64_T:
      size = sizeof(uint64_t);
      break;
    case CkDDT_AINT:
      size = sizeof(CkDDT_Aint);
      break;
    case CkDDT_LB:
    case CkDDT_UB:
      size = 0;
      break;
#if CMK_LONG_LONG_DEFINED
    case CkDDT_LONG_LONG_INT:
      size = sizeof(signed long long);
      break;
#endif
    default:
      size = 0;
  }

  extent      = size;
  lb          = 0;
  ub          = size;
  iscontig    = true;
  nameLen     = 0;
  baseType    = NULL;
  baseSize    = 0;
  baseExtent  = 0;
  baseIndex   = -1;
  numElements = 1;

  DDTDEBUG("CkDDT_DataType constructor: type=%d, size=%d, extent=%ld, iscontig=%d\n",
           type, size, extent, iscontig);
}


CkDDT_DataType::CkDDT_DataType(int datatype, int size, CkDDT_Aint extent, int count, CkDDT_Aint lb, CkDDT_Aint ub,
            bool iscontig, int baseSize, CkDDT_Aint baseExtent, CkDDT_DataType* baseType, int numElements, int baseIndex) :
    datatype(datatype), size(size), extent(extent), count(count), lb(lb), ub(ub), iscontig(iscontig),
    baseSize(baseSize), baseExtent(baseExtent), baseType(baseType), numElements(numElements),
    baseIndex(baseIndex), nameLen(0), isAbsolute(false)
{}

CkDDT_DataType::CkDDT_DataType(const CkDDT_DataType &obj, CkDDT_Aint _lb, CkDDT_Aint _extent)
{
  datatype    = obj.datatype;
  refCount    = obj.refCount;
  baseSize    = obj.baseSize;
  baseExtent  = obj.baseExtent;
  baseType    = obj.baseType;
  baseIndex   = obj.baseIndex;
  numElements = obj.numElements;
  size        = obj.size;
  count       = obj.count;
  isAbsolute  = obj.isAbsolute;
  nameLen     = obj.nameLen;
  memcpy(name, obj.name, nameLen+1);

  extent = _extent;
  lb     = _lb;
  ub     = lb + extent;

  if (extent != size) {
    iscontig = false;
  }
  else {
    if (baseType != NULL) {
      iscontig = baseType->isContig();
    }
    else {
      iscontig = true;
    }
  }
}

size_t
CkDDT_DataType::serialize(char* userdata, char* buffer, int num, int dir) const
{
  size_t bufSize = (size_t)num * (size_t)size;
  DDTDEBUG("CkDDT_Datatype::serialize %s %d objects of type %d (%d bytes)\n",
           (dir==1)?"packing":"unpacking", num, datatype, bufSize);
  if (iscontig) {
    serializeContig(userdata, buffer, bufSize, dir);
  }
  else {
    for (int i=0; i<num; i++) {
      serializeContig(userdata + i*extent, buffer + i*size, size, dir);
    }
  }
  return bufSize;
}

//Set name for a datatype, stripped of trailing whitespace
void
CkDDT_DataType::setName(const char *src)
{
  CkDDT_SetName(name, src, &nameLen);
}

void
CkDDT_DataType::getName(char *dest, int *len) const
{
  *len = nameLen;
  memcpy(dest, name, *len+1);
}

bool
CkDDT_DataType::isContig() const
{
  return iscontig;
}

void
CkDDT_DataType::setAbsolute(bool arg)
{
  isAbsolute = arg;
}

int
CkDDT_DataType::getSize(int num) const
{
  return num*size ;
}

CkDDT_Aint
CkDDT_DataType::getExtent(void) const
{
  return extent ;
}

int
CkDDT_DataType::getBaseSize(void) const
{
  return baseSize ;
}

int
CkDDT_DataType::getNumElements(void) const
{
  return numElements;
}

CkDDT_Aint
CkDDT_DataType::getLB(void) const
{
  return lb;
}

CkDDT_Aint
CkDDT_DataType::getUB(void) const
{
  return ub;
}

int
CkDDT_DataType::getType(void) const
{
  return datatype;
}

void
CkDDT_DataType::inrRefCount(void)
{
  refCount++ ;
}

int
CkDDT_DataType::getRefCount(void) const
{
  return refCount ;
}

void
CkDDT_DataType::pupType(PUP::er  &p, CkDDT* ddt)
{
  p(datatype);
  p(refCount);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  p(lb);
  p(ub);
  p(iscontig);
  p(isAbsolute);
  p(numElements);
  p(nameLen);
  p(name,CkDDT_MAX_NAME_LEN);
}

int
CkDDT_DataType::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 0;
  *na = 0;
  *nd = 0;
  *combiner = CkDDT_COMBINER_NAMED;
  return 0;
}

int
CkDDT_DataType::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  CmiPrintf("CkDDT_DataType::getContents: Shouldn't call getContents on primitive datatypes!\n");
  return -1;
}

CkDDT_Contiguous::CkDDT_Contiguous(int nCount, int bindex, CkDDT_DataType* oldType)
{
  datatype = CkDDT_CONTIGUOUS;
  count = nCount;
  baseType = oldType;
  baseIndex = bindex;
  baseSize = baseType->getSize();
  baseExtent = baseType->getExtent();
  size = count * baseSize;
  extent = count * baseExtent;
  numElements = count * baseType->getNumElements();
  lb = baseType->getLB();
  ub = lb + extent;

  if (extent != size) {
    iscontig = false;
  }
  else {
    iscontig = baseType->isContig();
  }
}

size_t
CkDDT_Contiguous::serialize(char* userdata, char* buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_Contiguous::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0  ;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(; num; num--) {
      bytesCopied += baseType->serialize(userdata, buffer, count, dir) ;
      buffer += size;
      userdata += extent;
    }
  }
  return bytesCopied ;
}

void
CkDDT_Contiguous::pupType(PUP::er &p, CkDDT *ddt)
{
  p(datatype);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  p(lb);
  p(ub);
  p(iscontig);
  p(numElements);
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Contiguous::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_CONTIGUOUS;
  return 0;
}

int
CkDDT_Contiguous::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  d[0] = baseIndex;
  return 0;
}

CkDDT_Vector::CkDDT_Vector(int nCount, int blength, int stride, int bindex, CkDDT_DataType* oldType)
{
  datatype = CkDDT_VECTOR;
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;
  baseIndex = bindex;
  baseType =  oldType;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;
  numElements = count * baseType->getNumElements();
  size = count *  blockLength * baseSize ;

  if (strideLength < 0) {
    extent = blockLength*baseExtent + (-1)*strideLength*(count-1)*baseExtent;
  }
  else {
    extent = (count*blockLength + ((strideLength-blockLength)*(count-1))) * baseExtent;
  }

  lb = baseType->getLB();
  if (strideLength < 0) {
    lb += (strideLength*baseExtent*(count-1));
  }
  ub = lb + extent;

  if (extent != size) {
    iscontig = false;
  }
  else {
    if (count==1 || (strideLength==1 && blockLength==1))
      iscontig = baseType->isContig();
    else
      iscontig = false;
  }
}

size_t
CkDDT_Vector::serialize(char* userdata, char* buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_Vector::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0 ;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)blockLength * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(;num;num--) {
      char* saveUserdata = userdata;
      for(int i = 0 ; i < count; i++) {
        bytesCopied += baseType->serialize(userdata, buffer, blockLength, dir);
        buffer += (blockLength*baseSize) ;
        userdata += (strideLength*baseExtent);
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied ;
}

void
CkDDT_Vector::pupType(PUP::er &p, CkDDT* ddt)
{
  p(datatype);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(blockLength);
  p(strideLength);
  p(baseIndex);
  p(lb);
  p(ub);
  p(iscontig);
  p(numElements);
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Vector::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 3;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_VECTOR;
  return 0;
}

int
CkDDT_Vector::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = blockLength;
  i[2] = strideLength;
  d[0] = baseIndex;
  return 0;
}

CkDDT_HVector::CkDDT_HVector(int nCount, int blength, int stride,  int bindex,
                         CkDDT_DataType* oldType)
{
  datatype = CkDDT_HVECTOR;
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;
  baseIndex = bindex;
  baseType = oldType ;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;
  numElements = count * baseType->getNumElements();
  size = count *  blockLength * baseSize ;

  if (strideLength < 0) {
    extent = blockLength*baseExtent + (-1)*strideLength*(count-1);
  }
  else {
    extent = count*blockLength*baseExtent + ((strideLength-blockLength*baseExtent)*(count-1));
  }

  lb = baseType->getLB();
  if (strideLength < 0) {
    lb += (strideLength*(count-1));
  }
  ub = lb + extent;

  if (extent != size) {
    iscontig = false;
  }
  else {
    if (count==1 || (strideLength==1 && blockLength==1))
      iscontig = baseType->isContig();
    else
      iscontig = false;
  }
}

size_t
CkDDT_HVector::serialize(char* userdata, char* buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_HVector::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0 ;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)blockLength * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(;num;num--) {
      char* saveUserdata = userdata;
      for(int i = 0 ; i < count; i++) {
        bytesCopied += baseType->serialize(userdata, buffer, blockLength, dir);
        buffer += (blockLength*baseSize);
        userdata += strideLength;
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied ;
}

void
CkDDT_HVector::pupType(PUP::er &p, CkDDT* ddt)
{
  CkDDT_Vector::pupType(p, ddt);
}

int
CkDDT_HVector::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 2;
  *na = 1;
  *nd = 1;
  *combiner = CkDDT_COMBINER_HVECTOR;
  return 0;
}

int
CkDDT_HVector::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = blockLength;
  a[0] = strideLength;
  d[0] = baseIndex;
  return 0;
}

CkDDT_Indexed::CkDDT_Indexed(int nCount, int* arrBlock, CkDDT_Aint* arrDisp, int bindex,
                         CkDDT_DataType* base)
    : CkDDT_DataType(CkDDT_INDEXED, 0, 0, nCount, numeric_limits<CkDDT_Aint>::max(),
		     numeric_limits<CkDDT_Aint>::min(), 0, base->getSize(), base->getExtent(),
		     base, nCount* base->getNumElements(), bindex),
    arrayBlockLength(new int[nCount]), arrayDisplacements(new CkDDT_Aint[nCount])
{
    CkDDT_Aint positiveExtent = 0;
    CkDDT_Aint negativeExtent = 0;
    for(int i=0; i<count; i++) {
        arrayBlockLength[i] = arrBlock[i] ;
        arrayDisplacements[i] = arrDisp[i] ;
        size += ( arrBlock[i] * baseSize) ;
        if (arrayDisplacements[i] < 0) {
          negativeExtent = std::min(arrayDisplacements[i]*baseExtent, negativeExtent);
          positiveExtent = std::max((arrayDisplacements[i] + arrayBlockLength[i])*baseExtent, positiveExtent);
        }
        else {
          positiveExtent = std::max((arrayDisplacements[i] + arrayBlockLength[i])*baseExtent, positiveExtent);
        }
    }

    extent = positiveExtent + (-1)*negativeExtent;
    lb = baseType->getLB() + *std::min_element(&arrayDisplacements[0],&arrayDisplacements[0] + nCount+1)*baseExtent;
    ub = lb + extent;

    /* set iscontig */
    if (extent != size) {
        iscontig = false;
    }
    else if (count == 1) {
        iscontig = baseType->isContig();
    }
    else {
        bool contig = true;
        for (int j=0; j<count; j++) {
            if (arrayDisplacements[j] != 1 || arrayBlockLength[j] != 1) {
                contig = false;
                break;
            }
        }
        iscontig = (contig && baseType->isContig());
    }
}

size_t
CkDDT_Indexed::serialize(char* userdata, char* buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_Indexed::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0 ;

  if (iscontig) {
    /* arrayBlockLength is either of size 1 or contains all 1s */
    bytesCopied = (size_t)num * (size_t)count * (size_t)arrayBlockLength[1] * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(;num;num--) {
      char* saveUserdata = userdata;
      for(int i = 0 ; i < count; i++) {
        userdata = saveUserdata + baseExtent * arrayDisplacements[i] ;
        for(int j = 0; j < arrayBlockLength[i] ; j++) {
          bytesCopied +=  baseType->serialize(userdata, buffer, 1, dir);
          buffer += baseSize;
          userdata += baseExtent;
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied ;
}

CkDDT_Indexed::~CkDDT_Indexed()
{
  delete [] arrayBlockLength ;
  delete [] arrayDisplacements ;
}

void
CkDDT_Indexed::pupType(PUP::er &p, CkDDT* ddt)
{
  p(datatype);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  p(lb);
  p(ub);
  p(iscontig);
  p(numElements);

  if(p.isUnpacking() )  arrayBlockLength = new int[count] ;
  p(arrayBlockLength, count);

  if(p.isUnpacking() )  arrayDisplacements = new CkDDT_Aint[count] ;
  p(arrayDisplacements, count);

  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Indexed::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count*2+1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_INDEXED;
  return 0;
}

int
CkDDT_Indexed::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    i[z+i[0]+1] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return 0;
}

CkDDT_HIndexed::CkDDT_HIndexed(int nCount, int* arrBlock, CkDDT_Aint* arrDisp,  int bindex,
                           CkDDT_DataType* base)
    : CkDDT_Indexed(nCount, arrBlock, arrDisp, bindex, base)
{
  datatype = CkDDT_HINDEXED;
  size = 0;
  ub = numeric_limits<CkDDT_Aint>::min();
  for (int i = 0; i<count; i++) {
      size += (arrBlock[i] * baseSize);
      ub = std::max(arrBlock[i]*baseExtent + baseType->getLB() + arrayDisplacements[i], ub);
  }

  lb = baseType->getLB() + *std::min_element(arrDisp, arrDisp+nCount+1);
  extent = ub - lb;

  /* set iscontig */
  if (extent != size) {
    iscontig = false;
  }
  else if (count == 1) {
    iscontig = baseType->isContig();
  }
  else {
    bool contig = true;
    for (int j=0; j<count; j++) {
      if (arrayDisplacements[j] != 1 || arrayBlockLength[j] != 1) {
        contig = false;
        break;
      }
    }
    iscontig = (contig && baseType->isContig());
  }
}

size_t
CkDDT_HIndexed::serialize(char* userdata, char* buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_HIndexed::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0 ;

  if (iscontig) {
    /* arrayBlockLength is either of size 1 or contains all 1s */
    bytesCopied = (size_t)num * (size_t)count * (size_t)arrayBlockLength[0] * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(;num;num--) {
      char* saveUserdata = userdata;
      for(int i = 0 ; i < count; i++) {
        userdata = (isAbsolute) ? (char*)arrayDisplacements[i] : saveUserdata+arrayDisplacements[i];
        for(int j = 0; j < arrayBlockLength[i] ; j++) {
          bytesCopied += baseType->serialize(userdata, buffer, 1, dir);
          buffer += baseSize;
          userdata += baseExtent;
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied ;
}

void
CkDDT_HIndexed::pupType(PUP::er &p, CkDDT* ddt)
{
  CkDDT_Indexed::pupType(p, ddt);
}

int
CkDDT_HIndexed::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count+1;
  *na = count;
  *nd = 1;
  *combiner = CkDDT_COMBINER_HINDEXED;
  return 0;
}

int
CkDDT_HIndexed::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return 0;
}

CkDDT_Indexed_Block::CkDDT_Indexed_Block(int count, int Blength, CkDDT_Aint *ArrDisp, int index,
  CkDDT_DataType *type)     : CkDDT_DataType(CkDDT_INDEXED_BLOCK, 0, 0, count, numeric_limits<CkDDT_Aint>::max(),
         numeric_limits<CkDDT_Aint>::min(), 0, type->getSize(), type->getExtent(),
         type, count * type->getNumElements(), index),
    BlockLength(Blength), arrayDisplacements(new CkDDT_Aint[count])
{
  CkDDT_Aint positiveExtent = 0;
  CkDDT_Aint negativeExtent = 0;
  for(int i=0; i<count; i++) {
    arrayDisplacements[i] = ArrDisp[i] ;
    size += ( BlockLength * baseSize) ;
    if (arrayDisplacements[i] < 0) {
      negativeExtent = std::min(arrayDisplacements[i]*baseExtent,negativeExtent);
      positiveExtent = std::max((arrayDisplacements[i]+BlockLength)*baseExtent, positiveExtent);
    }
    else {
      positiveExtent = std::max((arrayDisplacements[i]+BlockLength)*baseExtent, positiveExtent);
    }
  }

  extent = positiveExtent + (-1)*negativeExtent;
  lb = baseType->getLB() + *std::min_element(arrayDisplacements, arrayDisplacements + count+1)*baseExtent;
  ub = lb + extent;

  /* set iscontig */
  if (extent != size) {
    iscontig = false;
  }
  else if (count == 1) {
    iscontig = baseType->isContig();
  }
  else if (BlockLength != 1) {
    iscontig = false;
  }
  else {
    bool contig = true;
    for (int j=0; j<count; j++) {
      if (arrayDisplacements[j] != 1) {
        contig = false;
        break;
      }
    }
    iscontig = (contig && baseType->isContig());
  }
}

CkDDT_Indexed_Block::~CkDDT_Indexed_Block()
{
  delete [] arrayDisplacements;
}

size_t
CkDDT_Indexed_Block::serialize(char *userdata, char *buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_Indexed_Block::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0;

  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)BlockLength * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(;num;num--) {
      char* saveUserdata = userdata;
      for(int i = 0 ; i < count; i++) {
        userdata = saveUserdata + baseExtent * arrayDisplacements[i] ;
        for(int j = 0; j < BlockLength ; j++) {
          bytesCopied +=  baseType->serialize(userdata, buffer, 1, dir);
          buffer += baseSize;
          userdata += baseExtent;
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_Indexed_Block::pupType(PUP::er &p, CkDDT *ddt)
{
  p(datatype);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  p(lb);
  p(ub);
  p(iscontig);
  p(numElements);
  p(BlockLength);

  if(p.isUnpacking() )  arrayDisplacements = new CkDDT_Aint[count] ;
  p(arrayDisplacements, count);

  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Indexed_Block::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count*2+1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_INDEXED_BLOCK;
  return 0;
}

int
CkDDT_Indexed_Block::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = BlockLength;
  for(int z=0;z<i[0];z++) {
    i[z+2] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return 0;
}

CkDDT_HIndexed_Block::CkDDT_HIndexed_Block(int count, int Blength, CkDDT_Aint *ArrDisp, int index,
  CkDDT_DataType *type)     : CkDDT_Indexed_Block(count, Blength,ArrDisp,index,type)
{
  CkDDT_Aint positiveExtent = 0;
  CkDDT_Aint negativeExtent = 0;
  for(int i=0; i<count; i++) {
    arrayDisplacements[i] = ArrDisp[i] ;
    size += ( BlockLength * baseSize) ;
    if (this->arrayDisplacements[i] < 0) {
      negativeExtent = std::min(arrayDisplacements[i],negativeExtent);
      positiveExtent = std::max(arrayDisplacements[i] + BlockLength*baseExtent, positiveExtent);
    }
    else {
      positiveExtent = std::max(arrayDisplacements[i] + BlockLength*baseExtent, positiveExtent);
    }
  }

  extent = positiveExtent + (-1)*negativeExtent;
  lb = baseType->getLB() + *std::min_element(arrayDisplacements, arrayDisplacements + count+1);
  ub = lb + extent;

  /* set iscontig */
  if (extent != size) {
    iscontig = false;
  }
  else if (count == 1) {
    iscontig = baseType->isContig();
  }
  else if (BlockLength != 1) {
    iscontig = false;
  }
  else {
    bool contig = true;
    for (int j=0; j<count; j++) {
      if (arrayDisplacements[j] != 1) {
        contig = false;
        break;
      }
    }
    iscontig = (contig && baseType->isContig());
  }
}

CkDDT_HIndexed_Block::~CkDDT_HIndexed_Block()
{
  delete [] arrayDisplacements;
}

size_t
CkDDT_HIndexed_Block::serialize(char *userdata, char *buffer, int num, int dir) const
{
  DDTDEBUG("CkDDT_HIndexed_Block::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==1)?"packing":"unpacking", num, baseType->getType(), iscontig);
  size_t bytesCopied = 0;

  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)BlockLength * (size_t)baseSize ;
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    for(;num;num--) {
      char* saveUserdata = userdata;
      for(int i = 0 ; i < count; i++) {
        userdata = (isAbsolute) ? (char*)arrayDisplacements[i] : saveUserdata+arrayDisplacements[i];
        for(int j = 0; j < BlockLength ; j++) {
          bytesCopied +=  baseType->serialize(userdata, buffer, 1, dir);
          buffer += baseSize;
          userdata += baseExtent ;
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_HIndexed_Block::pupType(PUP::er &p, CkDDT *ddt)
{
  p(datatype);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  p(lb);
  p(ub);
  p(iscontig);
  p(numElements);
  p(BlockLength);

  if(p.isUnpacking() )  arrayDisplacements = new CkDDT_Aint[count] ;
  p(arrayDisplacements, count);

  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_HIndexed_Block::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count*2+1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_HINDEXED_BLOCK;
  return 0;
}

int
CkDDT_HIndexed_Block::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = BlockLength;
  for(int z=0;z<i[0];z++) {
    i[z+2] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return 0;
}

CkDDT_Struct::CkDDT_Struct(int nCount, int* arrBlock,
                       CkDDT_Aint* arrDisp, int *bindex, CkDDT_DataType** arrBase)
    : CkDDT_DataType(CkDDT_STRUCT, 0, 0, nCount, numeric_limits<CkDDT_Aint>::max(),
    numeric_limits<CkDDT_Aint>::min(), 0, 0, 0, NULL, 0, 0),
    arrayBlockLength(new int[nCount]), arrayDisplacements(new CkDDT_Aint[nCount]),
    index(new int[nCount]), arrayDataType(new CkDDT_DataType*[nCount])
{
  int saveExtent = 0;
  for (int i=0; i<count; i++) {
      arrayBlockLength[i] = arrBlock[i];
      arrayDisplacements[i] = arrDisp[i];
      arrayDataType[i] =  arrBase[i];
      numElements += arrayBlockLength[i] * arrayDataType[i]->getNumElements();
      index[i] = bindex[i];
      size += arrBlock[i]*arrayDataType[i]->getSize();
      if (arrayDataType[i]->getExtent() > saveExtent) {
        saveExtent = arrayDataType[i]->getExtent();
      }
  }

  bool explicit_lb = false;
  bool explicit_ub = false;
  for (int i=0; i<count; i++) {
      CkDDT_Aint xlb = arrayDataType[i]->getLB() + arrayDisplacements[i];
      CkDDT_Aint xub = arrayDisplacements[i] + arrayBlockLength[i]*arrayDataType[i]->getUB();
      if (arrayDataType[i]->getType() == CkDDT_LB) {
          if (!explicit_lb) lb = xlb;
          explicit_lb = true;
          if (xlb < lb) lb = xlb;
      } else if(arrayDataType[i]->getType() == CkDDT_UB) {
          if (!explicit_ub) ub = xub;
          explicit_ub = true;
          if (xub > ub) ub = xub;
      } else {
          if (!explicit_lb && xlb < lb) lb = xlb;
          if (!explicit_ub && xub > ub) ub = xub;
      }
  }

  extent = ub - lb;
  if (!explicit_ub && (saveExtent != 0) && (extent % saveExtent != 0)) {
      extent += (saveExtent - (extent % saveExtent));
  }

  /* set iscontig */
  if (extent != size) {
    iscontig = false;
  }
  else if (count == 1) {
    iscontig = arrayDataType[0]->isContig();
  }
  else {
    bool contig = true;
    for (int j=0; j<count; j++) {
      if (arrayDisplacements[j] != 1 || arrayBlockLength[j] != 1) {
        contig = false;
        break;
      }
    }
    if (contig) {
      for (int j=0; j<count; j++) {
        if (!arrayDataType[j]->isContig()) {
          contig = false;
          break;
        }
      }
      iscontig = contig;
    }
    else {
      iscontig = false;
    }
  }
  DDTDEBUG("type %d: ub=%ld, lb=%ld, extent=%ld, size=%d, iscontig=%d\n",datatype,ub,lb,extent,size,iscontig);
}

size_t
CkDDT_Struct::serialize(char* userdata, char* buffer, int num, int dir) const
{
  size_t bytesCopied = 0;

  if (iscontig) {
    DDTDEBUG("CkDDT_Struct::serialize, %s %d objects (iscontig=%d)\n",
             (dir==1)?"packing":"unpacking", num, iscontig);
    /* arrayBlockLength is either of size 1 or contains all 1s */
    for (int i=0; i<count; i++)
      bytesCopied += (size_t)num * (size_t)arrayBlockLength[0] * (size_t)arrayDataType[i]->getSize();
    serializeContig(userdata, buffer, bytesCopied, dir);
  }
  else {
    char* sbuf = userdata;
    char* dbuf = buffer;
    char* absoluteAddr = 0;
    for (; num; num--) {
      char *buf = buffer;
      for (int i=0; i<count; i++) {
        int saveSize = arrayDataType[i]->getSize();
        int saveExtent = arrayDataType[i]->getExtent();
        for (int j=0; j<arrayBlockLength[i]; j++) {
          DDTDEBUG("CkDDT_Struct::serialize %s block of type %d (size %d) from offset %d to offset %d\n",
                   (dir==1)?"packing":"unpacking", arrayDataType[i]->getType(),
                   saveSize, userdata + (j*saveExtent) + arrayDisplacements[i]-sbuf, buffer-dbuf);
          bytesCopied += arrayDataType[i]->serialize(
                         arrayDisplacements[i] + j*saveExtent + (isAbsolute ? absoluteAddr : userdata),
                         buffer,
                         1,
                         dir);
          buffer += saveSize;
        }
      }
      buffer = buf + size;
      userdata += extent;
      absoluteAddr += extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_Struct::pupType(PUP::er &p, CkDDT* ddt)
{
  p(datatype);
  p(size);
  p(extent);
  p(count);
  p(lb);
  p(ub);
  p(iscontig);
  p(numElements);
  if(p.isUnpacking())
  {
    arrayBlockLength = new int[count] ;
    arrayDisplacements = new CkDDT_Aint[count] ;
    index = new int[count] ;
    arrayDataType = new CkDDT_DataType*[count] ;
  }
  p(arrayBlockLength, count);
  p(arrayDisplacements, count);
  p(index, count);

  if(p.isUnpacking())
    for(int i=0 ; i < count; i++)
      arrayDataType[i] = ddt->getType(index[i]);
}

int
CkDDT_Struct::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count+1;
  *na = count;
  *nd = count;
  *combiner = CkDDT_COMBINER_STRUCT;
  return 0;
}

int
CkDDT_Struct::getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const
{
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
    d[z] = index[z];
  }
  return 0;
}

