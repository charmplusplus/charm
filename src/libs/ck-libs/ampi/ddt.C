#include "ddt.h"
#include <algorithm>
#include <limits>

using std::numeric_limits;

//Uncomment for debug print statements
#define DDTDEBUG(...) //CkPrintf(__VA_ARGS__)


/* Serialize a contiguous chunk of memory */
static inline void serializeContig(char* userdata, char* buffer, size_t size, int dir)
{
  if (dir==1) {
    memcpy(buffer, userdata, size);
  } else if (dir==-1) {
    memcpy(userdata, buffer, size);
  }
#if CMK_ERROR_CHECKING
  else {
    CkAbort("CkDDT: Invalid dir given to serialize a contiguous type!");
  }
#endif
}

CkDDT_DataType*
CkDDT::getType(int nIndex) const
{
#if CMK_ERROR_CHECKING
  if (nIndex < 0 || nIndex > typeTable.size())
    CkAbort("CkDDT: Invalid type index passed to getType!");
#endif
  return typeTable[nIndex];
}

void
CkDDT::pup(PUP::er &p)
{
  p|types;
  if (p.isUnpacking())
  {
    typeTable.resize(types.size(), nullptr);
    for(int i=0 ; i < types.size(); i++)
    {
      switch(types[i])
      {
        case MPI_DATATYPE_NULL:
          break;
        case CkDDT_CONTIGUOUS:
          typeTable[i] = new CkDDT_Contiguous;
          break;
        case CkDDT_VECTOR:
          typeTable[i] = new CkDDT_Vector;
          break;
        case CkDDT_HVECTOR:
          typeTable[i] = new CkDDT_HVector;
          break;
        case CkDDT_INDEXED:
          typeTable[i] = new CkDDT_Indexed;
          break;
        case CkDDT_HINDEXED:
          typeTable[i] = new CkDDT_HIndexed;
          break;
        case CkDDT_INDEXED_BLOCK:
          typeTable[i] = new CkDDT_Indexed_Block;
          break;
        case CkDDT_HINDEXED_BLOCK:
          typeTable[i] = new CkDDT_HIndexed_Block;
          break;
        case CkDDT_STRUCT:
          typeTable[i] = new CkDDT_Struct;
          break;
        default: //CkDDT_PRIMITIVE
          typeTable[i] = new CkDDT_DataType;
          break;
      }
    }
  }

  for(int i=0; i < types.size(); i++)
  {
    if(types[i] != MPI_DATATYPE_NULL)
    {
      typeTable[i]->pupType(p, this);
    }
  }
}

void
CkDDT::freeType(int index)
{
  CkAssert(types.size() == typeTable.size());
  if (index > CkDDT_MAX_PRIMITIVE_TYPE) {
    // Decrement the ref count and free the type if there are no references to it.
    if (typeTable[index]->decRefCount() == 0) {
      // Remove a reference from this type's base type(s).
      if (typeTable[index]->getType() == CkDDT_STRUCT) {
        int count = typeTable[index]->getCount();
        vector<int> baseIndices = (static_cast<CkDDT_Struct&> (*typeTable[index])).getBaseIndices();
        for (int i=0; i < count; i++)
          freeType(baseIndices[i]);
      } else {
        freeType(typeTable[index]->getBaseIndex());
      }

      // Free non-primitive type
      delete typeTable[index];
      typeTable[index] = nullptr;
      types[index] = MPI_DATATYPE_NULL;
      // Free all NULL types from back of typeTable
      while (typeTable.back() == nullptr) {
        typeTable.pop_back();
        CkAssert(types.back() == MPI_DATATYPE_NULL);
        types.pop_back();
      }
    }
  }
}

CkDDT::~CkDDT()
{
  for(int i=0; i < typeTable.size(); i++)
  {
    if(typeTable[i] != nullptr)
    {
       delete typeTable[i];
    }
  }
}

int
CkDDT::insertType(CkDDT_DataType* ptr, int type)
{
  // Search thru non-primitive types for a free one first:
  CkAssert(types.size() == typeTable.size());
  for (int i=CkDDT_MAX_PRIMITIVE_TYPE+1; i<types.size(); i++) {
    if (types[i] == MPI_DATATYPE_NULL) {
      types[i] = type;
      typeTable[i] = ptr;
      return i;
    }
  }
  types.push_back(type);
  typeTable.push_back(ptr);
  return types.size()-1;
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

MPI_Aint
CkDDT::getExtent(int nIndex) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getExtent();
}

MPI_Aint
CkDDT::getLB(int nIndex) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getLB();
}

MPI_Aint
CkDDT::getUB(int nIndex) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getUB();
}

MPI_Aint
CkDDT::getTrueExtent(int nIndex) const
{
  CkDDT_DataType *dttype = getType(nIndex);
  return dttype->getTrueExtent();
}

MPI_Aint
CkDDT::getTrueLB(int nIndex) const
{
  CkDDT_DataType *dttype = getType(nIndex);
  return dttype->getTrueLB();
}

void
CkDDT::createDup(int nIndexOld, int *nIndexNew)
{
  CkDDT_DataType *dttype = getType(nIndexOld);
  CkDDT_DataType *type;

  switch(dttype->getType()) {
    case CkDDT_CONTIGUOUS:
      type = new CkDDT_Contiguous(static_cast<CkDDT_Contiguous&> (*dttype));
      break;
    case CkDDT_VECTOR:
      type = new CkDDT_Vector(static_cast<CkDDT_Vector&> (*dttype));
      break;
    case CkDDT_HVECTOR:
      type = new CkDDT_HVector(static_cast<CkDDT_HVector&> (*dttype));
      break;
    case CkDDT_INDEXED:
      type = new CkDDT_Indexed(static_cast<CkDDT_Indexed&> (*dttype));
      break;
    case CkDDT_HINDEXED:
      type = new CkDDT_HIndexed(static_cast<CkDDT_HIndexed&> (*dttype));
      break;
    case CkDDT_STRUCT:
      type = new CkDDT_Struct(static_cast<CkDDT_Struct&> (*dttype));
      break;
    case CkDDT_INDEXED_BLOCK:
      type = new CkDDT_Indexed_Block(static_cast<CkDDT_Indexed_Block&> (*dttype));
      break;
    case CkDDT_HINDEXED_BLOCK:
      type = new CkDDT_HIndexed_Block(static_cast<CkDDT_HIndexed_Block&> (*dttype));
      break;
    default:
      type = new CkDDT_DataType(*dttype);
      break;
  }

  *nIndexNew = insertType(type, types[nIndexOld]);
}

int CkDDT::getEnvelope(int nIndex, int *ni, int *na, int *nd, int *combiner) const
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getEnvelope(ni, na, nd, combiner);
}

int CkDDT::getContents(int nIndex, int ni, int na, int nd, int i[], MPI_Aint a[], int d[])
{
  CkDDT_DataType* dttype = getType(nIndex);
  int ret = dttype->getContents(ni, na, nd, i, a, d);
  if (dttype->getType() == CkDDT_STRUCT) {
    int count = dttype->getCount();
    vector<CkDDT_DataType*> baseTypes = (static_cast<CkDDT_Struct&> (*dttype)).getBaseTypes();
    for (int i=0; i < count; i++)
      baseTypes[i]->incRefCount();
  } else {
    dttype->getBaseType()->incRefCount();
  }
  return ret;
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
CkDDT::createResized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newType)
{
  CkDDT_DataType *dttype = getType(oldtype);
  CkDDT_DataType *type;

  switch(dttype->getType()) {
    case CkDDT_CONTIGUOUS:
      type = new CkDDT_Contiguous(static_cast<CkDDT_Contiguous&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_VECTOR:
      type = new CkDDT_Vector(static_cast<CkDDT_Vector&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_HVECTOR:
      type = new CkDDT_HVector(static_cast<CkDDT_HVector&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_INDEXED:
      type = new CkDDT_Indexed(static_cast<CkDDT_Indexed&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_HINDEXED:
      type = new CkDDT_HIndexed(static_cast<CkDDT_HIndexed&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_STRUCT:
      type = new CkDDT_Struct(static_cast<CkDDT_Struct&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_INDEXED_BLOCK:
      type = new CkDDT_Indexed_Block(static_cast<CkDDT_Indexed_Block&> (*dttype));
      type->setSize(lb, extent);
      break;
    case CkDDT_HINDEXED_BLOCK:
      type = new CkDDT_HIndexed_Block(static_cast<CkDDT_HIndexed_Block&> (*dttype));
      type->setSize(lb, extent);
      break;
    default:
      type = new CkDDT_DataType(*dttype, lb, extent);
      break;
  }

  *newType = insertType(type, types[oldtype]);
}

void
CkDDT::newContiguous(int count, MPI_Datatype oldType, MPI_Datatype *newType)
{
  CkDDT_DataType *type = new CkDDT_Contiguous(count, oldType, typeTable[oldType]);
  *newType = insertType(type, CkDDT_CONTIGUOUS);
}

void
CkDDT::newVector(int count, int blocklength, int stride,
                 MPI_Datatype oldType, MPI_Datatype* newType)
{
  CkDDT_DataType* type = new CkDDT_Vector(count, blocklength, stride, oldType, typeTable[oldType]);
  *newType = insertType(type, CkDDT_VECTOR);
}

void
CkDDT::newHVector(int count, int blocklength, int stride,
                  MPI_Datatype oldtype, MPI_Datatype* newType)
{
  CkDDT_DataType* type =
    new CkDDT_HVector(count, blocklength, stride, oldtype, typeTable[oldtype]);
  *newType = insertType(type, CkDDT_HVECTOR);
}

void
CkDDT::newIndexed(int count, const int* arrbLength, MPI_Aint* arrDisp,
                  MPI_Datatype oldtype, MPI_Datatype* newType)
{
  CkDDT_DataType* type =
    new CkDDT_Indexed(count, arrbLength, arrDisp, oldtype, typeTable[oldtype]);
  *newType = insertType(type, CkDDT_INDEXED);
}

void
CkDDT::newHIndexed(int count, const int* arrbLength, const MPI_Aint* arrDisp,
                   MPI_Datatype oldtype, MPI_Datatype* newType)
{
  CkDDT_DataType* type =
    new CkDDT_HIndexed(count, arrbLength, arrDisp, oldtype, typeTable[oldtype]);
  *newType = insertType(type, CkDDT_HINDEXED);
}

void
CkDDT::newIndexedBlock(int count, int Blocklength, const int *arrDisp, MPI_Datatype oldtype,
                       MPI_Datatype *newType)
{
  // Convert arrDisp from an array of int's to an array of MPI_Aint's. This is needed because
  // MPI_Type_create_indexed_block takes ints and MPI_Type_create_hindexed_block takes MPI_Aint's
  // and we use Indexed_Block to represent both of those datatypes internally.
  std::vector<MPI_Aint> arrDispAint(count);
  for (int i=0; i<count; i++) {
    arrDispAint[i] = static_cast<MPI_Aint>(arrDisp[i]);
  }
  CkDDT_DataType *type = new CkDDT_Indexed_Block(count, Blocklength, arrDispAint.data(), oldtype, typeTable[oldtype]);
  *newType = insertType(type, CkDDT_INDEXED_BLOCK);
}

void
CkDDT::newHIndexedBlock(int count, int Blocklength, const MPI_Aint *arrDisp, MPI_Datatype oldtype,
                  MPI_Datatype *newType)
{
  CkDDT_DataType *type = new CkDDT_HIndexed_Block(count, Blocklength, arrDisp, oldtype, typeTable[oldtype]);
  *newType = insertType(type, CkDDT_HINDEXED_BLOCK);
}

void
CkDDT::newStruct(int count, const int* arrbLength, const MPI_Aint* arrDisp,
                 const MPI_Datatype *oldtype, MPI_Datatype* newType)
{
  vector<CkDDT_DataType *> olddatatypes(count);
  for(int i=0;i<count;i++){
    olddatatypes[i] = getType(oldtype[i]);
  }

  CkDDT_DataType* type =
    new CkDDT_Struct(count, arrbLength, arrDisp, oldtype, olddatatypes.data());
  *newType = insertType(type, CkDDT_STRUCT);
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
    case MPI_DOUBLE:
      size = sizeof(double);
      break;
    case MPI_INT:
      size = sizeof(signed int);
      break;
    case MPI_FLOAT:
      size = sizeof(float);
      break;
    case MPI_CHAR:
      size = sizeof(char);
      break;
    case MPI_BYTE:
      size = 1 ;
      break;
    case MPI_PACKED:
      size = 1 ;
      break;
    case MPI_COMPLEX:
    case MPI_FLOAT_COMPLEX:
      size =  2 * sizeof(float) ;
      break;
    case MPI_DOUBLE_COMPLEX:
      size =  2 * sizeof(double) ;
      break;
    case MPI_LONG_DOUBLE_COMPLEX:
      size =  2 * sizeof(long double) ;
      break;
    case MPI_C_BOOL:
      /* Should be C99 _Bool instead of C++ bool, but MSVC doesn't support that */
      size = sizeof(bool) ;
      break;
    case MPI_LOGICAL:
      size =  sizeof(int) ;
      break;
    case MPI_SHORT:
      size = sizeof(signed short int);
      break ;
    case MPI_LONG:
      size = sizeof(signed long);
      break ;
    case MPI_UNSIGNED_CHAR:
      size = sizeof(unsigned char);
      break;
    case MPI_UNSIGNED_SHORT:
      size = sizeof(unsigned short);
      break;
    case MPI_UNSIGNED:
      size = sizeof(unsigned);
      break ;
    case MPI_UNSIGNED_LONG:
      size = sizeof(unsigned long);
      break ;
    case MPI_LONG_DOUBLE:
      size = sizeof(long double);
      break ;
    case MPI_FLOAT_INT:
      size = sizeof(FloatInt);
      break;
    case MPI_DOUBLE_INT:
      size = sizeof(DoubleInt);
      break;
    case MPI_LONG_INT:
      size = sizeof(LongInt);
      break;
    case MPI_2INT:
      size = sizeof(IntInt);
      break;
    case MPI_SHORT_INT:
      size = sizeof(ShortInt);
      break;
    case MPI_LONG_DOUBLE_INT:
      size = sizeof(LongdoubleInt);
      break;
    case MPI_2FLOAT:
      size = sizeof(FloatFloat);
      break;
    case MPI_2DOUBLE:
      size = sizeof(DoubleDouble);
      break;
    case MPI_SIGNED_CHAR:
      size = sizeof(signed char);
      break;
    case MPI_UNSIGNED_LONG_LONG:
      size = sizeof(unsigned long long);
      break;
    case MPI_WCHAR:
      size = sizeof(wchar_t);
      break;
    case MPI_INT8_T:
      size = sizeof(int8_t);
      break;
    case MPI_INT16_T:
      size = sizeof(int16_t);
      break;
    case MPI_INT32_T:
      size = sizeof(int32_t);
      break;
    case MPI_INT64_T:
      size = sizeof(int64_t);
      break;
    case MPI_UINT8_T:
      size = sizeof(uint8_t);
      break;
    case MPI_UINT16_T:
      size = sizeof(uint16_t);
      break;
    case MPI_UINT32_T:
      size = sizeof(uint32_t);
      break;
    case MPI_UINT64_T:
      size = sizeof(uint64_t);
      break;
    case MPI_AINT:
      size = sizeof(MPI_Aint);
      break;
    case MPI_LB:
    case MPI_UB:
      size = 0;
      break;
#if CMK_LONG_LONG_DEFINED
    case MPI_LONG_LONG_INT:
      size = sizeof(signed long long);
      break;
#endif
    default:
      size = 0;
  }

  extent      = size;
  lb          = 0;
  ub          = size;
  trueExtent  = size;
  trueLB      = 0;
  iscontig    = true;
  baseType    = NULL;
  baseSize    = 0;
  baseExtent  = 0;
  baseIndex   = -1;
  numElements = 1;
  refCount    = 1;

  DDTDEBUG("CkDDT_DataType constructor: type=%d, size=%d, extent=%ld, iscontig=%d\n",
           type, size, extent, iscontig);
}


CkDDT_DataType::CkDDT_DataType(int datatype, int size, MPI_Aint extent, int count, MPI_Aint lb, MPI_Aint ub,
            bool iscontig, int baseSize, MPI_Aint baseExtent, CkDDT_DataType* baseType, int numElements, int baseIndex,
            MPI_Aint trueExtent, MPI_Aint trueLB) :
    datatype(datatype), refCount(1), size(size), extent(extent), count(count), lb(lb), ub(ub), trueExtent(trueExtent),
    trueLB(trueLB), iscontig(iscontig), baseSize(baseSize), baseExtent(baseExtent), baseType(baseType),
    numElements(numElements), baseIndex(baseIndex), isAbsolute(false)
{
  if (baseType)
    baseType->incRefCount();
}

CkDDT_DataType::CkDDT_DataType(const CkDDT_DataType &obj)  :
  datatype(obj.datatype)
  ,refCount(1)
  ,size(obj.size)
  ,extent(obj.extent)
  ,count(obj.count)
  ,lb(obj.lb)
  ,ub(obj.ub)
  ,trueExtent(obj.trueExtent)
  ,trueLB(obj.trueLB)
  ,iscontig(obj.iscontig)
  ,baseSize(obj.baseSize)
  ,baseExtent(obj.baseExtent)
  ,baseType(obj.baseType)
  ,numElements(obj.numElements)
  ,baseIndex(obj.baseIndex)
  ,isAbsolute(obj.isAbsolute)
  ,name(obj.name)
{
  if (baseType)
    baseType->incRefCount();
}

// TODO: this constructor can be combined with the previous one in C++11.
CkDDT_DataType::CkDDT_DataType(const CkDDT_DataType &obj, MPI_Aint _lb, MPI_Aint _extent)
{
  datatype    = obj.datatype;
  refCount    = 1;
  baseSize    = obj.baseSize;
  baseExtent  = obj.baseExtent;
  baseType    = obj.baseType;
  baseIndex   = obj.baseIndex;
  numElements = obj.numElements;
  size        = obj.size;
  trueExtent  = obj.trueExtent;
  trueLB      = obj.trueLB;
  count       = obj.count;
  isAbsolute  = obj.isAbsolute;
  name        = obj.name;

  if (baseType)
    baseType->incRefCount();

  setSize(_lb, _extent);
}

void
CkDDT_DataType::setSize(MPI_Aint _lb, MPI_Aint _extent)
{
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
  CkDDT_SetName(name, src);
}

void
CkDDT_DataType::getName(char *dest, int *len) const
{
  int length = *len = name.size();
  memcpy(dest, &name[0], length);
  dest[length] = '\0';
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
  return num*size;
}

MPI_Aint
CkDDT_DataType::getExtent() const
{
  return extent;
}

int
CkDDT_DataType::getCount() const
{
  return count;
}

MPI_Aint
CkDDT_DataType::getTrueExtent() const
{
  return trueExtent;
}

MPI_Aint
CkDDT_DataType::getTrueLB() const
{
  return trueLB;
}

int
CkDDT_DataType::getBaseSize() const
{
  return baseSize;
}

MPI_Aint
CkDDT_DataType::getBaseExtent() const
{
  return baseExtent;
}

CkDDT_DataType*
CkDDT_DataType::getBaseType() const
{
  return baseType;
}

int
CkDDT_DataType::getNumElements() const
{
  return numElements;
}

MPI_Aint
CkDDT_DataType::getLB() const
{
  return lb;
}

MPI_Aint
CkDDT_DataType::getUB() const
{
  return ub;
}

int
CkDDT_DataType::getBaseIndex() const
{
  return baseIndex;
}

int
CkDDT_DataType::getType() const
{
  return datatype;
}

void
CkDDT_DataType::incRefCount()
{
  CkAssert(refCount > 0);
  if (datatype > CkDDT_MAX_PRIMITIVE_TYPE)
    refCount++;
}

// Callers of this function should always check its return value and free the type only if it returns 0.
int
CkDDT_DataType::decRefCount(void)
{
  CkAssert(refCount > 0);
  if (datatype > CkDDT_MAX_PRIMITIVE_TYPE)
    return --refCount;
  return -1;
}

void
CkDDT_DataType::pupType(PUP::er  &p, CkDDT* ddt)
{
  p|datatype;
  p|refCount;
  p|size;
  p|extent;
  p|count;
  p|baseSize;
  p|baseExtent;
  p|baseIndex;
  p|trueExtent;
  p|trueLB;
  p|lb;
  p|ub;
  p|iscontig;
  p|isAbsolute;
  p|numElements;
  p|name;
  if (p.isUnpacking()) baseType = NULL;
}

int
CkDDT_DataType::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 0;
  *na = 0;
  *nd = 0;
  *combiner = CkDDT_COMBINER_NAMED;
  return MPI_SUCCESS;
}

int
CkDDT_DataType::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  CkPrintf("CkDDT_DataType::getContents: Shouldn't call getContents on primitive datatypes!\n");
  return MPI_ERR_TYPE;
}

CkDDT_Contiguous::CkDDT_Contiguous(int nCount, int bindex, CkDDT_DataType* oldType)
{
  datatype = CkDDT_CONTIGUOUS;
  count = nCount;
  baseType = oldType;
  baseIndex = bindex;
  baseSize = baseType->getSize();
  baseExtent = baseType->getExtent();
  refCount = 1;
  baseType->incRefCount();
  size = count * baseSize;
  numElements = count * baseType->getNumElements();

  if(baseType->getLB() > baseType->getUB()) {
    lb = baseType->getLB() + (baseExtent*(count-1));
    ub = baseType->getUB();
    trueLB = baseType->getTrueLB() + (baseExtent*(count-1));
    trueExtent = baseType->getTrueLB() + baseType->getTrueExtent() - ((count-1)*baseExtent);
  } else {
    lb = baseType->getLB();
    ub = lb + count * baseExtent;
    trueLB = baseType->getTrueLB();
    trueExtent = ((count - 1) * baseExtent) + baseType->getTrueExtent();
  }

  extent = ub - lb;

  if (extent != size || count == 0) {
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
  CkDDT_DataType::pupType(p, ddt);
  if (p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Contiguous::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_CONTIGUOUS;
  return MPI_SUCCESS;
}

int
CkDDT_Contiguous::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  d[0] = baseIndex;
  return MPI_SUCCESS;
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
  refCount = 1;
  baseType->incRefCount();
  numElements = count * baseType->getNumElements();
  size = count *  blockLength * baseSize ;

  int absBaseExtent = std::abs(baseExtent);
  int absStrideLength = std::abs(strideLength);

  if(baseType->getLB() > baseType->getUB()) {
    if (strideLength > 0) {
      // Negative Extent with positive stride
      lb = baseType->getUB() + (((strideLength*count)-2-(absStrideLength-blockLength))*baseExtent);
      ub = baseType->getUB();
      trueLB = lb - baseType->getLB();
    } else {
      // Negative extent and stride
      lb = baseType->getLB() + ((blockLength-1)*baseExtent);
      ub = baseType->getUB() + (strideLength*(count-1)*baseExtent);
      trueLB = baseType->getLB() - baseType->getUB() + (blockLength*baseExtent);
    }
  } else {
    if (strideLength > 0) {
      // Positive extent and stride
      lb = baseType->getLB();
      ub = lb + (count*blockLength + ((strideLength-blockLength)*(count-1))) * baseExtent;
      trueLB = baseType->getTrueLB();
    } else {
      // Negative stride and positive extent
      lb = baseType->getLB() + (strideLength*baseExtent*(count-1));
      ub = lb + blockLength*baseExtent + absStrideLength*(count-1)*baseExtent;
      trueLB = baseType->getTrueLB() + ((count-1) * strideLength * baseType->getExtent());
    }
  }

  extent = ub - lb;

  if (absStrideLength < blockLength) {
    trueExtent =
      ((count-1) * stride * absBaseExtent) +
      (blockLength * absBaseExtent) -
      (absBaseExtent - baseType->getTrueExtent());
  } else {
    trueExtent = (((absStrideLength*count)-(absStrideLength-blockLength))*absBaseExtent) - (absBaseExtent - baseType->getTrueExtent());
  }

  if (extent != size || count == 0) {
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
  CkDDT_DataType::pupType(p, ddt);
  p|blockLength;
  p|strideLength;
  if (p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Vector::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 3;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_VECTOR;
  return MPI_SUCCESS;
}

int
CkDDT_Vector::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = blockLength;
  i[2] = strideLength;
  d[0] = baseIndex;
  return MPI_SUCCESS;
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
  refCount = 1;
  baseType->incRefCount();
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

  trueExtent = extent;
  trueLB = lb;
  if (extent != size || count == 0) {
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
  return MPI_SUCCESS;
}

int
CkDDT_HVector::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = blockLength;
  a[0] = strideLength;
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

CkDDT_Indexed::CkDDT_Indexed(int nCount, const int* arrBlock, const MPI_Aint* arrDisp, int bindex,
                         CkDDT_DataType* base)
    : CkDDT_DataType(CkDDT_INDEXED, 0, 0, nCount, numeric_limits<MPI_Aint>::max(),
		     numeric_limits<MPI_Aint>::min(), 0, base->getSize(), base->getExtent(),
		     base, nCount* base->getNumElements(), bindex, 0, 0),
    arrayBlockLength(nCount), arrayDisplacements(nCount)
{
    MPI_Aint positiveExtent = 0;
    MPI_Aint negativeExtent = 0;
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

    if (count == 0) {
        lb = 0;
        ub = 0;
    } else {
        int i=0;
        while (arrayBlockLength[i] == 0) {
            /* Find lowest index that isn't empty */
            i++;
            if (i == count -1) {
                i = 0;
                break;
            }
        }

        lb = baseType->getLB() + arrayDisplacements[i]*baseExtent;

        int j = count-1;
        while (arrayBlockLength[j] == 0) {
            /* Find highest index that isn't empty */
            j--;
            if (j == 0) {
                break;
            }
        }
        ub = baseType->getLB() + (arrayBlockLength[j] + arrayDisplacements[j])*baseExtent;
    }

    trueExtent = extent;
    trueLB = lb;

    /* set iscontig */
    if (extent != size || count == 0) {
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
{}

void
CkDDT_Indexed::pupType(PUP::er &p, CkDDT* ddt)
{
  CkDDT_DataType::pupType(p, ddt);
  p|arrayBlockLength;
  p|arrayDisplacements;
  if (p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Indexed::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count*2+1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_INDEXED;
  return MPI_SUCCESS;
}

int
CkDDT_Indexed::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    i[z+i[0]+1] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

CkDDT_HIndexed::CkDDT_HIndexed(int nCount, const int* arrBlock, const MPI_Aint* arrDisp,  int bindex,
                           CkDDT_DataType* base)
    : CkDDT_Indexed(nCount, arrBlock, arrDisp, bindex, base)
{
  datatype = CkDDT_HINDEXED;
  size = 0;
  ub = numeric_limits<MPI_Aint>::min();
  for (int i = 0; i<count; i++) {
      size += (arrBlock[i] * baseSize);
      ub = std::max(arrBlock[i]*baseExtent + baseType->getLB() + arrayDisplacements[i], ub);
  }

  if (count == 0) {
    lb = 0;
    ub = 0;
  } else {
    int i=0;
    while (arrayBlockLength[i] == 0) {
      /* Find lowest index that isn't empty */
      i++;
      if (i == count -1) {
        i = 0;
        break;
      }
    }

    lb = baseType->getLB() + arrayDisplacements[i];

    int j = count-1;
    while (arrayBlockLength[j] == 0) {
      /* Find highest index that isn't empty */
      j--;
      if (j == 0) {
        break;
      }
    }
    ub = baseType->getLB() + (arrayBlockLength[j]*baseExtent) +  arrayDisplacements[j];
  }
  extent = ub - lb;

  trueExtent = extent;
  trueLB = lb;

  /* set iscontig */
  if (extent != size || count == 0) {
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
  return MPI_SUCCESS;
}

int
CkDDT_HIndexed::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

CkDDT_Indexed_Block::CkDDT_Indexed_Block(int count, int Blength, const MPI_Aint *ArrDisp, int index,
  CkDDT_DataType *type)     : CkDDT_DataType(CkDDT_INDEXED_BLOCK, 0, 0, count, numeric_limits<MPI_Aint>::max(),
         numeric_limits<MPI_Aint>::min(), 0, type->getSize(), type->getExtent(),
         type, count * type->getNumElements(), index, 0, 0),
    BlockLength(Blength), arrayDisplacements(count)
{
  MPI_Aint positiveExtent = 0;
  MPI_Aint negativeExtent = 0;
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
  if (count == 0) {
    lb = baseType->getLB();
  } else {
    lb = baseType->getLB() + *std::min_element(&arrayDisplacements[0], &arrayDisplacements[0] + count)*baseExtent;
  }
  ub = lb + extent;

  trueExtent = extent;
  trueLB = lb;

  /* set iscontig */
  if (extent != size || count == 0) {
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
{}

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
  CkDDT_DataType::pupType(p, ddt);
  p|BlockLength;
  p|arrayDisplacements;
  if (p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int
CkDDT_Indexed_Block::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count+2;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_INDEXED_BLOCK;
  return MPI_SUCCESS;
}

int
CkDDT_Indexed_Block::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = BlockLength;
  for(int z=0;z<i[0];z++) {
    i[z+2] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

CkDDT_HIndexed_Block::CkDDT_HIndexed_Block(int count, int Blength, const MPI_Aint *ArrDisp, int index,
  CkDDT_DataType *type)     : CkDDT_Indexed_Block(count, Blength,ArrDisp,index,type)
{
  MPI_Aint positiveExtent = 0;
  MPI_Aint negativeExtent = 0;
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
  if (count == 0) {
    lb = baseType->getLB();
  } else {
    lb = baseType->getLB() + *std::min_element(&arrayDisplacements[0], &arrayDisplacements[0] + count);
  }
  ub = lb + extent;

  trueExtent = extent;
  trueLB = lb;

  /* set iscontig */
  if (extent != size || count == 0) {
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
{}

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
  CkDDT_Indexed_Block::pupType(p, ddt);
}

int
CkDDT_HIndexed_Block::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = 2;
  *na = count;
  *nd = 1;
  *combiner = CkDDT_COMBINER_HINDEXED_BLOCK;
  return MPI_SUCCESS;
}

int
CkDDT_HIndexed_Block::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  i[1] = BlockLength;
  for(int z=0;z<i[0];z++) {
    a[z] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

CkDDT_Struct::CkDDT_Struct(int nCount, const int* arrBlock,
                       const MPI_Aint* arrDisp, const int *bindex, CkDDT_DataType** arrBase)
    : CkDDT_DataType(CkDDT_STRUCT, 0, 0, nCount, 0,
    0, 0, 0, 0, NULL, 0, 0, 0, 0),
    arrayBlockLength(nCount), arrayDisplacements(nCount),
    index(nCount), arrayDataType(nCount)
{
  int saveExtent = 0;
  for (int i=0; i<count; i++) {
      arrayBlockLength[i] = arrBlock[i];
      arrayDisplacements[i] = arrDisp[i];
      arrayDataType[i] =  arrBase[i];
      arrayDataType[i]->incRefCount();
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
      MPI_Aint xlb = arrayDataType[i]->getLB() + arrayDisplacements[i];
      MPI_Aint xub = arrayDisplacements[i] + arrayBlockLength[i]*arrayDataType[i]->getUB();
      if (arrayDataType[i]->getType() == MPI_LB) {
          if (!explicit_lb) lb = xlb;
          explicit_lb = true;
          if (xlb < lb) lb = xlb;
      } else if(arrayDataType[i]->getType() == MPI_UB) {
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

  trueLB = -1;
  trueExtent = 0;
  for (int i=0; i<count; i++) {
    if (!(arrayDataType[i]->getType() == MPI_LB || arrayDataType[i]->getType() == MPI_UB)) {
      if (trueLB > arrayDisplacements[i] || trueLB == -1) {
        trueLB = arrayDisplacements[i];
      }
    }
    trueExtent += arrayDataType[i]->getTrueExtent() * arrBlock[i];
  }

  /* set iscontig */
  if (extent != size || count == 0) {
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
  CkDDT_DataType::pupType(p, ddt);
  p|arrayBlockLength;
  p|arrayDisplacements;
  p|index;

  if (p.isUnpacking()) {
    arrayDataType.resize(count);
    for(int i=0 ; i < count; i++)
      arrayDataType[i] = ddt->getType(index[i]);
  }
}

int
CkDDT_Struct::getEnvelope(int *ni, int *na, int *nd, int *combiner) const
{
  *ni = count+1;
  *na = count;
  *nd = count;
  *combiner = CkDDT_COMBINER_STRUCT;
  return MPI_SUCCESS;
}

int
CkDDT_Struct::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const
{
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
    d[z] = index[z];
  }
  return MPI_SUCCESS;
}

const vector<int> &
CkDDT_Struct::getBaseIndices() const
{
  return index;
}

const vector<CkDDT_DataType*> &
CkDDT_Struct::getBaseTypes() const
{
  return arrayDataType;
}

