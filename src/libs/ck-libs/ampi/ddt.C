#include "ddt.h"
#include <algorithm>
#include <limits>

void
CkDDT::pup(PUP::er &p) noexcept
{
  p|types;
  if (p.isUnpacking()) {
    userTypeTable.resize(types.size(), nullptr);
    for (int i=0; i<types.size(); i++) {
      switch (types[i]) {
        case MPI_DATATYPE_NULL:
          break;
        case CkDDT_CONTIGUOUS:
          userTypeTable[i] = new CkDDT_Contiguous;
          break;
        case CkDDT_VECTOR:
          userTypeTable[i] = new CkDDT_Vector;
          break;
        case CkDDT_HVECTOR:
          userTypeTable[i] = new CkDDT_HVector;
          break;
        case CkDDT_INDEXED_BLOCK:
          userTypeTable[i] = new CkDDT_Indexed_Block;
          break;
        case CkDDT_HINDEXED_BLOCK:
          userTypeTable[i] = new CkDDT_HIndexed_Block;
          break;
        case CkDDT_INDEXED:
          userTypeTable[i] = new CkDDT_Indexed;
          break;
        case CkDDT_HINDEXED:
          userTypeTable[i] = new CkDDT_HIndexed;
          break;
        case CkDDT_STRUCT:
          userTypeTable[i] = new CkDDT_Struct;
          break;
        default: // predefined type
          userTypeTable[i] = new CkDDT_DataType;
          break;
      }
    }
  }

  for (int i=0; i<types.size(); i++) {
    if (types[i] != MPI_DATATYPE_NULL) {
      userTypeTable[i]->pupType(p, this);
    }
  }
}

void
CkDDT::freeType(int index) noexcept
{
  CkAssert(types.size() == userTypeTable.size());
  if (index > AMPI_MAX_PREDEFINED_TYPE) {
    int idx = index - AMPI_MAX_PREDEFINED_TYPE - 1;
    // Decrement the ref count and free the type if there are no references to it.
    if (userTypeTable[idx]->decRefCount() == 0) {
      // Remove a reference from this type's base type(s).
      if (userTypeTable[idx]->getType() == CkDDT_STRUCT) {
        int count = userTypeTable[idx]->getCount();
        std::vector<int> &baseIndices = static_cast<CkDDT_Struct &>(*userTypeTable[idx]).getBaseIndices();
        for (int i=0; i<count; i++) {
          freeType(baseIndices[i]);
        }
      }
      else {
        freeType(userTypeTable[idx]->getBaseIndex());
      }

      // Free non-primitive type
      delete userTypeTable[idx];
      userTypeTable[idx] = nullptr;
      types[idx] = MPI_DATATYPE_NULL;
      // Free all NULL types from back of userTypeTable
      while (!userTypeTable.empty() && userTypeTable.back() == nullptr) {
        userTypeTable.pop_back();
        CkAssert(types.back() == MPI_DATATYPE_NULL);
        types.pop_back();
      }
    }
  }
  CkAssert(types.size() == userTypeTable.size());
}

CkDDT::~CkDDT() noexcept
{
  for (int i=0; i<userTypeTable.size(); i++) {
    if (userTypeTable[i] != nullptr) {
      delete userTypeTable[i];
    }
  }
}

int
CkDDT::insertType(CkDDT_DataType* ptr, int type) noexcept
{
  // Search thru non-predefined types for a free one first:
  CkAssert(types.size() == userTypeTable.size());
  for (int i=0; i<types.size(); i++) {
    if (types[i] == MPI_DATATYPE_NULL) {
      types[i] = type;
      userTypeTable[i] = ptr;
      return AMPI_MAX_PREDEFINED_TYPE + 1 + i;
    }
  }
  types.push_back(type);
  userTypeTable.push_back(ptr);
  return AMPI_MAX_PREDEFINED_TYPE + types.size();
}

void
CkDDT::createDup(int nIndexOld, int *nIndexNew) noexcept
{
  CkDDT_DataType *dttype = getType(nIndexOld);
  CkDDT_DataType *type;
  int typeClass;

  switch (dttype->getType()) {
    case CkDDT_CONTIGUOUS:
      type = new CkDDT_Contiguous(static_cast<CkDDT_Contiguous&> (*dttype));
      typeClass = CkDDT_CONTIGUOUS;
      break;
    case CkDDT_VECTOR:
      type = new CkDDT_Vector(static_cast<CkDDT_Vector&> (*dttype));
      typeClass = CkDDT_VECTOR;
      break;
    case CkDDT_HVECTOR:
      type = new CkDDT_HVector(static_cast<CkDDT_HVector&> (*dttype));
      typeClass = CkDDT_HVECTOR;
      break;
    case CkDDT_INDEXED_BLOCK:
      type = new CkDDT_Indexed_Block(static_cast<CkDDT_Indexed_Block&> (*dttype));
      typeClass = CkDDT_INDEXED_BLOCK;
      break;
    case CkDDT_HINDEXED_BLOCK:
      type = new CkDDT_HIndexed_Block(static_cast<CkDDT_HIndexed_Block&> (*dttype));
      typeClass = CkDDT_HINDEXED_BLOCK;
      break;
    case CkDDT_INDEXED:
      type = new CkDDT_Indexed(static_cast<CkDDT_Indexed&> (*dttype));
      typeClass = CkDDT_INDEXED;
      break;
    case CkDDT_HINDEXED:
      type = new CkDDT_HIndexed(static_cast<CkDDT_HIndexed&> (*dttype));
      typeClass = CkDDT_HINDEXED;
      break;
    case CkDDT_STRUCT:
      type = new CkDDT_Struct(static_cast<CkDDT_Struct&> (*dttype));
      typeClass = CkDDT_STRUCT;
      break;
    default:
      type = new CkDDT_DataType(*dttype);
      typeClass = dttype->getType();
      break;
  }

  *nIndexNew = insertType(type, typeClass);
}

int
CkDDT::getEnvelope(int nIndex, int *ni, int *na, int *nd, int *combiner) const noexcept
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getEnvelope(ni, na, nd, combiner);
}

int
CkDDT::getContents(int nIndex, int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) noexcept
{
  CkDDT_DataType* dttype = getType(nIndex);
  int ret = dttype->getContents(ni, na, nd, i, a, d);
  if (dttype->getType() == CkDDT_STRUCT) {
    int count = dttype->getCount();
    std::vector<CkDDT_DataType *> &baseTypes = static_cast<CkDDT_Struct &>(*dttype).getBaseTypes();
    for (int i=0; i<count; i++) {
      baseTypes[i]->incRefCount();
    }
  }
  else {
    dttype->getBaseType()->incRefCount();
  }
  return ret;
}

void
CkDDT::createResized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newType) noexcept
{
  CkDDT_DataType *dttype = getType(oldtype);
  CkDDT_DataType *type;
  int typeClass;

  switch (dttype->getType()) {
    case CkDDT_CONTIGUOUS:
      type = new CkDDT_Contiguous(static_cast<CkDDT_Contiguous &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_CONTIGUOUS;
      break;
    case CkDDT_VECTOR:
      type = new CkDDT_Vector(static_cast<CkDDT_Vector &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_VECTOR;
      break;
    case CkDDT_HVECTOR:
      type = new CkDDT_HVector(static_cast<CkDDT_HVector &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_HVECTOR;
      break;
    case CkDDT_INDEXED_BLOCK:
      type = new CkDDT_Indexed_Block(static_cast<CkDDT_Indexed_Block &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_INDEXED_BLOCK;
      break;
    case CkDDT_HINDEXED_BLOCK:
      type = new CkDDT_HIndexed_Block(static_cast<CkDDT_HIndexed_Block &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_HINDEXED_BLOCK;
      break;
    case CkDDT_INDEXED:
      type = new CkDDT_Indexed(static_cast<CkDDT_Indexed &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_INDEXED;
      break;
    case CkDDT_HINDEXED:
      type = new CkDDT_HIndexed(static_cast<CkDDT_HIndexed &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_HINDEXED;
      break;
    case CkDDT_STRUCT:
      type = new CkDDT_Struct(static_cast<CkDDT_Struct &>(*dttype));
      type->setSize(lb, extent);
      typeClass = CkDDT_STRUCT;
      break;
    default:
      type = new CkDDT_DataType(*dttype, lb, extent);
      typeClass = dttype->getType();
      break;
  }

  *newType = insertType(type, typeClass);
}

void
CkDDT::newContiguous(int count, MPI_Datatype oldType, MPI_Datatype *newType) noexcept
{
  CkDDT_DataType *type = new CkDDT_Contiguous(count, oldType, getType(oldType));
  *newType = insertType(type, CkDDT_CONTIGUOUS);
}

void
CkDDT::newVector(int count, int blocklength, int stride,
                 MPI_Datatype oldType, MPI_Datatype* newType) noexcept
{
  CkDDT_DataType* type = new CkDDT_Vector(count, blocklength, stride, oldType, getType(oldType));
  *newType = insertType(type, CkDDT_VECTOR);
}

void
CkDDT::newHVector(int count, int blocklength, int stride,
                  MPI_Datatype oldtype, MPI_Datatype* newType) noexcept
{
  CkDDT_DataType* type = new CkDDT_HVector(count, blocklength, stride, oldtype, getType(oldtype));
  *newType = insertType(type, CkDDT_HVECTOR);
}

void
CkDDT::newIndexedBlock(int count, int Blocklength, const int *arrDisp, MPI_Datatype oldtypeIdx,
                       MPI_Datatype *newType) noexcept
{
  // Convert arrDisp from an array of int's to an array of MPI_Aint's. This is needed because
  // MPI_Type_create_indexed_block takes ints and MPI_Type_create_hindexed_block takes MPI_Aint's
  // and we use HIndexed_Block to represent both of those datatypes internally.
  CkDDT_DataType* oldtype = getType(oldtypeIdx);
  std::vector<MPI_Aint> arrDispBytes(count);
  for (int i=0; i<count; i++) {
    arrDispBytes[i] = static_cast<MPI_Aint>(arrDisp[i] * oldtype->getExtent());
  }
  CkDDT_DataType *type = new CkDDT_Indexed_Block(count, Blocklength, arrDispBytes.data(),
                                                 arrDisp, oldtypeIdx, oldtype);
  *newType = insertType(type, CkDDT_INDEXED_BLOCK);
}

void
CkDDT::newHIndexedBlock(int count, int Blocklength, const MPI_Aint *arrDisp, MPI_Datatype oldtype,
                        MPI_Datatype *newType) noexcept
{
  CkDDT_DataType *type = new CkDDT_HIndexed_Block(count, Blocklength, arrDisp,
                                                  oldtype, getType(oldtype));
  *newType = insertType(type, CkDDT_HINDEXED_BLOCK);
}

void
CkDDT::newIndexed(int count, const int* arrbLength, MPI_Aint* arrDisp,
                  MPI_Datatype oldtypeIdx, MPI_Datatype* newType) noexcept
{
  CkDDT_DataType* oldtype = getType(oldtypeIdx);
  std::vector<MPI_Aint> dispBytesArr(count);
  for (int i=0; i<count; i++) {
    dispBytesArr[i] = arrDisp[i] * oldtype->getExtent();
  }
  CkDDT_DataType* type = new CkDDT_Indexed(count, arrbLength, dispBytesArr.data(), arrDisp,
                                           oldtypeIdx, oldtype);
  *newType = insertType(type, CkDDT_INDEXED);
}

void
CkDDT::newHIndexed(int count, const int* arrbLength, const MPI_Aint* arrDisp,
                   MPI_Datatype oldtype, MPI_Datatype* newType) noexcept
{
  CkDDT_DataType* type = new CkDDT_HIndexed(count, arrbLength, arrDisp, oldtype, getType(oldtype));
  *newType = insertType(type, CkDDT_HINDEXED);
}

void
CkDDT::newStruct(int count, const int* arrbLength, const MPI_Aint* arrDisp,
                 const MPI_Datatype *oldtype, MPI_Datatype* newType) noexcept
{
  std::vector<CkDDT_DataType *> olddatatypes(count);
  for(int i=0;i<count;i++){
    olddatatypes[i] = getType(oldtype[i]);
  }
  CkDDT_DataType* type = new CkDDT_Struct(count, arrbLength, arrDisp, oldtype, olddatatypes.data());
  *newType = insertType(type, CkDDT_STRUCT);
}

std::string
CkDDT_DataType::getConfig() const noexcept
{
  std::string res(getName());
  res+=": ";
  res+=getTypeMap();
  res+=" lb=" + std::to_string(lb);
  res+=" ub=" + std::to_string(ub);
  res+=" extent=" + std::to_string(extent);
  res+=" trueExtent=" + std::to_string(trueExtent);
  res+=" trueLB=" + std::to_string(trueLB);
  res+=" size=" + std::to_string(size);
  res+=" iscontig=" + std::to_string(iscontig);

  return res;
}

std::string
CkDDT_DataType::getTypeMap() const noexcept
{
  std::string res("{");
  for(int i=0; i<count; i++) {
    res+=("(");
    res+=(baseType ? baseType->getName() : getName());
    res+=(",");
    res+=(std::to_string(lb));
    res+=(")");
    if (i!=count -1)
      res+=(",");
  }
  res+=("}");
  return res;
}

std::string
CkDDT_Contiguous::getTypeMap() const noexcept
{
  std::string res("{");
  for(int i=0; i<count; i++) {
    res+=("(");
    res+=("TYPE_"+std::to_string(baseIndex));
    res+=(",");
    res+=(std::to_string(i*baseType->getExtent()));
    res+=(")");
    if (i!=count -1)
      res+=(",");
  }
  res+=("}");
  return res;
}

std::string
CkDDT_Vector::getTypeMap() const noexcept
{
  std::string res("{");
  int disp = 0;
  for(int i=0; i<count; i++) {
    for(int j=0; j<blockLength; j++) {
      res+=("(");
      res+=("TYPE_"+std::to_string(baseIndex));
      res+=(",");
      res+=(std::to_string(j*baseType->getExtent()+disp));
      res+=(")");
      if (j!=blockLength -1)
        res+=(",");
    }
    disp += strideLength*baseType->getExtent();
    res+=(";");
  }
  res+=("}");
  return res;
}

std::string
CkDDT_HVector::getTypeMap() const noexcept
{
  std::string res("{");
  int disp = 0;
  for(int i=0; i<count; i++) {
    for(int j=0; j<blockLength; j++) {
      res+=("(");
      res+=("TYPE_"+std::to_string(baseIndex));
      res+=(",");
      res+=(std::to_string(j*baseType->getExtent()+disp));
      res+=(")");
      if (j!=blockLength -1)
        res+=(",");
    }
    disp += strideLength;
  }
  res+=("}");
  return res;
}

std::string
CkDDT_HIndexed::getTypeMap() const noexcept
{
  std::string res("{");
  for(int i=0; i<count; i++) {
    res+=("(");
    res+=("TYPE_"+std::to_string(baseIndex));
    res+=(",");
    res+=(std::to_string(arrayDisplacements[i]));
    res+=(")");
    if (i!=count -1)
      res+=(",");
  }
  res+=("}");
  return res;
}

std::string
CkDDT_HIndexed_Block::getTypeMap() const noexcept
{
  std::string res("{");
  for(int i=0; i<count; i++) {
    res+=("(");
    res+=("TYPE_"+std::to_string(baseIndex));
    res+=(",");
    res+=(std::to_string(arrayDisplacements[i]));
    res+=(")");
    if (i!=count -1)
      res+=(",");
  }
  res+=("}");
  return res;
}

std::string
CkDDT_Struct::getTypeMap() const noexcept
{
  std::string res("{");
  for(int i=0; i<count; i++) {
    res+=("(");
    res+=(arrayDataType[i]->getName());
    res+=(",");
    res+=(std::to_string(arrayDisplacements[i]));
    res+=(")");
    if (i!=count -1)
      res+=(",");
  }
  res+=("}");
  return res;
}

CkDDT_DataType::CkDDT_DataType(int type) noexcept : datatype(type)
{
  count = 1;
  switch (datatype) {
    case MPI_DOUBLE:
      size         = sizeof(double);
      numElements  = 1;
      name = "MPI_DOUBLE";
      break;
    case MPI_INT:
      size         = sizeof(signed int);
      numElements  = 1;
      name = "MPI_INT";
      break;
    case MPI_FLOAT:
      size         = sizeof(float);
      numElements  = 1;
      name = "MPI_FLOAT";
      break;
    case MPI_CHAR:
      size         = sizeof(char);
      numElements  = 1;
      name = "MPI_CHAR";
      break;
    case MPI_BYTE:
      size         = 1 ;
      numElements  = 1;
      name = "MPI_BYTE";
      break;
    case MPI_PACKED:
      size         = 1 ;
      numElements  = 1;
      name = "MPI_PACKED";
      break;
    case MPI_C_BOOL:
      /* Should be C99 _Bool instead of C++ bool, but MSVC doesn't support that */
      size         = sizeof(bool) ;
      numElements  = 1;
      name = "MPI_C_BOOL";
      break;
    case MPI_LOGICAL:
      size         =  sizeof(int) ;
      numElements  = 1;
      name = "MPI_LOGICAL";
      break;
    case MPI_SHORT:
      size         = sizeof(signed short int);
      numElements  = 1;
      name = "MPI_SHORT";
      break;
    case MPI_LONG:
      size         = sizeof(signed long);
      numElements  = 1;
      name = "MPI_LONG";
      break;
    case MPI_UNSIGNED_CHAR:
      size         = sizeof(unsigned char);
      numElements  = 1;
      name = "MPI_UNSIGNED_CHAR";
      break;
    case MPI_UNSIGNED_SHORT:
      size         = sizeof(unsigned short);
      numElements  = 1;
      name = "MPI_UNSIGNED_SHORT";
      break;
    case MPI_UNSIGNED:
      size         = sizeof(unsigned);
      numElements  = 1;
      name = "MPI_UNSIGNED";
      break;
    case MPI_UNSIGNED_LONG:
      size         = sizeof(unsigned long);
      numElements  = 1;
      name = "MPI_UNSIGNED_LONG";
      break;
    case MPI_LONG_DOUBLE:
      size         = sizeof(long double);
      numElements  = 1;
      name = "MPI_LONG_DOUBLE";
      break;
    case MPI_SIGNED_CHAR:
      size         = sizeof(signed char);
      numElements  = 1;
      name = "MPI_SIGNED_CHAR";
      break;
    case MPI_UNSIGNED_LONG_LONG:
      size         = sizeof(unsigned long long);
      numElements  = 1;
      name = "MPI_UNSIGNED_LONG_LONG";
      break;
    case MPI_WCHAR:
      size         = sizeof(wchar_t);
      numElements  = 1;
      name = "MPI_WCHAR";
      break;
    case MPI_INT8_T:
      size         = sizeof(int8_t);
      numElements  = 1;
      name = "MPI_INT8_T";
      break;
    case MPI_INT16_T:
      size         = sizeof(int16_t);
      numElements  = 1;
      name = "MPI_INT16_T";
      break;
    case MPI_INT32_T:
      size         = sizeof(int32_t);
      numElements  = 1;
      name = "MPI_INT32_T";
      break;
    case MPI_INT64_T:
      size         = sizeof(int64_t);
      numElements  = 1;
      name = "MPI_INT64_T";
      break;
    case MPI_UINT8_T:
      size         = sizeof(uint8_t);
      numElements  = 1;
      name = "MPI_UINT8_T";
      break;
    case MPI_UINT16_T:
      size         = sizeof(uint16_t);
      numElements  = 1;
      name = "MPI_UINT16_T";
      break;
    case MPI_UINT32_T:
      size         = sizeof(uint32_t);
      numElements  = 1;
      name = "MPI_UINT32_T";
      break;
    case MPI_UINT64_T:
      size         = sizeof(uint64_t);
      numElements  = 1;
      name = "MPI_UINT64_T";
      break;
    case MPI_AINT:
      size         = sizeof(MPI_Aint);
      numElements  = 1;
      name = "MPI_AINT";
      break;
    case MPI_LB:
      size         = 0;
      numElements  = 0;
      name = "MPI_LB";
      break;
    case MPI_UB:
      size         = 0;
      numElements  = 0;
      name = "MPI_UB";
      break;
#if CMK_LONG_LONG_DEFINED
    case MPI_LONG_LONG_INT:
      size         = sizeof(signed long long);
      name = "MPI_LONG_LONG_INT";
      break;
#endif
    default:
      CkAbort("CkDDT: Trying to make primitive type with unsupported type");
  }

  trueExtent  = size;
  extent      = size;
  lb          = 0;
  ub          = size;
  trueLB      = 0;
  iscontig    = true;
  baseType    = NULL;
  baseSize    = 0;
  baseExtent  = 0;
  baseIndex   = -1;
  refCount    = 1;

  DDTDEBUG("CkDDT_DataType() %s\n", getConfig().c_str());
}

CkDDT_DataType::CkDDT_DataType(int datatype, int size, MPI_Aint extent, int count, MPI_Aint lb,
                               MPI_Aint ub, bool iscontig, int baseSize, MPI_Aint baseExtent,
                               CkDDT_DataType* baseType, int numElements, int baseIndex,
                               MPI_Aint trueExtent, MPI_Aint trueLB) noexcept :
    iscontig(iscontig), isAbsolute(false), size(size), count(count), datatype(datatype),
    refCount(1), baseSize(baseSize), baseIndex(baseIndex), numElements(numElements),
    extent(extent), ub(ub), lb(lb), trueExtent(trueExtent), trueLB(trueLB),
    baseExtent(baseExtent), baseType(baseType)
{
  if (baseType) {
    baseType->incRefCount();
  }
}

CkDDT_DataType::CkDDT_DataType(const CkDDT_DataType &obj, MPI_Aint _lb/*=0*/, MPI_Aint _extent/*=0*/) noexcept :
  iscontig(obj.iscontig)
  ,isAbsolute(obj.isAbsolute)
  ,size(obj.size)
  ,count(obj.count)
  ,datatype(obj.datatype)
  ,refCount(1)
  ,baseSize(obj.baseSize)
  ,baseIndex(obj.baseIndex)
  ,numElements(obj.numElements)
  ,extent(obj.extent)
  ,ub(obj.ub)
  ,lb(obj.lb)
  ,trueExtent(obj.trueExtent)
  ,trueLB(obj.trueLB)
  ,baseExtent(obj.baseExtent)
  ,baseType(obj.baseType)
  ,name(obj.name)
{
  if (baseType) {
    baseType->incRefCount();
  }
  if ((_lb != 0 || _lb != obj.lb) ||
      (_extent != 0 || _extent != obj.extent)) {
    setSize(_lb, _extent);
  }
}

void
CkDDT_DataType::setSize(MPI_Aint _lb, MPI_Aint _extent) noexcept
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

int
CkDDT_DataType::getNumBasicElements(int bytes) const noexcept
{
  int extent = getSize();
  if (extent == 0) {
    return 0;
  }
  else {
    return (bytes/extent) * getNumElements();
  }
}

void
CkDDT_DataType::pupType(PUP::er  &p, CkDDT* ddt) noexcept
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
  p|attributes;
  p|name;
  if (p.isUnpacking()) {
    baseType = NULL;
  }
}

int
CkDDT_DataType::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = 0;
  *na = 0;
  *nd = 0;
  *combiner = MPI_COMBINER_NAMED;
  return MPI_SUCCESS;
}

int
CkDDT_DataType::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  return MPI_ERR_TYPE;
}

CkDDT_Contiguous::CkDDT_Contiguous(int nCount, int bindex, CkDDT_DataType* oldType) noexcept
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

  if (baseType->getLB() > baseType->getUB()) {
    lb = baseType->getLB() + (baseExtent*(count-1));
    ub = baseType->getUB();
    trueLB = baseType->getTrueLB() + (baseExtent*(count-1));
    trueExtent = baseType->getTrueLB() + baseType->getTrueExtent() - ((count-1)*baseExtent);
  }
  else {
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
  DDTDEBUG("CkDDT_Contiguous() %s\n", getConfig().c_str());
}

size_t
CkDDT_Contiguous::serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_Contiguous::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);
  size_t bytesCopied = 0;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    for (; num>0; num--) {
      int bytesProcessed = baseType->serialize(userdata, buffer, count, msgLength, dir);
      bytesCopied += bytesProcessed;
      msgLength -= bytesProcessed;
      buffer += size;
      userdata += extent;
      if (msgLength == 0) {
        return bytesCopied;
      }
    }
  }
  return bytesCopied;
}

void
CkDDT_Contiguous::pupType(PUP::er &p, CkDDT *ddt) noexcept
{
  CkDDT_DataType::pupType(p, ddt);
  if (p.isUnpacking()) {
    baseType = ddt->getType(baseIndex);
  }
}

int
CkDDT_Contiguous::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = 1;
  *na = 0;
  *nd = 1;
  *combiner = MPI_COMBINER_CONTIGUOUS;
  return MPI_SUCCESS;
}

int
CkDDT_Contiguous::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_Contiguous::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_Vector::CkDDT_Vector(int nCount, int blength, int stride, int bindex, CkDDT_DataType* oldType) noexcept
{
  datatype = CkDDT_VECTOR;
  count = nCount;
  blockLength = blength;
  strideLength = stride;
  baseIndex = bindex;
  baseType = oldType;
  baseSize = baseType->getSize();
  baseExtent = baseType->getExtent();
  refCount = 1;
  baseType->incRefCount();
  numElements = count * baseType->getNumElements();
  size = count * blockLength * baseSize;

  int absBaseExtent = std::abs(baseExtent);
  int absStrideLength = std::abs(strideLength);

  if (baseType->getLB() > baseType->getUB()) {
    if (strideLength > 0) {
      // Negative Extent with positive stride
      lb = baseType->getUB() + (((strideLength*count)-2-(absStrideLength-blockLength))*baseExtent);
      ub = baseType->getUB();
      trueLB = lb - baseType->getLB();
    }
    else {
      // Negative extent and stride
      lb = baseType->getLB() + ((blockLength-1)*baseExtent);
      ub = baseType->getUB() + (strideLength*(count-1)*baseExtent);
      trueLB = baseType->getLB() - baseType->getUB() + (blockLength*baseExtent);
    }
  }
  else {
    if (strideLength > 0) {
      // Positive extent and stride
      lb = baseType->getLB();
      ub = lb + (count*blockLength + ((strideLength-blockLength)*(count-1))) * baseExtent;
      trueLB = baseType->getTrueLB();
    }
    else {
      // Negative stride and positive extent
      lb = baseType->getLB() + (strideLength*baseExtent*(count-1));
      ub = lb + blockLength*baseExtent + absStrideLength*(count-1)*baseExtent;
      trueLB = baseType->getTrueLB() + ((count-1) * strideLength * baseType->getExtent());
    }
  }

  extent = ub - lb;

  if (absStrideLength < blockLength) {
    trueExtent = ((count-1) * stride * absBaseExtent) + (blockLength * absBaseExtent) -
                 (absBaseExtent - baseType->getTrueExtent());
  }
  else {
    trueExtent = (((absStrideLength*count)-(absStrideLength-blockLength))*absBaseExtent) -
                 (absBaseExtent - baseType->getTrueExtent());
  }

  if (extent != size || count == 0) {
    iscontig = false;
  }
  else {
    if (count==1 || (strideLength==1 && blockLength==1)) {
      iscontig = baseType->isContig();
    }
    else {
      iscontig = false;
    }
  }
  DDTDEBUG("CkDDT_Vector() %s\n", getConfig().c_str());
}

size_t
CkDDT_Vector::serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_Vector::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);

  size_t bytesCopied = 0;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)blockLength * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    for (; num>0; num--) {
      char* saveUserdata = userdata;
      for (int i=0; i<count; i++) {
        int bytesProcessed = baseType->serialize(userdata, buffer, blockLength, msgLength, dir);
        bytesCopied += bytesProcessed;
        msgLength -= bytesProcessed;
        buffer += (blockLength*baseSize);
        userdata += (strideLength*baseExtent);
        if (msgLength == 0) {
          return bytesCopied;
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_Vector::pupType(PUP::er &p, CkDDT* ddt) noexcept
{
  CkDDT_DataType::pupType(p, ddt);
  p|blockLength;
  p|strideLength;
  if (p.isUnpacking()) {
    baseType = ddt->getType(baseIndex);
  }
}

int
CkDDT_Vector::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = 3;
  *na = 0;
  *nd = 1;
  *combiner = MPI_COMBINER_VECTOR;
  return MPI_SUCCESS;
}

int
CkDDT_Vector::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  i[1] = blockLength;
  i[2] = strideLength;
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_Vector::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_HVector::CkDDT_HVector(int nCount, int blength, int stride,  int bindex,
                         CkDDT_DataType* oldType) noexcept
{
  datatype = CkDDT_HVECTOR;
  count = nCount;
  blockLength = blength;
  strideLength = stride;
  baseIndex = bindex;
  baseType = oldType;
  baseSize = baseType->getSize();
  baseExtent = baseType->getExtent();
  refCount = 1;
  baseType->incRefCount();
  numElements = count * baseType->getNumElements();
  size = count *  blockLength * baseSize;

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
    if (count==1 || (strideLength==1 && blockLength==1)) {
      iscontig = baseType->isContig();
    }
    else {
      iscontig = false;
    }
  }
  DDTDEBUG("CkDDT_HVector() %s\n", getConfig().c_str());
}

size_t
CkDDT_HVector::serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_HVector::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);

  size_t bytesCopied = 0;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)blockLength * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    for (; num>0; num--) {
      char* saveUserdata = userdata;
      for (int i=0; i<count; i++) {
        int bytesProcessed = baseType->serialize(userdata, buffer, blockLength, msgLength, dir);
        bytesCopied += bytesProcessed;
        msgLength -= bytesProcessed;
        buffer += (blockLength*baseSize);
        userdata += strideLength;
        if (msgLength == 0) {
          return bytesCopied;
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_HVector::pupType(PUP::er &p, CkDDT* ddt) noexcept
{
  CkDDT_Vector::pupType(p, ddt);
}

int
CkDDT_HVector::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = 2;
  *na = 1;
  *nd = 1;
  *combiner = MPI_COMBINER_HVECTOR;
  return MPI_SUCCESS;
}

int
CkDDT_HVector::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  i[1] = blockLength;
  a[0] = strideLength;
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_HVector::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_Indexed_Block::CkDDT_Indexed_Block(int count, int Blength, const MPI_Aint *arrBytesDisp,
                                         const int *ArrDisp, int index, CkDDT_DataType *type) noexcept
  : CkDDT_HIndexed_Block(count, Blength, arrBytesDisp, index, type)
{
  for (int i=0; i<count; i++) {
    arrayDisplacements[i] = static_cast<MPI_Aint>(ArrDisp[i]);
  }
  datatype = CkDDT_INDEXED_BLOCK;
}

size_t
CkDDT_Indexed_Block::serialize(char *userdata, char *buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_Indexed_Block::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);

  size_t bytesCopied = 0;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)blockLength * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    for (; num>0; num--) {
      char* saveUserdata = userdata;
      for (int i=0; i<count; i++) {
        userdata = saveUserdata + baseExtent * arrayDisplacements[i];
        for (int j=0; j<blockLength ; j++) {
          int bytesProcessed = baseType->serialize(userdata, buffer, 1, msgLength, dir);
          bytesCopied += bytesProcessed;
          msgLength -= bytesProcessed;
          buffer += baseSize;
          userdata += baseExtent;
          if (msgLength == 0) {
            return bytesCopied;
          }
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_Indexed_Block::pupType(PUP::er &p, CkDDT *ddt) noexcept
{
  CkDDT_HIndexed_Block::pupType(p, ddt);
}

int
CkDDT_Indexed_Block::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = count+2;
  *na = 0;
  *nd = 1;
  *combiner = MPI_COMBINER_INDEXED_BLOCK;
  return MPI_SUCCESS;
}

int
CkDDT_Indexed_Block::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  i[1] = blockLength;
  for (int z=0; z<i[0]; z++) {
    i[z+2] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_Indexed_Block::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_HIndexed_Block::CkDDT_HIndexed_Block(int count, int Blength, const MPI_Aint *ArrDisp,
                                           int index, CkDDT_DataType *type) noexcept
  : CkDDT_DataType(CkDDT_INDEXED_BLOCK, 0, 0, count, 0, 0, 0, type->getSize(),
                   type->getExtent(), type, count * type->getNumElements(), index, 0, 0),
    blockLength(Blength),
    arrayDisplacements(count)
{
  datatype = CkDDT_HINDEXED_BLOCK;
  bool validElem = false;
  MPI_Aint positiveExtent = 0;
  MPI_Aint negativeExtent = 0;
  for (int i=0; i<count; i++) {
    arrayDisplacements[i] = ArrDisp[i];
    size += Blength * baseSize;
    if (Blength > 0) {
      if (!validElem) {
        negativeExtent = arrayDisplacements[i];
        positiveExtent = arrayDisplacements[i] + (Blength*baseExtent);
        validElem = true;
      }
      negativeExtent = std::min(arrayDisplacements[i], negativeExtent);
      positiveExtent = std::max(arrayDisplacements[i] + (Blength*baseExtent), positiveExtent);
    }
  }
  lb = baseType->getLB() + negativeExtent;
  ub = baseType->getLB() + positiveExtent;
  extent = positiveExtent + (-1)*negativeExtent;

  trueExtent = extent;
  trueLB = lb;

  /* set iscontig */
  if (extent != size || count == 0) {
    iscontig = false;
  }
  else if (count == 1) {
    iscontig = baseType->isContig();
  }
  else if (blockLength != 1) {
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
  DDTDEBUG("CkDDT_{H}Indexed_Block() %s\n", getConfig().c_str());
}

size_t
CkDDT_HIndexed_Block::serialize(char *userdata, char *buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_HIndexed_Block::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);

  size_t bytesCopied = 0;
  if (iscontig) {
    bytesCopied = (size_t)num * (size_t)count * (size_t)blockLength * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    for (; num>0; num--) {
      char* saveUserdata = userdata;
      for (int i=0; i<count; i++) {
        userdata = (isAbsolute) ? (char*)arrayDisplacements[i] : saveUserdata+arrayDisplacements[i];
        for (int j=0; j<blockLength ; j++) {
          int bytesProcessed = baseType->serialize(userdata, buffer, 1, msgLength, dir);
          bytesCopied += bytesProcessed;
          msgLength -= bytesProcessed;
          buffer += baseSize;
          userdata += baseExtent ;
          if (msgLength == 0) {
            return bytesCopied;
          }
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_HIndexed_Block::pupType(PUP::er &p, CkDDT *ddt) noexcept
{
  CkDDT_DataType::pupType(p, ddt);
  p|blockLength;
  p|arrayDisplacements;
  if (p.isUnpacking()) {
    baseType = ddt->getType(baseIndex);
  }
}

int
CkDDT_HIndexed_Block::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = 2;
  *na = count;
  *nd = 1;
  *combiner = MPI_COMBINER_HINDEXED_BLOCK;
  return MPI_SUCCESS;
}

int
CkDDT_HIndexed_Block::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  i[1] = blockLength;
  for (int z=0; z<i[0]; z++) {
    a[z] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_HIndexed_Block::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_Indexed::CkDDT_Indexed(int nCount, const int* arrBlock, const MPI_Aint* arrBytesDisp,
                             const MPI_Aint* arrDisp, int bindex, CkDDT_DataType* base) noexcept
  : CkDDT_HIndexed(nCount, arrBlock, arrBytesDisp, bindex, base)
{
  for (int i=0; i<count; i++) {
    arrayDisplacements[i] = arrDisp[i];
  }
  datatype = CkDDT_INDEXED;
}

size_t
CkDDT_Indexed::serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_Indexed::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);

  size_t bytesCopied = 0;
  if (iscontig) {
    /* arrayBlockLength is either of size 1 or contains all 1s */
    bytesCopied = (size_t)num * (size_t)count * (size_t)arrayBlockLength[1] * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    char* saveUserdata = userdata;
    for (int iter=0; iter<num; iter++) {
      for (int i=0; i<count; i++) {
        userdata = saveUserdata + (baseExtent * arrayDisplacements[i]) + (iter * extent);
        for (int j=0; j<arrayBlockLength[i]; j++) {
          int bytesProcessed = baseType->serialize(userdata, buffer, 1, msgLength, dir);
          bytesCopied += bytesProcessed;
          msgLength -= bytesProcessed;
          buffer += baseSize;
          userdata += baseExtent;
          if (msgLength == 0) {
            return bytesCopied;
          }
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_Indexed::pupType(PUP::er &p, CkDDT* ddt) noexcept
{
  CkDDT_DataType::pupType(p, ddt);
  p|arrayBlockLength;
  p|arrayDisplacements;
  if (p.isUnpacking()) {
    baseType = ddt->getType(baseIndex);
  }
}

int
CkDDT_Indexed::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = count*2+1;
  *na = 0;
  *nd = 1;
  *combiner = MPI_COMBINER_INDEXED;
  return MPI_SUCCESS;
}

int
CkDDT_Indexed::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  for(int z=0; z<i[0]; z++) {
    i[z+1] = arrayBlockLength[z];
    i[z+i[0]+1] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_Indexed::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_HIndexed::CkDDT_HIndexed(int nCount, const int* arrBlock, const MPI_Aint* arrDisp, int bindex,
                               CkDDT_DataType* base) noexcept
  : CkDDT_DataType(CkDDT_HINDEXED, 0, 0, nCount, 0, 0, 0, base->getSize(), base->getExtent(),
                   base, 0, bindex, 0, 0),
    arrayBlockLength(nCount),
    arrayDisplacements(nCount)
{
  datatype = CkDDT_HINDEXED;
  size = 0;
  if (count == 0) {
    lb = 0;
    ub = 0;
    extent = 0;
  }
  else {
    bool validElem = false;
    MPI_Aint positiveExtent = 0;
    MPI_Aint negativeExtent = 0;
    for (int i=0; i<count; i++) {
      arrayBlockLength[i] = arrBlock[i];
      arrayDisplacements[i] = arrDisp[i];
      size += ( arrBlock[i] * baseSize);
      if (arrayBlockLength[i] > 0) {
        if (!validElem) {
          negativeExtent = arrayDisplacements[i];
          positiveExtent = arrayDisplacements[i] + (arrayBlockLength[i]*baseExtent);
          validElem = true;
        }
        negativeExtent = std::min(arrayDisplacements[i], negativeExtent);
        positiveExtent = std::max(arrayDisplacements[i] + (arrayBlockLength[i]*baseExtent), positiveExtent);
      }
    }
    lb = baseType->getLB() + negativeExtent;
    ub = baseType->getLB() + positiveExtent;
    extent = positiveExtent + (-1)*negativeExtent;
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
  DDTDEBUG("CkDDT_HIndexed() %s\n", getConfig().c_str());
}

size_t
CkDDT_HIndexed::serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  DDTDEBUG("CkDDT_HIndexed::serialize, %s %d objects of type %d (iscontig=%d)\n",
           (dir==PACK)?"packing":"unpacking", num, baseType->getType(), (int)iscontig);

  size_t bytesCopied = 0;
  if (iscontig) {
    /* arrayBlockLength is either of size 1 or contains all 1s */
    bytesCopied = (size_t)num * (size_t)count * (size_t)arrayBlockLength[0] * (size_t)baseSize;
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    for (; num>0; num--) {
      char *saveUserdata = userdata;
      for (int i=0; i<count; i++) {
        userdata = (isAbsolute) ? (char*)arrayDisplacements[i] : saveUserdata+arrayDisplacements[i];
        for (int j=0; j<arrayBlockLength[i]; j++) {
          int bytesProcessed = baseType->serialize(userdata, buffer, 1, msgLength, dir);
          bytesCopied += bytesProcessed;
          msgLength -= bytesProcessed;
          buffer += baseSize;
          userdata += baseExtent;
          if (msgLength == 0) {
            return bytesCopied;
          }
        }
      }
      userdata = saveUserdata + extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_HIndexed::pupType(PUP::er &p, CkDDT* ddt) noexcept
{
  CkDDT_DataType::pupType(p, ddt);
  p|arrayBlockLength;
  p|arrayDisplacements;
  if (p.isUnpacking()) {
    baseType = ddt->getType(baseIndex);
  }
}

int
CkDDT_HIndexed::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = count+1;
  *na = count;
  *nd = 1;
  *combiner = MPI_COMBINER_HINDEXED;
  return MPI_SUCCESS;
}

int
CkDDT_HIndexed::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  for (int z=0; z<i[0]; z++) {
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return MPI_SUCCESS;
}

int
CkDDT_HIndexed::getNumBasicElements(int bytes) const noexcept
{
  return getBaseType()->getNumBasicElements(bytes);
}

CkDDT_Struct::CkDDT_Struct(int nCount, const int* arrBlock, const MPI_Aint* arrDisp,
                           const int *bindex, CkDDT_DataType** arrBase,
                           const char* name/*=nullptr*/) noexcept
  : CkDDT_DataType(CkDDT_STRUCT, 0, 0, nCount, 0, 0, 0, 0, 0, NULL, 0, 0, 0, 0),
    arrayBlockLength(nCount),
    arrayDisplacements(nCount),
    index(nCount),
    arrayDataType(nCount)
{
  if (name != nullptr) setName(name);
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
    }
    else if(arrayDataType[i]->getType() == MPI_UB) {
      if (!explicit_ub) ub = xub;
      explicit_ub = true;
      if (xub > ub) ub = xub;
    }
    else {
      if (!explicit_lb && xlb < lb) lb = xlb;
      if (!explicit_ub && xub > ub) ub = xub;
    }
  }

  extent = ub - lb;
  if (!explicit_ub && (saveExtent != 0) && (extent % saveExtent != 0)) {
    extent += (saveExtent - (extent % saveExtent));
  }

  trueLB = 0;
  MPI_Aint trueUB = 0;
  int trueUBi = 0;
  bool trueLB_set = false;
  bool trueUB_set = false;
  bool empty = true;
  for (int i=0; i<count; i++) {
    if (!(arrayDataType[i]->getType() == MPI_LB || arrayDataType[i]->getType() == MPI_UB)) {
      if (trueLB > arrayDisplacements[i] || !trueLB_set) {
        trueLB_set = true;
        trueLB = arrayDisplacements[i];
      }
      empty = false;
      if (trueUB < arrayDisplacements[i] || !trueUB_set) {
        trueUBi = i;
        trueUB_set = true;
        trueUB = arrayDisplacements[i];
      }
    }
  }

  if (empty) {
    trueExtent = 0;
  } else {
    trueExtent = trueUB + (arrayDataType[trueUBi]->getExtent()*arrayBlockLength[trueUBi]) - trueLB;
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
  DDTDEBUG("CkDDT_Struct() %s\n", getConfig().c_str());
}

size_t
CkDDT_Struct::serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
{
  size_t bytesCopied = 0;

  if (iscontig) {
    DDTDEBUG("CkDDT_Struct::serialize, %s %d objects (iscontig=%d)\n",
             (dir==PACK)?"packing":"unpacking", num, (int)iscontig);
    /* arrayBlockLength is either of size 1 or contains all 1s */
    for (int i=0; i<count; i++) {
      bytesCopied += (size_t)num * (size_t)arrayBlockLength[0] * (size_t)arrayDataType[i]->getSize();
    }
    serializeContig(userdata, buffer, std::min(bytesCopied, (size_t)msgLength), dir);
  }
  else {
    char* sbuf = userdata;
    char* dbuf = buffer;
    char* absoluteOffset = (isAbsolute) ? 0 : userdata;
    for (; num>0; num--) {
      char *buf = buffer;
      for (int i=0; i<count; i++) {
        int saveSize = arrayDataType[i]->getSize();
        int saveExtent = arrayDataType[i]->getExtent();
        for (int j=0; j<arrayBlockLength[i]; j++) {
          DDTDEBUG("CkDDT_Struct::serialize %s block of type %d (size %d) from offset %d to offset %d\n",
                   (dir==PACK)?"packing":"unpacking", arrayDataType[i]->getType(),
                   saveSize, absoluteOffset + (j*saveExtent) + arrayDisplacements[i]-sbuf, buffer-dbuf);
          if (msgLength == 0) {
            return bytesCopied;
          }
          int bytesProcessed = arrayDataType[i]->serialize(
                         arrayDisplacements[i] + j*saveExtent + absoluteOffset,
                         buffer,
                         1,
                         msgLength,
                         dir);
          bytesCopied += bytesProcessed;
          msgLength -= bytesProcessed;
          buffer += saveSize;
        }
      }
      buffer = buf + size;
      absoluteOffset += extent;
    }
  }
  return bytesCopied;
}

void
CkDDT_Struct::pupType(PUP::er &p, CkDDT* ddt) noexcept
{
  CkDDT_DataType::pupType(p, ddt);
  p|arrayBlockLength;
  p|arrayDisplacements;
  p|index;
  if (p.isUnpacking()) {
    arrayDataType.resize(count);
    for (int i=0; i<count; i++) {
      arrayDataType[i] = ddt->getType(index[i]);
    }
  }
}

int
CkDDT_Struct::getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept
{
  *ni = count+1;
  *na = count;
  *nd = count;
  *combiner = MPI_COMBINER_STRUCT;
  return MPI_SUCCESS;
}

int
CkDDT_Struct::getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept
{
  i[0] = count;
  for (int z=0; z<i[0]; z++) {
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
    d[z] = index[z];
  }
  return MPI_SUCCESS;
}

int
CkDDT_Struct::getNumBasicElements(int bytes) const noexcept
{
  int size = getSize();
  if (size == 0) {
    return 0;
  }

  int rem = bytes % size;
  const std::vector<CkDDT_DataType *> &types = getBaseTypes();
  int basicTypes = 0;
  for (int i=0; i<types.size(); i++) {
    basicTypes += types[i]->getNumBasicElements(types[i]->getSize());
  }

  int count = (bytes / size) * basicTypes;
  if (rem == 0) {
    return count;
  }

  for (int i=0; i<types.size(); i++) {
    int type_size = types[i]->getSize();
    if ((type_size * arrayBlockLength[i]) > rem) {
      return count + types[i]->getNumBasicElements(rem);
    }
    else {
      count += types[i]->getNumBasicElements(type_size) * arrayBlockLength[i];
      rem -= type_size * arrayBlockLength[i];
    }
    if (rem == 0) {
      return count;
    }
  }
  return count;
}

