#include "ddt.h"
#include <algorithm>
#include <limits>

using std::numeric_limits;

#define DDTDEBUG /* CmiPrintf */

CkDDT_DataType*
CkDDT::getType(int nIndex)
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


int
CkDDT::isContig(int nIndex)
{
  return getType(nIndex)->isContig();
}

int
CkDDT::getSize(int nIndex, int count)
{
  CkDDT_DataType* dttype = getType(nIndex);
  return count*dttype->getSize();
}

int
CkDDT::getExtent(int nIndex)
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getExtent();
}

int
CkDDT::getLB(int nIndex)
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getLB();
}

int
CkDDT::getUB(int nIndex)
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getUB();
}

int CkDDT::getEnvelope(int nIndex, int *ni, int *na, int *nd, int *combiner)
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getEnvelope(ni, na, nd, combiner);
}
int CkDDT::getContents(int nIndex, int ni, int na, int nd, int i[], int a[], int d[])
{
  CkDDT_DataType* dttype = getType(nIndex);
  return dttype->getContents(ni, na, nd, i, a, d);
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
CkDDT::newIndexed(int count, int* arrbLength, int* arrDisp,
                  CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =
    new CkDDT_Indexed(count, arrbLength, arrDisp, oldtype, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_INDEXED ;
}

void
CkDDT::newHIndexed(int count, int* arrbLength, int* arrDisp,
                   CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =
    new CkDDT_HIndexed(count, arrbLength, arrDisp, oldtype, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_HINDEXED ;
}

void
CkDDT::newStruct(int count, int* arrbLength, int* arrDisp,
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

CkDDT_DataType::CkDDT_DataType(int type):datatype(type)
{
  count = 1;
  switch(datatype) {
    case CkDDT_DOUBLE:
      size = sizeof(double);
      break;
    case CkDDT_INT:
      size = sizeof(int);
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
    case CkDDT_DOUBLE_COMPLEX:
      size =  2 * sizeof(double) ;
      break;
    case CkDDT_LOGICAL:
      size =  sizeof(int) ;
      break;
    case CkDDT_SHORT:
      size = sizeof(short);
      break ;
    case CkDDT_LONG:
      size = sizeof(long);
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
      size = sizeof(float)+sizeof(int);
      break;
    case CkDDT_DOUBLE_INT:
      size = sizeof(double)+sizeof(int);
      break;
    case CkDDT_LONG_INT:
      size = sizeof(long)+sizeof(int);
      break;
    case CkDDT_2INT:
      size = 2*sizeof(int);
      break;
    case CkDDT_SHORT_INT:
      size = sizeof(short)+sizeof(int);
      break;
    case CkDDT_LONG_DOUBLE_INT:
      size = sizeof(long double)+sizeof(int);
      break;
    case CkDDT_2FLOAT:
      size = 2*sizeof(float);
      break;
    case CkDDT_2DOUBLE:
      size = 2*sizeof(double);
      break;
    case CkDDT_LB:
    case CkDDT_UB:
      size = 0;
      break;
#if CMK_LONG_LONG_DEFINED
    case CkDDT_LONG_LONG_INT:
      size = sizeof(CmiInt8);
      break;
#endif
    default:
      size = 0;
  }
  extent = size;
  lb = 0;
  ub = size;
  iscontig = 1;
  DDTDEBUG("CkDDT_DataType constructor: type=%d, size=%d, extent=%d\n",type,size,extent);
}


CkDDT_DataType::CkDDT_DataType(int datatype, int size, int extent, int count, int lb, int ub,
            int iscontig, int baseSize, int baseExtent, CkDDT_DataType* baseType, int baseIndex) :
    datatype(datatype), size(size), extent(extent), count(count), lb(lb), ub(ub), iscontig(iscontig),
    baseSize(baseSize), baseExtent(baseExtent), baseType(baseType), baseIndex(baseIndex)
{}

CkDDT_DataType::CkDDT_DataType(const CkDDT_DataType& obj)
{
  datatype = obj.datatype ;
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseType = obj.baseType;
  baseIndex = obj.baseIndex;
  iscontig = obj.iscontig;
}


//Assignment Operator
CkDDT_DataType&
CkDDT_DataType::operator=(const CkDDT_DataType& obj)
{
  if(this == &obj)
    return *this ;

  datatype = obj.datatype ;
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseType = obj.baseType;
  baseIndex = obj.baseIndex;
  iscontig = obj.iscontig;

  return *this;
}

int
CkDDT_DataType::serialize(char* userdata, char* buffer, int num, int dir)
{
  if(dir==1) {
    memcpy(buffer, userdata, num*size );
  } else if (dir==(-1)){
    memcpy(userdata, buffer, num*size );
  } else {
    CmiAbort("CkDDT: Invalid dir in serialize.\n");
  }
  return size ;
}

int
CkDDT_DataType::isContig()
{
  return iscontig;
}

int
CkDDT_DataType::getSize(int num)
{
  return num*size ;
}

int
CkDDT_DataType::getExtent(void)
{
  return extent ;
}

int
CkDDT_DataType::getBaseSize(void)
{
  return baseSize ;
}

int
CkDDT_DataType::getLB(void){
  return lb;
}

int
CkDDT_DataType::getUB(void){
  return ub;
}

int
CkDDT_DataType::getType(void){
  return datatype;
}

void
CkDDT_DataType::inrRefCount(void)
{
  refCount++ ;
}

int
CkDDT_DataType::getRefCount(void)
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
}

int CkDDT_DataType::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = 0;
  *na = 0;
  *nd = 0;
  *combiner = CkDDT_COMBINER_NAMED;
  return 0;
}

int CkDDT_DataType::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
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
  size = count * baseSize ;
  extent = count * baseExtent ;
  iscontig = 1;
  
  lb = baseType->getLB();
  ub = lb + extent;
  iscontig = oldType->isContig();
}

CkDDT_Contiguous::CkDDT_Contiguous(const CkDDT_Contiguous& obj)
{
  datatype = obj.datatype;
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseSize = obj.baseSize ;
  baseExtent = obj.baseExtent ;
  baseType = obj.baseType ;
  baseIndex = obj.baseIndex;
  lb = baseType->getLB();
  ub = lb + extent;
  iscontig = obj.iscontig;  
}

CkDDT_Contiguous&
CkDDT_Contiguous::operator=(const CkDDT_Contiguous& obj)
{
  if(this == &obj)
    return *this ;

  datatype = obj.datatype;
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseSize = obj.baseSize ;
  baseExtent = obj.baseExtent ;
  baseType = obj.baseType ;
  baseIndex = obj.baseIndex;
  lb = obj.lb;
  ub = obj.ub;
  iscontig = obj.iscontig;  
  return *this ;
}

int
CkDDT_Contiguous::serialize(char* userdata, char* buffer, int num, int dir)
{
  int bytesCopied  = 0  ;
  for(; num; num--) {
    bytesCopied += baseType->serialize(userdata, buffer, count, dir);
    buffer += (count*baseSize) ;
    userdata += (count*baseExtent) ;
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
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int CkDDT_Contiguous::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = 1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_CONTIGUOUS;
  return 0;
}

int CkDDT_Contiguous::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
  i[0] = count;
  d[0] = baseIndex;
  return 0;
}

CkDDT_Vector::CkDDT_Vector(int nCount, int blength, int stride, int bindex, CkDDT_DataType* type)
{
  datatype = CkDDT_VECTOR;
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;

  baseIndex = bindex;
  baseType =  type;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  size = count *  blockLength * baseSize ;
  extent = size + ( (strideLength - blockLength) * (count-1) * baseSize ) ;

  lb = baseType->getLB();
  ub = lb + extent;
  iscontig = 0;  
}

int
CkDDT_Vector::serialize(char* userdata, char* buffer, int num, int dir)
{
  int  bytesCopied = 0  ;
  for(;num;num--) {
    for(int i = 0 ; i < count; i++) {
      bytesCopied += baseType->serialize(userdata, buffer, blockLength, dir);
      buffer += (blockLength*baseSize) ;
      userdata += (strideLength*baseExtent);
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
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int CkDDT_Vector::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = 3;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_VECTOR;
  return 0;
}

int CkDDT_Vector::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
  i[0] = count;
  i[1] = blockLength;
  i[2] = strideLength;
  d[0] = baseIndex;
  return 0;
}

CkDDT_HVector::CkDDT_HVector(int nCount, int blength, int stride,  int bindex,
                         CkDDT_DataType* type)
{
  datatype = CkDDT_HVECTOR;
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;

  baseIndex = bindex;
  baseType = type ;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  size = count *  blockLength * baseSize ;
  extent = size + ( strideLength * (count-1) ) ;

  lb = baseType->getLB();
  ub = lb + extent;
  iscontig = 0;  
}

int
CkDDT_HVector::serialize(char* userdata, char* buffer, int num, int dir)
{
  int  bytesCopied = 0 ;
  for(;num;num--) {
    for(int i = 0 ; i < count; i++) {
      bytesCopied += baseType->serialize(userdata, buffer, blockLength, dir);
      buffer += (blockLength*baseSize) ;
      userdata += strideLength;
    }
  }
  return bytesCopied ;
}

void
CkDDT_HVector::pupType(PUP::er &p, CkDDT* ddt)
{
  CkDDT_Vector::pupType(p, ddt);
}

int CkDDT_HVector::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = 2;
  *na = 1;
  *nd = 1;
  *combiner = CkDDT_COMBINER_HVECTOR;
  return 0;
}

int CkDDT_HVector::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
  i[0] = count;
  i[1] = blockLength;
  a[0] = strideLength;
  d[0] = baseIndex;
  return 0;
}

CkDDT_Indexed::CkDDT_Indexed(int nCount, int* arrBlock, int* arrDisp, int bindex,
                         CkDDT_DataType* base)
    : CkDDT_DataType(CkDDT_INDEXED, 0, 0, nCount, numeric_limits<int>::max(),
		     numeric_limits<int>::min(), 0, base->getSize(), base->getExtent(),
		     base, bindex),
    arrayBlockLength(new int[nCount]), arrayDisplacements(new int[nCount])
{
    for(int i=0; i<count; i++) {
        arrayBlockLength[i] = arrBlock[i] ;
        arrayDisplacements[i] = arrDisp[i] ;
        size += ( arrBlock[i] * baseSize) ;
        extent += ((arrBlock[i]*baseExtent) + (arrayDisplacements[i]*baseExtent));
    }

    lb = baseType->getLB();
    ub = lb + extent;
}

int
CkDDT_Indexed::serialize(char* userdata, char* buffer, int num, int dir)
{
  char* tbuf = userdata ;
  int bytesCopied = 0 ;

  for(;num;num--) {
    for(int i = 0 ; i < count; i++) {
      userdata = tbuf + baseSize * arrayDisplacements[i] ;
      for(int j = 0; j < arrayBlockLength[i] ; j++) {
        bytesCopied +=  baseType->serialize(userdata, buffer, 1, dir);
        buffer += baseSize ;
        userdata += baseExtent ;
      }
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
  
  if(p.isUnpacking() )  arrayBlockLength = new int[count] ;
  p(arrayBlockLength, count);

  if(p.isUnpacking() )  arrayDisplacements = new int[count] ;
  p(arrayDisplacements, count);

  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

int CkDDT_Indexed::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = count*2+1;
  *na = 0;
  *nd = 1;
  *combiner = CkDDT_COMBINER_INDEXED;
  return 0;
}

int CkDDT_Indexed::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    i[z+i[0]+1] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return 0;
}

CkDDT_HIndexed::CkDDT_HIndexed(int nCount, int* arrBlock, int* arrDisp,  int bindex,
                           CkDDT_DataType* base)
    : CkDDT_Indexed(nCount, arrBlock, arrDisp, bindex, base)
{
  datatype = CkDDT_HINDEXED;
  size = 0;
  ub = numeric_limits<int>::min();
  for (int i = 0; i<count; i++) {
      size += (arrBlock[i] * baseSize);
      ub = std::max(arrBlock[i]*baseExtent + baseType->getLB() + arrayDisplacements[i], ub);
  }

  lb = baseType->getLB() + *std::min_element(arrDisp, arrDisp+nCount+1);
  extent = ub - lb;
}

int
CkDDT_HIndexed::serialize(char* userdata, char* buffer, int num, int dir)
{
  char* tbuf = userdata ;
  int bytesCopied = 0 ;

  for(;num;num--) {
    for(int i = 0 ; i < count; i++) {
      userdata = tbuf + arrayDisplacements[i] ;
      for(int j = 0; j < arrayBlockLength[i] ; j++) {
        bytesCopied += baseType->serialize(userdata, buffer, 1, dir);
        buffer += baseSize ;
        userdata += baseExtent ;
      }
    }
  }
  return bytesCopied ;
}

void
CkDDT_HIndexed::pupType(PUP::er &p, CkDDT* ddt)
{
  CkDDT_Indexed::pupType(p, ddt);
}

int CkDDT_HIndexed::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = count+1;
  *na = count;
  *nd = 1;
  *combiner = CkDDT_COMBINER_HINDEXED;
  return 0;
}

int CkDDT_HIndexed::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
  }
  d[0] = baseIndex;
  return 0;
}

CkDDT_Struct::CkDDT_Struct(int nCount, int* arrBlock,
                       int* arrDisp, int *bindex, CkDDT_DataType** arrBase)
    : CkDDT_DataType(CkDDT_STRUCT, 0, 0, nCount, numeric_limits<int>::max(),
		     numeric_limits<int>::min(), 0, 0, 0, NULL, 0),
    arrayBlockLength(new int[nCount]), arrayDisplacements(new int[nCount]),
    index(new int[nCount]), arrayDataType(new CkDDT_DataType*[nCount])
{
  for (int i=0; i<count; i++) {
      arrayBlockLength[i] = arrBlock[i];
      arrayDisplacements[i] = arrDisp[i];
      arrayDataType[i] =  arrBase[i];
      index[i] = bindex[i];
      size += arrBlock[i]*arrayDataType[i]->getSize();
  }

  bool explicit_lb = false;
  bool explicit_ub = false;
  for (int i=0; i<count; i++) {
      int xlb = arrayDataType[i]->getLB() + arrDisp[i];
      int xub = arrayDataType[i]->getUB() + arrDisp[i];
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
  DDTDEBUG("type %d: ub=%d, lb=%d, extent=%d, size=%d\n",datatype,ub,lb,extent,size);
}

int CkDDT_Struct::serialize(char* userdata, char* buffer, int num, int dir) {
  char* sbuf = userdata;
  char* dbuf = buffer;
  int bytesCopied = 0;

  for (; num; num--) {
      for (int i=0; i<count; i++) {
          for (int j=0; j<arrayBlockLength[i]; j++) {
              DDTDEBUG("writing block of type %d from offset %d to offset %d\n",
                      arrayDataType[i]->getType(),
                      userdata-sbuf,
                      buffer-dbuf);
              bytesCopied += arrayDataType[i]->serialize(
                      userdata + arrayDisplacements[i],
                      buffer,
                      1,
                      dir);
              buffer += arrayDataType[i]->getSize();
          }
      }
      userdata += extent;
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
  if(p.isUnpacking())
  {
    arrayBlockLength = new int[count] ;
    arrayDisplacements = new int[count] ;
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

int CkDDT_Struct::getEnvelope(int *ni, int *na, int *nd, int *combiner){
  *ni = count+1;
  *na = count;
  *nd = count;
  *combiner = CkDDT_COMBINER_STRUCT;
  return 0;
}

int CkDDT_Struct::getContents(int ni, int na, int nd, int i[], int a[], int d[]){
  i[0] = count;
  for(int z=0;z<i[0];z++){
    i[z+1] = arrayBlockLength[z];
    a[z] = arrayDisplacements[z];
    d[z] = index[z];
  }
  return 0;
}

