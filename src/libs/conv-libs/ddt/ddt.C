#include "ddt.h"

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
        case CkDDT_PRIMITIVE:
          typeTable[i] = new CkDDT_DataType ;
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
        default:
          //Not a defined type.
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
  delete typeTable[*index];
  typeTable[*index] = 0 ;
  types[*index] = CkDDT_TYPE_NULL ;
  *index = -1 ;
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

void 
CkDDT::newContiguous(int count, CkDDT_Type oldType, CkDDT_Type *newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType *type  = new CkDDT_Contiguous(count, index, typeTable[oldType]);
  typeTable[index] = type ;
  types[index] = CkDDT_CONTIGUOUS ;
}

void 
CkDDT::newVector(int count, int blocklength, int stride, 
                 CkDDT_Type oldType, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type = 
    new CkDDT_Vector(count, blocklength, stride, index, typeTable[oldType]);
  typeTable[index] = type ;
  types[index] = CkDDT_VECTOR ;
}

void 
CkDDT::newHVector(int count, int blocklength, int stride, 
                  CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =  
    new CkDDT_HVector(count, blocklength, stride, index, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_HVECTOR ;
}

void 
CkDDT::newIndexed(int count, int* arrbLength, int* arrDisp, 
                  CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =  
    new CkDDT_Indexed(count, arrbLength, arrDisp, index, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_INDEXED ;
}

void 
CkDDT::newHIndexed(int count, int* arrbLength, int* arrDisp, 
                   CkDDT_Type oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType* type =  
    new CkDDT_HIndexed(count, arrbLength, arrDisp, index, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = CkDDT_HINDEXED ;
}

void 
CkDDT::newStruct(int count, int* arrbLength, int* arrDisp, 
                 CkDDT_Type *oldtype, CkDDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  CkDDT_DataType **oldtypes = new CkDDT_DataType*[count];
  for(int i=0;i<count;i++)
    oldtypes[i] = typeTable[oldtype[i]];
  CkDDT_DataType* type =  
    new CkDDT_Struct(count, arrbLength, arrDisp, oldtype, oldtypes);
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
    default:
      break;
  }
  extent = size ;
}

CkDDT_DataType::CkDDT_DataType(const CkDDT_DataType& obj)
{
  datatype = obj.datatype ;
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseType = obj.baseType;
  baseIndex = obj.baseIndex;
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
    CkAbort("CkDDT: Invalid dir in serialize.\n");
  }
  return size ;
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
}

CkDDT_Contiguous::CkDDT_Contiguous(int nCount, int bindex, CkDDT_DataType* oldType)
{
  count = nCount ;

  baseType = oldType;
  baseIndex = bindex;
  baseSize = baseType->getSize();
  baseExtent = baseType->getExtent() ;

  size = count * baseSize ;
  extent = count * baseExtent ; ;
}

CkDDT_Contiguous::CkDDT_Contiguous(const CkDDT_Contiguous& obj)
{
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseSize = obj.baseSize ;
  baseExtent = obj.baseExtent ;
  baseType = obj.baseType ;
  baseIndex = obj.baseIndex;
}

CkDDT_Contiguous& 
CkDDT_Contiguous::operator=(const CkDDT_Contiguous& obj)
{
  if(this == &obj)
    return *this ;

  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseSize = obj.baseSize ;
  baseExtent = obj.baseExtent ;
  baseType = obj.baseType ;
  baseIndex = obj.baseIndex;

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
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

CkDDT_Vector::CkDDT_Vector(int nCount, int blength, int stride, int bindex, CkDDT_DataType* type)
{
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;

  baseIndex = bindex;
  baseType =  type;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  size = count *  blockLength * baseSize ;
  extent = size + ( (strideLength - blockLength) * (count-1) * baseSize ) ;
}

int 
CkDDT_Vector::serialize(char* userdata, char* buffer, int num, int dir)
{
  int  bytesCopied = 0  ;
  for(;num;num--) {
    for(int i = 0 ; i < count; i++) {
      bytesCopied += baseType->serialize(userdata, buffer, blockLength, dir);
      buffer += (blockLength*baseSize) ;
      userdata += ((blockLength+strideLength)*baseExtent);
    }
  }
  return bytesCopied ;
}

void 
CkDDT_Vector::pupType(PUP::er &p, CkDDT* ddt)
{  
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(blockLength);
  p(strideLength);
  p(baseIndex);
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

CkDDT_HVector::CkDDT_HVector(int nCount, int blength, int stride,  int bindex,
                         CkDDT_DataType* type)
{
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;

  baseIndex = bindex;
  baseType = type ;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  size = count *  blockLength * baseSize ;
  extent = size + ( strideLength * (count-1) ) ;
}

int 
CkDDT_HVector::serialize(char* userdata, char* buffer, int num, int dir)
{
  int  bytesCopied = 0 ;

  for(;num;num--) {
    for(int i = 0 ; i < count; i++) {
      bytesCopied += baseType->serialize(userdata, buffer, blockLength, dir);
      buffer += (blockLength*baseSize) ;
      userdata += ((blockLength+strideLength)*baseExtent);
    }
  }
  return bytesCopied ;
}

void 
CkDDT_HVector::pupType(PUP::er &p, CkDDT* ddt)
{  
  CkDDT_Vector::pupType(p, ddt);
}

CkDDT_Indexed::CkDDT_Indexed(int nCount, int* arrBlock, int* arrDisp, int bindex,
                         CkDDT_DataType* base)
{
  count = nCount ;

  baseType = base;
  baseIndex = bindex;

  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  arrayBlockLength = new int[count] ;
  arrayDisplacements = new int[count] ;

  for(int i = 0 ; i < count ; i++) {
    arrayBlockLength[i] = arrBlock[i] ;
    arrayDisplacements[i] = arrDisp[i] ;
    size = size + ( arrBlock[i] * baseSize) ;
    extent += ((arrBlock[i]*baseExtent) + (arrayDisplacements[i]*baseExtent));
  }
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
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);

  if(p.isUnpacking() )  arrayBlockLength = new int[count] ;
  p(arrayBlockLength, count);

  if(p.isUnpacking() )  arrayDisplacements = new int[count] ;
  p(arrayDisplacements, count);

  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

CkDDT_HIndexed::CkDDT_HIndexed(int nCount, int* arrBlock, int* arrDisp,  int bindex,
                           CkDDT_DataType* base)
{
  count = nCount ;

  baseType = base;
  baseIndex = bindex;

  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  arrayBlockLength = new int[count] ;
  arrayDisplacements = new int[count] ;

  for(int i = 0 ; i < count ; i++) {
    arrayBlockLength[i] = arrBlock[i] ;
    arrayDisplacements[i] = arrDisp[i] ;
    size = size + ( arrBlock[i] * baseSize) ;
    extent += ( (arrBlock[i] * baseExtent) + arrayDisplacements[i]  );
  }
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

CkDDT_Struct::CkDDT_Struct(int nCount, int* arrBlock, 
                       int* arrDisp, int *bindex, CkDDT_DataType** arrBase)
{
  int basesize ;
  int baseextent ;

  count = nCount ;

  arrayBlockLength = new int[count] ;
  arrayDisplacements = new int[count] ;
  arrayDataType = new CkDDT_DataType*[count];
  index = new int[count];
  //check this...

  for(int i=0 ; i < count ; i++) {
    arrayBlockLength[i] = arrBlock[i] ;
    arrayDisplacements[i] = arrDisp[i] ;
    arrayDataType[i] =  arrBase[i]; 
    index[i] = bindex[i];
    basesize = arrayDataType[i]->getSize();
    baseextent = arrayDataType[i]->getExtent();

    size = size + ( arrBlock[i] * basesize) ;
    extent += ((arrBlock[i]*baseextent) + (arrayDisplacements[i]*baseextent));
  }
}

int 
CkDDT_Struct::serialize(char* userdata, char* buffer, int num, int dir)
{
  char* tbuf = userdata ;
  int bytesCopied = 0 ;

  for(;num;num--) {
    for(int i = 0 ; i < count ; i++) {
      userdata = tbuf + arrayDisplacements[i] ;
      for(int j = 0 ; j < arrayBlockLength[i] ; j++) {
        bytesCopied += arrayDataType[i]->serialize(userdata, buffer, 1, dir);
        buffer += arrayDataType[i]->getSize();
        userdata += arrayDataType[i]->getExtent();
      }
    }
  }
  return bytesCopied ;
}

void 
CkDDT_Struct::pupType(PUP::er &p, CkDDT* ddt)
{
  p(size);
  p(extent);
  p(count);

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
