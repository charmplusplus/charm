#include "ddt.h"

DDT_DataType* 
DDT::getType(int nIndex)
{
  if( (nIndex >= 0) && (nIndex < max_types))
    return typeTable[nIndex] ;
  else
    return 0 ;
}

void 
DDT::pup(PUP::er &p)
{
  p(max_types);
  p(num_types);
  if(p.isUnpacking())
  {
    typeTable = new DDT_DataType*[max_types];
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
        case DDT_PRIMITIVE:
          typeTable[i] = new DDT_DataType ;
          break ;
        case DDT_CONTIGUOUS:
          typeTable[i] = new DDT_Contiguous ;
          break ;
        case DDT_VECTOR:
          typeTable[i] = new DDT_Vector ;
          break ;
        case DDT_HVECTOR:
          typeTable[i] = new DDT_HVector ;
          break ;
        case DDT_INDEXED:
          typeTable[i] = new DDT_Indexed ;
          break ;
        case DDT_HINDEXED:
          typeTable[i] = new DDT_HIndexed ;
          break ;
        case DDT_STRUCT:
          typeTable[i] = new DDT_Struct ;
          break ;
        default:
          //Not a defined type.
          break ;
      }
    } //End of for loop
  } //end if p.Unpacking()

  for(i=0; i < max_types ; i++)
  {
    if(types[i] != DDT_TYPE_NULL)
    {
      typeTable[i]->pupType(p, this);
      if(p.isDeleting())
        delete typeTable[i];
    }
  }
  if(p.isDeleting())
  {
    delete[] typeTable;
    delete[] types;
  }
}

int  
DDT::getNextFreeIndex(void)
{
  int  i;

  if(num_types < max_types)
    return num_types++;
  for(i=0; i<num_types; i++)
    if(typeTable[i] == 0)
      return i ;
  int newmax = max_types*2;
  DDT_DataType** newtable = new DDT_DataType*[newmax];
  int *newtype = new int[newmax];
  for(i=0;i<max_types;i++)
  {
    newtable[i] = typeTable[i];
    newtype[i] = types[i];
  }
  for(i=max_types;i<newmax;i++)
  {
    newtable[i] = 0;
    newtype[i] = DDT_TYPE_NULL;
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
DDT::freeType(int* index)
{
  // FIXME: Use reference counting
  delete typeTable[*index];
  typeTable[*index] = 0 ;
  types[*index] = DDT_TYPE_NULL ;
  *index = -1 ;
}

DDT::~DDT()
{
  delete[] typeTable ;
  delete[] types;
}


int 
DDT::getSize(int nIndex, int count)
{
  DDT_DataType* dttype = getType(nIndex);
  return count*dttype->getSize();
}

int 
DDT::getExtent(int nIndex)
{
  DDT_DataType* dttype = getType(nIndex);
  return dttype->getExtent();
}

void 
DDT::newContiguous(int count, DDT_Type oldType, DDT_Type *newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType *type  = new DDT_Contiguous(count, index, typeTable[oldType]);
  typeTable[index] = type ;
  types[index] = DDT_CONTIGUOUS ;
}

void 
DDT::newVector(int count, int blocklength, int stride, 
                 DDT_Type oldType, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type = 
    new DDT_Vector(count, blocklength, stride, index, typeTable[oldType]);
  typeTable[index] = type ;
  types[index] = DDT_VECTOR ;
}

void 
DDT::newHVector(int count, int blocklength, int stride, 
                  DDT_Type oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_HVector(count, blocklength, stride, index, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = DDT_HVECTOR ;
}

void 
DDT::newIndexed(int count, int* arrbLength, int* arrDisp, 
                  DDT_Type oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_Indexed(count, arrbLength, arrDisp, index, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = DDT_INDEXED ;
}

void 
DDT::newHIndexed(int count, int* arrbLength, int* arrDisp, 
                   DDT_Type oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_HIndexed(count, arrbLength, arrDisp, index, typeTable[oldtype]);
  typeTable[index] = type ;
  types[index] = DDT_HINDEXED ;
}

void 
DDT::newStruct(int count, int* arrbLength, int* arrDisp, 
                 DDT_Type *oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType **oldtypes = new DDT_DataType*[count];
  for(int i=0;i<count;i++)
    oldtypes[i] = typeTable[oldtype[i]];
  DDT_DataType* type =  
    new DDT_Struct(count, arrbLength, arrDisp, oldtype, oldtypes);
  typeTable[index] = type ;
  types[index] = DDT_STRUCT ;
}

DDT_DataType::DDT_DataType(int type):datatype(type)
{
  count = 1;
  switch(datatype) {
    case DDT_DOUBLE:
      size = sizeof(double);
      break;
    case DDT_INT:
      size = sizeof(int);
      break;
    case DDT_FLOAT:
      size = sizeof(float);
      break;
    case DDT_CHAR:
      size = sizeof(char);
      break;
    case DDT_BYTE:
      size = 1 ;
      break;
    case DDT_PACKED:
      size = 1 ;
      break;
    case DDT_COMPLEX:
      size =  2 * sizeof(double) ;
      break;
    case DDT_LOGICAL:
      size =  sizeof(int) ;
      break;
    case DDT_SHORT:
      size = sizeof(short);
      break ;
    case DDT_LONG:
      size = sizeof(long);
      break ;
    case DDT_UNSIGNED_CHAR:
      size = sizeof(unsigned char);
      break;
    case DDT_UNSIGNED_SHORT:
      size = sizeof(unsigned short);
      break;
    case DDT_UNSIGNED:
      size = sizeof(unsigned);
      break ;
    case DDT_UNSIGNED_LONG:
      size = sizeof(unsigned long);
      break ;
    case DDT_LONG_DOUBLE:
      size = sizeof(long double);
      break ;
    default:
      break;
  }
  extent = size ;
}

DDT_DataType::DDT_DataType(const DDT_DataType& obj)
{
  datatype = obj.datatype ;
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseType = obj.baseType;
  baseIndex = obj.baseIndex;
}


//Assignment Operator
DDT_DataType& 
DDT_DataType::operator=(const DDT_DataType& obj)
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
DDT_DataType::serialize(char* userdata, char* buffer, int num, int dir)
{
  if(dir==1) {
    memcpy(buffer, userdata, num*size );
  } else if (dir==(-1)){
    memcpy(userdata, buffer, num*size );
  } else {
    CkAbort("DDT: Invalid dir in serialize.\n");
  }
  return size ;
}

int
DDT_DataType::getSize(int num)
{
  return num*size ;
}

int
DDT_DataType::getExtent(void)
{
  return extent ;
}

void 
DDT_DataType::inrRefCount(void)
{
  refCount++ ;
}

int 
DDT_DataType::getRefCount(void)
{
  return refCount ;
}

void 
DDT_DataType::pupType(PUP::er  &p, DDT* ddt)
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

DDT_Contiguous::DDT_Contiguous(int nCount, int bindex, DDT_DataType* oldType)
{
  count = nCount ;

  baseType = oldType;
  baseIndex = bindex;
  baseSize = baseType->getSize();
  baseExtent = baseType->getExtent() ;

  size = count * baseSize ;
  extent = count * baseExtent ; ;
}

DDT_Contiguous::DDT_Contiguous(const DDT_Contiguous& obj)
{
  size = obj.size ;
  extent = obj.extent ;
  count = obj.count ;
  baseSize = obj.baseSize ;
  baseExtent = obj.baseExtent ;
  baseType = obj.baseType ;
  baseIndex = obj.baseIndex;
}

DDT_Contiguous& 
DDT_Contiguous::operator=(const DDT_Contiguous& obj)
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
DDT_Contiguous::serialize(char* userdata, char* buffer, int num, int dir)
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
DDT_Contiguous::pupType(PUP::er &p, DDT *ddt)
{
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(baseIndex);
  if(p.isUnpacking()) baseType = ddt->getType(baseIndex);
}

DDT_Vector::DDT_Vector(int nCount, int blength, int stride, int bindex, DDT_DataType* type)
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
DDT_Vector::serialize(char* userdata, char* buffer, int num, int dir)
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
DDT_Vector::pupType(PUP::er &p, DDT* ddt)
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

DDT_HVector::DDT_HVector(int nCount, int blength, int stride,  int bindex,
                         DDT_DataType* type)
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
DDT_HVector::serialize(char* userdata, char* buffer, int num, int dir)
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
DDT_HVector::pupType(PUP::er &p, DDT* ddt)
{  
  DDT_Vector::pupType(p, ddt);
}

DDT_Indexed::DDT_Indexed(int nCount, int* arrBlock, int* arrDisp, int bindex,
                         DDT_DataType* base)
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
DDT_Indexed::serialize(char* userdata, char* buffer, int num, int dir)
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

DDT_Indexed::~DDT_Indexed()
{
  delete [] arrayBlockLength ;
  delete [] arrayDisplacements ;
}

void 
DDT_Indexed::pupType(PUP::er &p, DDT* ddt)
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

DDT_HIndexed::DDT_HIndexed(int nCount, int* arrBlock, int* arrDisp,  int bindex,
                           DDT_DataType* base)
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
DDT_HIndexed::serialize(char* userdata, char* buffer, int num, int dir)
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
DDT_HIndexed::pupType(PUP::er &p, DDT* ddt)
{
  DDT_Indexed::pupType(p, ddt);
}

DDT_Struct::DDT_Struct(int nCount, int* arrBlock, 
                       int* arrDisp, int *bindex, DDT_DataType** arrBase)
{
  int basesize ;
  int baseextent ;

  count = nCount ;

  arrayBlockLength = new int[count] ;
  arrayDisplacements = new int[count] ;
  arrayDataType = new DDT_DataType*[count];
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
DDT_Struct::serialize(char* userdata, char* buffer, int num, int dir)
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
DDT_Struct::pupType(PUP::er &p, DDT* ddt)
{
  p(size);
  p(extent);
  p(count);

  if(p.isUnpacking())
  {
    arrayBlockLength = new int[count] ;
    arrayDisplacements = new int[count] ;
    index = new int[count] ;
    arrayDataType = new DDT_DataType*[count] ;
  }
  p(arrayBlockLength, count);
  p(arrayDisplacements, count);
  p(index, count);

  if(p.isUnpacking())
    for(int i=0 ; i < count; i++)
      arrayDataType[i] = ddt->getType(index[i]);
}
