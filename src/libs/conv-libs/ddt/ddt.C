#include "ddt.h"

int DDT_Send(DDT* ddt, void* msg, int count, DDT_Type type, void* recvMsg)
{
  int bytesCopied , extentType ;
  char* oldBuffer = (char*) msg ;
  char* newBuffer = (char*) recvMsg ;


  char* tempOldBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  DDT_DataType* dttype = ddt->getType(type) ;
  extentType = dttype->getExtent();

  for(int i = 0 ; i < count ; i++)
  {
    bytesCopied = dttype->copyBuffer(oldBuffer, newBuffer);
    oldBuffer = oldBuffer + extentType ;
    newBuffer = newBuffer + bytesCopied ;
  }

  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ;
  if(1) printf("NumBytes Copied = %d\n", bytesCopied );
  return 0 ;
}

DDT_DataType* 
DDT::getType(int nIndex)
{
  if( (nIndex > 0) && (nIndex < MAX_TYPES))
    return typeTable[nIndex] ;
  else
    return 0 ;
}

void 
DDT::pup(PUP::er  &p)
{
  p(currentIndex);
  p(nextFreeIndex);
  p(types,MAX_TYPES);

  //unPacking
  if(p.isUnpacking())
  {
    for(int i = 0 ; i < MAX_TYPES; i++)
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

  for(int i = 0 ; i < MAX_TYPES ; i++) {
    if(types[i] != DDT_TYPE_NULL)
      typeTable[i]->pup(p);
  }
}

int  
DDT::getNextFreeIndex(void)
{
  int  i ;

  for(i=currentIndex; i<MAX_TYPES; i++) {
    if(typeTable[i] == 0) {
      return i ;
    }
  }
  for(i=0; i < currentIndex; i++) {
    if(typeTable[i] == 0) {
      return i ;
    }
  }
  return -1 ; //No free Index Available
}

void 
DDT::freeType(int* index)
{
  typeTable[*index] = 0 ;
  types[*index] = DDT_TYPE_NULL ;
  *index = -1 ;
}

DDT::~DDT()
{
  delete [] typeTable ;
}


int 
DDT::getSize(int nIndex)
{
  DDT_DataType* dttype = getType(nIndex);
  return dttype->getSize();
}

int 
DDT::getExtent(int nIndex)
{
  DDT_DataType* dttype = getType(nIndex);
  return dttype->getExtent();
}

int 
DDT::Type_Contiguous(int count, DDT_Type oldType, DDT_Type *newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType *type  = new DDT_Contiguous(count, typeTable[oldType]);
  typeTable[index] = type ;
  types[index] = DDT_CONTIGUOUS ;
  currentIndex = index ;
  return 0 ;
}

int 
DDT::Type_Vector(int count, int blocklength, int stride, 
                 DDT_Type oldType, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type = 
    new DDT_Vector(count, blocklength, stride, typeTable[oldType]);

  typeTable[index] = type ;
  types[index] = DDT_VECTOR ;
  currentIndex = index ;
  return 0 ;
}

int 
DDT::Type_HVector(int count, int blocklength, int stride, 
                  DDT_Type oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_HVector(count, blocklength, stride, typeTable[oldtype]);

  typeTable[index] = type ;
  types[index] = DDT_HVECTOR ;
  currentIndex = index ;
  return 0 ;
}

int 
DDT::Type_Indexed(int count, int* arrbLength, int* arrDisp, 
                  DDT_Type oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_Indexed(count, arrbLength, arrDisp, typeTable[oldtype]);

  typeTable[index] = type ;
  types[index] = DDT_INDEXED ;
  currentIndex = index ;
  return 0 ;
}

int 
DDT::Type_HIndexed(int count, int* arrbLength, int* arrDisp, 
                   DDT_Type oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_HIndexed(count, arrbLength, arrDisp, typeTable[oldtype]);

  typeTable[index] = type ;
  types[index] = DDT_HINDEXED ;
  currentIndex = index ;
  return 0 ;
}

int 
DDT::Type_Struct(int count, int* arrbLength, int* arrDisp, 
                 DDT_Type* oldtype, DDT_Type* newType)
{
  int index = *newType =  getNextFreeIndex() ;
  DDT_DataType* type =  
    new DDT_Struct(this, count, arrbLength, arrDisp, oldtype);

  typeTable[index] = type ;
  types[index] = DDT_STRUCT ;
  currentIndex = index ;
  return 0 ;
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

  return *this;
}

int 
DDT_DataType::copyBuffer(char* oldBuffer, char* newBuffer)
{
  int bytesCopied = size ;
  memcpy(newBuffer, oldBuffer, bytesCopied );

  return bytesCopied ;
}

int
DDT_DataType::getSize(void)
{
  return size ;
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
DDT_DataType::pup(PUP::er  &p)
{
  p(datatype);
  p(refCount);
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
}

DDT_Contiguous::DDT_Contiguous(int nCount, DDT_DataType* oldType)
{
  count = nCount ;

  baseType = oldType;
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

  return *this ;
}

int 
DDT_Contiguous::copyBuffer(char* oldBuffer, char* newBuffer)
{
  int bytesCopied  = 0  ;
  char* tempOldBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  for(int i = 0 ; i < count ; i ++) {
    bytesCopied += baseType->copyBuffer(oldBuffer, newBuffer);
    newBuffer += baseSize ;
    oldBuffer += baseExtent ;
  }

  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ;
  return bytesCopied ;
}

void 
DDT_Contiguous::pup(PUP::er &p)
{
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);

  baseType->pup(p);
}

DDT_Vector::DDT_Vector(int nCount, int blength, int stride, DDT_DataType* type)
{
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;

  baseType =  type;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  size = count *  blockLength * baseSize ;
  extent = size + ( (strideLength - blockLength) * (count-1) * baseSize ) ;
  
}

int 
DDT_Vector::copyBuffer(char* oldBuffer, char* newBuffer)
{
  char* tempOldBuffer = oldBuffer ;
  char* tempBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  int  bytesCopied = 0  ;

  for(int i = 0 ; i < count; i++) {
    oldBuffer = tempBuffer ;
    for(int j = 0; j < blockLength; j++) {
      bytesCopied += baseType->copyBuffer(oldBuffer, newBuffer);
      newBuffer += baseSize ;
    }  
    tempBuffer += strideLength * baseSize ;
  }
  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ; 

  return bytesCopied ;
}

void 
DDT_Vector::pup(PUP::er &p)
{  
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);
  p(blockLength);
  p(strideLength);

  baseType->pup(p);
}

DDT_HVector::DDT_HVector(int nCount, int blength, int stride, 
                         DDT_DataType* type)
{
  count = nCount ;
  blockLength = blength ;
  strideLength = stride ;

  baseType = type ;
  baseSize = baseType->getSize() ;
  baseExtent = baseType->getExtent() ;

  size = count *  blockLength * baseSize ;
  extent = size + ( strideLength * (count-1) ) ;
}

int 
DDT_HVector::copyBuffer(char* oldBuffer, char* newBuffer)
{
  char* tempOldBuffer = oldBuffer ;
  char* tempBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  int  bytesCopied = 0 ;

  for(int i = 0 ; i < count; i++) {
    oldBuffer = tempBuffer ;
    for(int j = 0; j < blockLength; j++) {
      bytesCopied += baseType->copyBuffer(oldBuffer, newBuffer);
      newBuffer += baseSize ;
    }  
    tempBuffer += strideLength ;
  }
  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ; 

  return bytesCopied ;
}

void 
DDT_HVector::pup(PUP::er &p)
{  
  DDT_Vector::pup(p);
}

DDT_Indexed::DDT_Indexed(int nCount, int* arrBlock, int* arrDisp, 
                         DDT_DataType* base)
{
  count = nCount ;

  baseType = base;

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
DDT_Indexed::copyBuffer(char* oldBuffer, char* newBuffer)
{
  char* tempOldBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  int bytesCopied = 0 ;

  for(int i = 0 ; i < count; i++) {
    oldBuffer = tempOldBuffer + baseSize * arrayDisplacements[i] ;
    for(int j = 0; j < arrayBlockLength[i] ; j++) {
      bytesCopied +=  baseType->copyBuffer(oldBuffer, newBuffer);
      newBuffer += baseSize ;
      oldBuffer += baseExtent ;
    }
  }

  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ;

  return bytesCopied ;
}

DDT_Indexed::~DDT_Indexed()
{
  delete [] arrayBlockLength ;
  delete [] arrayDisplacements ;
}

void 
DDT_Indexed::pup(PUP::er &p)
{
  p(size);
  p(extent);
  p(count);
  p(baseSize);
  p(baseExtent);

  if(p.isUnpacking() )  arrayBlockLength = new int[count] ;
  p(arrayBlockLength, count);

  if(p.isUnpacking() )  arrayDisplacements = new int[count] ;
  p(arrayDisplacements, count);

  baseType->pup(p);
}

DDT_HIndexed::DDT_HIndexed(int nCount, int* arrBlock, int* arrDisp, 
                           DDT_DataType* base)
{
  count = nCount ;

  baseType = base;

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
DDT_HIndexed::copyBuffer(char* oldBuffer, char* newBuffer)
{
  char* tempOldBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  int bytesCopied = 0 ;

  for(int i = 0 ; i < count; i++) {
    oldBuffer = tempOldBuffer + arrayDisplacements[i] ;
    for(int j = 0; j < arrayBlockLength[i] ; j++) {
      bytesCopied += baseType->copyBuffer(oldBuffer, newBuffer);
      newBuffer += baseSize ;
      oldBuffer += baseExtent ;
    }
  }

  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ;

  return bytesCopied ;
}

void 
DDT_HIndexed::pup(PUP::er &p)
{
  DDT_Indexed::pup(p);
}

DDT_Struct::DDT_Struct(DDT* ddt, int nCount, int* arrBlock, 
                       int* arrDisp, DDT_Type* arrBase)
{
  int basesize ;
  int baseextent ;

  count = nCount ;

  arrayBlockLength = new int[count] ;
  arrayDisplacements = new int[count] ;
  //check this...

  for(int i = 0 ; i < count ; i++) {
    arrayBlockLength[i] = arrBlock[i] ;
    arrayDisplacements[i] = arrDisp[i] ;

    arrayDataType[i] =  ddt->getType(arrBase[i]); 
    basesize = arrayDataType[i]->getSize();
    baseextent = arrayDataType[i]->getExtent();

    size = size + ( arrBlock[i] * basesize) ;
    extent += ((arrBlock[i]*baseextent) + (arrayDisplacements[i]*baseextent));
  }
}

int 
DDT_Struct::copyBuffer(char* oldBuffer, char* newBuffer)
{
  char* tempOldBuffer = oldBuffer ;
  char* tempNewBuffer = newBuffer ;
  int bytesCopied = 0 ;

  for(int i = 0 ; i < count ; i++) {
    oldBuffer = tempOldBuffer + arrayDisplacements[i] ;
    for(int j = 0 ; j < arrayBlockLength[i] ; j++) {
      bytesCopied += arrayDataType[i]->copyBuffer(oldBuffer, newBuffer);
      newBuffer += arrayDataType[i]->getSize();
      oldBuffer += arrayDataType[i]->getExtent();
    }
  }
  oldBuffer = tempOldBuffer ;
  newBuffer = tempNewBuffer ;

  return bytesCopied ;
}

void 
DDT_Struct::pup(PUP::er &p)
{
  p(size);
  p(extent);
  p(count);

  if(p.isUnpacking() )  arrayBlockLength = new int[count] ;
  p(arrayBlockLength, count);

  if(p.isUnpacking() )  arrayDisplacements = new int[count] ;
  p(arrayDisplacements, count);
  
  if(p.isUnpacking())
    types = new int[count] ;
  p(types, count);

  if(p.isUnpacking())
  {
    for(int i = 0 ; i < count; i++)
    {
      switch(types[i])
      {
        case DDT_PRIMITIVE:
          arrayDataType[i] = new DDT_DataType ;
          break ;
        case DDT_CONTIGUOUS:
          arrayDataType[i] = new DDT_Contiguous ;
          break ;
        case DDT_VECTOR:
          arrayDataType[i] = new DDT_Vector ;
          break ;
        case DDT_HVECTOR:
          arrayDataType[i] = new DDT_HVector ;
          break ;
        case DDT_INDEXED:
          arrayDataType[i] = new DDT_Indexed ;
          break ;
        case DDT_HINDEXED:
          arrayDataType[i] = new DDT_HIndexed ;
          break ;
        case DDT_STRUCT:
          arrayDataType[i] = new DDT_Struct ;
          break ;
        default:
          //Not a defined type.
          break ;
      }
    } //End of for loop
  } //end if p.Unpacking()

  for(int i = 0 ; i < count ; i++)
  {
    if(types[i] != DDT_TYPE_NULL)
      arrayDataType[i]->pup(p);
  }
}
