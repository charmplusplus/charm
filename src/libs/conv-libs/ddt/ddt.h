#ifndef __DDT_H_
#define __DDT_H_

#include <string.h>
#include <stdio.h>
#include "charm++.h"

#define DDT_DOUBLE         0
#define DDT_INT            1
#define DDT_FLOAT          2
#define DDT_COMPLEX        3
#define DDT_LOGICAL        4
#define DDT_CHAR           5
#define DDT_BYTE           6
#define DDT_PACKED         7
#define DDT_SHORT          8
#define DDT_LONG           9
#define DDT_UNSIGNED_CHAR  10
#define DDT_UNSIGNED_SHORT 11
#define DDT_UNSIGNED       12
#define DDT_UNSIGNED_LONG  13
#define DDT_LONG_DOUBLE    14

#define DDT_TYPE_NULL  -1
#define DDT_PRIMITIVE  14
#define DDT_CONTIGUOUS 15
#define DDT_VECTOR     16
#define DDT_HVECTOR    17
#define DDT_INDEXED    18
#define DDT_HINDEXED   19
#define DDT_STRUCT     20

typedef int DDT_Type ;
class DDT ;

/* This class maintains the data for primitive data types 
 * and also acts as Base Class
   for all derived types.

  Members:

  datatype - Used for primitive datatypes for size calculations
  refCount - to keep track of how many references are present to this type.

  size - size of one unit of datatype
  extent - extent is different from size in that it also counts 
           displacements between blocks.
  count -  count of base datatype present in this datatype
  baseSize - size of Base Datatype
  baseExtent - extent of Base dataType

  Methods:

  getSize -  returns the size of the datatype. 
  getExtent - returns the extent of the datatype.

  inrRefCount - increament the reference count.
  getRefCount - returns the RefCount

  serialize - This is the function which actually copies the contents from
    user's space to buffer if dir=1 or reverse if dir=0 
    according to the datatype.
*/

class DDT_DataType {

  private:
    int datatype;
    int refCount;

  protected:
    int size;
    int extent;
    int count;

    int baseSize;
    int baseExtent;
    DDT_DataType *baseType;
    int baseIndex;

  public:
    DDT_DataType() { } ;
    ~DDT_DataType() { }  ;
    DDT_DataType(int type) ;
    DDT_DataType(const DDT_DataType& obj) ;
    DDT_DataType& operator=(const DDT_DataType& obj);

    virtual int getSize(int count=1);
    virtual int getExtent();
    virtual void inrRefCount() ;
    virtual int getRefCount() ;
    virtual void pupType(PUP::er &p, DDT* ddt) ;

    virtual int serialize(char* userdata, char* buffer, int num, int dir);
};

/*   
  This class maintains the type Contiguous. 
  It constructs a typemap consisting of the
  replication of a datatype into contiguous locations. 
*/

class DDT_Contiguous : public DDT_DataType {
  
  public:
  DDT_Contiguous() { };
  DDT_Contiguous(int count, int index, DDT_DataType* oldType);
  DDT_Contiguous(const DDT_Contiguous& obj) ;
  DDT_Contiguous& operator=(const DDT_Contiguous& obj);
  virtual int serialize(char* userdata, char* buffer, int num, int dir);
  virtual  void pupType(PUP::er &p, DDT* ddt) ;
};

/*
   This class maintains the Vector Datatype. 
   Vector type allows replication of a datatype into
   locations that consist of equally spaced blocks. 
   Each block is obtained by concatenating the
   same number of copies of the old datatype. 
   The spacing between blocks is a multiple of the
   extent of the old datatype.
*/

class DDT_Vector : public DDT_DataType {

  protected:
    int blockLength ;
    int strideLength ;
  public:
    DDT_Vector(int count, int blklen, int stride, int index,
               DDT_DataType* type);
    DDT_Vector(const DDT_Vector& obj) ;
    DDT_Vector& operator=(const DDT_Vector& obj);
    DDT_Vector() { } ;
    ~DDT_Vector() { } ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual  void pupType(PUP::er &p, DDT* ddt) ;
};

/*
  This class maintains the HVector Datatype. 
  HVector type allows replication of a datatype into locations 
  that consist of equally spaced 
  blocks. Each block is obtained by concatenating the same number of 
  copies of the old datatype.
  The Vector type assumes that the stride between successive blocks 
  is a multiple of the oldtype
  extent. HVector type allows a stride which consists of an 
  arbitrary number of bytes.
*/

class DDT_HVector : public DDT_Vector {

  public:
    DDT_HVector() { } ;
    DDT_HVector(int nCount,int blength,int strideLen,int index,
                DDT_DataType* type);
    ~DDT_HVector() { } ;
    DDT_HVector(const DDT_HVector& obj) ;
    DDT_HVector& operator=(const DDT_HVector& obj);
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual void pupType(PUP::er &p, DDT* ddt);
};

/*
  The Indexed type allows one to specify a noncontiguous data 
  layout where displacements between
  successive blocks need not be equal. 
  This allows one to gather arbitrary entries from an array
  and make a single buffer out of it. 
  All block displacements are measured  in units of oldtype extent.

  arrayBlockLength - holds the array of block lengths 
  arrayDisplacements - holds the array of displacements.
*/

class DDT_Indexed : public DDT_DataType {

  protected:
    int* arrayBlockLength ;
    int* arrayDisplacements ;

  public:

    DDT_Indexed(int count, int* arrBlock, int* arrDisp, int index, 
                DDT_DataType* type);
    DDT_Indexed(const DDT_Indexed& obj);
    DDT_Indexed& operator=(const DDT_Indexed& obj) ;
    DDT_Indexed() { } ;
    ~DDT_Indexed() ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual  void pupType(PUP::er &p, DDT* ddt) ;
};

/*
  The HIndexed type allows one to specify a noncontiguous data 
  layout where displacements between
  successive blocks need not be equal. 
  This allows one to gather arbitrary entries from an array
  and make a single buffer out of it. 
  Unlike Indexed type , block displacements are arbitrary
  number of bytes.

  arrayBlockLength - holds the array of block lengths 
  arrayDisplacements - holds the array of displacements.
*/
class DDT_HIndexed : public DDT_Indexed {

  public:
    DDT_HIndexed() { } ;
    DDT_HIndexed(int count, int* arrBlock, int* arrDisp, int index, 
                 DDT_DataType* type);
    DDT_HIndexed(const DDT_HIndexed& obj);
    DDT_HIndexed& operator=(const DDT_HIndexed& obj) ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual void pupType(PUP::er &p, DDT* ddt);
};

/*
  DDT_Struct is the most general type constructor. 
  It further generalizes DDT_HIndexed in
  that it allows each block to consist of replications of 
  different datatypes. 
  The intent is to allow descriptions of arrays of structures, 
  as a single datatype.

  arrayBlockLength - array of block lengths
  arrayDisplacements - array of displacements
  arrayDataType - array of DataTypes.
*/

class DDT_Struct : public DDT_DataType {

  protected:
    int* arrayBlockLength ;
    int* arrayDisplacements ;
    int* index;
    DDT_DataType** arrayDataType; 

  public:
    DDT_Struct() { } ;
    DDT_Struct(int count, int* arrBlock, int* arrDisp, int *index, 
               DDT_DataType **type);
    DDT_Struct(const DDT_Struct& obj);
    DDT_Struct& operator=(const DDT_Struct& obj) ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual  void pupType(PUP::er &p, DDT* ddt) ;
};

/*
  This class maintains the typeTable of the derived datatypes.
  First few entries of the table contain primitive datatypes.
  index - holds the current available index in the table where 
          new datatype can be allocated.
  typeTable - holds the table of DDT_DataType

  Type_Contiguous - 
  Type_Vector - 
  Type_HVector - 
  Type_Indexed - 
  Type_HIndexed - 
  Type_Struct - builds the new type 
                Contiguous/Vector/Hvector/Indexed/HIndexed/Struct  from the old
                Type provided and stores the new type in the table.
*/

class DDT {
  private:
    DDT_DataType**  typeTable;
    int*  types; //used for pup
    int max_types;
    int num_types;

  public:

  DDT(void*) {} // emulates migration constructor
  DDT()
  {
    max_types = 20;
    typeTable = new DDT_DataType*[max_types];
    types = new int[max_types];
    typeTable[0] = new DDT_DataType(DDT_DOUBLE);
    typeTable[1] = new DDT_DataType(DDT_INT);
    typeTable[2] = new DDT_DataType(DDT_FLOAT);
    typeTable[3] = new DDT_DataType(DDT_COMPLEX);
    typeTable[4] = new DDT_DataType(DDT_LOGICAL);
    typeTable[5] = new DDT_DataType(DDT_CHAR);
    typeTable[6] = new DDT_DataType(DDT_BYTE);
    typeTable[7] = new DDT_DataType(DDT_PACKED);
    typeTable[8] = new DDT_DataType(DDT_SHORT);
    typeTable[9] = new DDT_DataType(DDT_LONG);
    typeTable[10] = new DDT_DataType(DDT_UNSIGNED_CHAR);
    typeTable[11] = new DDT_DataType(DDT_UNSIGNED_SHORT);
    typeTable[12] = new DDT_DataType(DDT_UNSIGNED);
    typeTable[13] = new DDT_DataType(DDT_UNSIGNED_SHORT);
    typeTable[14] = new DDT_DataType(DDT_LONG_DOUBLE);
    num_types = 15;
    int i;
    for(i=0 ; i < num_types; i++)
      types[i] = DDT_PRIMITIVE ;
    for(i=num_types ; i < max_types; i++)
    {
      typeTable[i] = 0;
      types[i] = DDT_TYPE_NULL ;
    }
  }

  void newContiguous(int count, DDT_Type  oldType, DDT_Type* newType);
  void newVector(int count, int blocklength, int stride, DDT_Type oldtype, 
                DDT_Type* newtype);
  void newHVector(int count, int blocklength, int stride, DDT_Type oldtype, 
                 DDT_Type* newtype);
  void newIndexed(int count, int* arrbLength, int* arrDisp , DDT_Type oldtype, 
                 DDT_Type* newtype);
  void newHIndexed(int count, int* arrbLength, int* arrDisp , DDT_Type oldtype, 
                  DDT_Type* newtype);
  void newStruct(int count, int* arrbLength, int* arrDisp , DDT_Type *oldtype, 
                DDT_Type* newtype);
  void  freeType(int* index);
  int   getNextFreeIndex(void) ;
  void  pup(PUP::er &p);
  DDT_DataType*  getType(int nIndex);
  int  getSize(int nIndex, int count=1);
  int  getExtent(int nIndex);
  ~DDT() ;
};

#endif
