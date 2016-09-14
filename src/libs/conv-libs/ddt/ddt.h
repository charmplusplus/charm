#ifndef __CkDDT_H_
#define __CkDDT_H_

#include "pup.h"

#define CkDDT_MAXTYPE           100

#define CkDDT_TYPE_NULL          -1
#define CkDDT_DOUBLE              0
#define CkDDT_INT                 1
#define CkDDT_FLOAT               2
#define CkDDT_COMPLEX             3
#define CkDDT_LOGICAL             4
#define CkDDT_C_BOOL              5
#define CkDDT_CHAR                6
#define CkDDT_BYTE                7
#define CkDDT_PACKED              8
#define CkDDT_SHORT               9
#define CkDDT_LONG                10
#define CkDDT_UNSIGNED_CHAR       11
#define CkDDT_UNSIGNED_SHORT      12
#define CkDDT_UNSIGNED            13
#define CkDDT_UNSIGNED_LONG       14
#define CkDDT_LONG_DOUBLE         15
#define CkDDT_FLOAT_INT           16
#define CkDDT_DOUBLE_INT          17
#define CkDDT_LONG_INT            18
#define CkDDT_2INT                19
#define CkDDT_SHORT_INT           20
#define CkDDT_LONG_DOUBLE_INT     21
#define CkDDT_2FLOAT              22
#define CkDDT_2DOUBLE             23
#define CkDDT_LB                  24
#define CkDDT_UB                  25
#define CkDDT_LONG_LONG_INT       26
#define CkDDT_DOUBLE_COMPLEX      27
#define CkDDT_SIGNED_CHAR         28
#define CkDDT_UNSIGNED_LONG_LONG  29
#define CkDDT_WCHAR               30
#define CkDDT_INT8_T              31
#define CkDDT_INT16_T             32
#define CkDDT_INT32_T             33
#define CkDDT_INT64_T             34
#define CkDDT_UINT8_T             35
#define CkDDT_UINT16_T            36
#define CkDDT_UINT32_T            37
#define CkDDT_UINT64_T            38
#define CkDDT_FLOAT_COMPLEX       39
#define CkDDT_LONG_DOUBLE_COMPLEX 40
#define CkDDT_AINT                41

#define CkDDT_CONTIGUOUS          42
#define CkDDT_VECTOR              43
#define CkDDT_HVECTOR             44
#define CkDDT_INDEXED             45
#define CkDDT_HINDEXED            46
#define CkDDT_STRUCT              47
#define CkDDT_INDEXED_BLOCK       48
#define CkDDT_HINDEXED_BLOCK      49

/* for the datatype decoders */
#define CkDDT_COMBINER_NAMED          1
#define CkDDT_COMBINER_CONTIGUOUS     2
#define CkDDT_COMBINER_VECTOR         3
#define CkDDT_COMBINER_HVECTOR        4
#define CkDDT_COMBINER_INDEXED        5
#define CkDDT_COMBINER_HINDEXED       6
#define CkDDT_COMBINER_STRUCT         7
#define CkDDT_COMBINER_INDEXED_BLOCK  8
#define CkDDT_COMBINER_HINDEXED_BLOCK 9

#define CkDDT_MAX_NAME_LEN         255

typedef intptr_t CkDDT_Aint;

/* Helper function to set names (used by AMPI too).
 * Leading whitespaces are significant, trailing whitespaces are not. */
inline void CkDDT_SetName(char *dst, const char *src, int *len)
{
  CmiAssert(strlen(src) < CkDDT_MAX_NAME_LEN-1);
  int end = strlen(src)-1;
  while ((end>0) && (src[end]==' '))
    end--;
  *len = (end==0) ? 0 : end+1;
  memcpy(dst, src, *len);
  dst[*len] = '\0';
}

typedef int CkDDT_Type ;
class CkDDT ;

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
  name - user specified name for datatype
  nameLen - length of user specified name for datatype

  Methods:

  getSize -  returns the size of the datatype. 
  getExtent - returns the extent of the datatype.

  inrRefCount - increament the reference count.
  getRefCount - returns the RefCount

  serialize - This is the function which actually copies the contents from
    user's space to buffer if dir=1 or reverse if dir=-1
    according to the datatype.

  setName - set the name of datatype
  getName - get the name of datatype
  setAbsolute - tells DDT's serialize methods that we are dealing with absolute addresses
*/

class CkDDT_DataType {

  protected:
    int datatype;
    int refCount;

  protected:
    int size;
    CkDDT_Aint extent;
    int count;
    CkDDT_Aint lb;
    CkDDT_Aint ub;
    bool iscontig;
    int baseSize;
    CkDDT_Aint baseExtent;
    CkDDT_DataType *baseType;
    int baseIndex;
    int numElements;
    char name[CkDDT_MAX_NAME_LEN];
    int nameLen;
    bool isAbsolute;

  private:
    CkDDT_DataType(const CkDDT_DataType& obj);
    CkDDT_DataType& operator=(const CkDDT_DataType& obj);

  public:
    CkDDT_DataType() { }
    virtual ~CkDDT_DataType() { }
    CkDDT_DataType(int type);
    CkDDT_DataType(int datatype, int size, CkDDT_Aint extent, int count, CkDDT_Aint lb, CkDDT_Aint ub,
            bool iscontig, int baseSize, CkDDT_Aint baseExtent,
            CkDDT_DataType* baseType, int numElements, int baseIndex);
    CkDDT_DataType(const CkDDT_DataType &obj, CkDDT_Aint _lb, CkDDT_Aint _extent);

    virtual bool isContig(void) const;
    virtual int getSize(int count=1) const;
    virtual CkDDT_Aint getExtent(void) const;
    virtual int getBaseSize(void) const;
    virtual CkDDT_Aint getLB(void) const;
    virtual CkDDT_Aint getUB(void) const;
    virtual int getType(void) const;
    virtual int getNumElements(void) const;
    virtual void inrRefCount(void) ;
    virtual int getRefCount(void) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt) ;

    virtual int getEnvelope(int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) const;
    virtual int getContents(int max_integers, int max_addresses, int max_datatypes,
                           int array_of_integers[], CkDDT_Aint array_of_addresses[], int array_of_datatypes[]) const;
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;

    void setName(const char *src);
    void getName(char *dest, int *len) const;
    void setAbsolute(bool arg);
};

/*
  This class maintains the type Contiguous.
  It constructs a typemap consisting of the
  replication of a datatype into contiguous locations.
*/

class CkDDT_Contiguous : public CkDDT_DataType {

 private:
  CkDDT_Contiguous(const CkDDT_Contiguous& obj);
  CkDDT_Contiguous& operator=(const CkDDT_Contiguous& obj);

 public:
  CkDDT_Contiguous() { };
  CkDDT_Contiguous(int count, int index, CkDDT_DataType* oldType);
  virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
  virtual void pupType(PUP::er &p, CkDDT* ddt) ;
  virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
  virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
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

class CkDDT_Vector : public CkDDT_DataType {

  protected:
    int blockLength ;
    int strideLength ;

  private:
    CkDDT_Vector(const CkDDT_Vector& obj);
    CkDDT_Vector& operator=(const CkDDT_Vector& obj);

  public:
    CkDDT_Vector(int count, int blklen, int stride, int index,
                CkDDT_DataType* type);
    CkDDT_Vector() { } ;
    ~CkDDT_Vector() { } ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
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

class CkDDT_HVector : public CkDDT_Vector {

  private:
    CkDDT_HVector(const CkDDT_HVector& obj) ;
    CkDDT_HVector& operator=(const CkDDT_HVector& obj);

  public:
    CkDDT_HVector() { } ;
    CkDDT_HVector(int nCount,int blength,int strideLen,int index,
                CkDDT_DataType* type);
    ~CkDDT_HVector() { } ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
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

class CkDDT_Indexed : public CkDDT_DataType {

  protected:
    int* arrayBlockLength ;
    CkDDT_Aint* arrayDisplacements ;

  private:
    CkDDT_Indexed(const CkDDT_Indexed& obj);
    CkDDT_Indexed& operator=(const CkDDT_Indexed& obj) ;

  public:
    CkDDT_Indexed(int count, int* arrBlock, CkDDT_Aint* arrDisp, int index,
                CkDDT_DataType* type);
    CkDDT_Indexed() { } ;
    ~CkDDT_Indexed() ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
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
class CkDDT_HIndexed : public CkDDT_Indexed {

  private:
    CkDDT_HIndexed(const CkDDT_HIndexed& obj);
    CkDDT_HIndexed& operator=(const CkDDT_HIndexed& obj);

  public:
    CkDDT_HIndexed() { } ;
    CkDDT_HIndexed(int count, int* arrBlock, CkDDT_Aint* arrDisp, int index,
                 CkDDT_DataType* type);
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
};

/*
  The Indexed_Block type allows one to specify a noncontiguous data
  layout where displacements between
  successive blocks need not be equal.
  This allows one to gather arbitrary entries from an array
  and make a single buffer out of it.
  All block displacements are measured  in units of oldtype extent.
  The only difference between this Datatype and CkDDT_Indexed is the fact that
    all blockLengths are the same here, so there is no array of BlockLengths

  BlockLength - the length of all blocks
  arrayDisplacements - holds the array of displacements.
*/

class CkDDT_Indexed_Block : public CkDDT_DataType
{

  protected:
    int BlockLength;
    CkDDT_Aint *arrayDisplacements;

  private:
    CkDDT_Indexed_Block(const CkDDT_Indexed_Block &obj);
    CkDDT_Indexed_Block& operator=(const CkDDT_Indexed_Block &obj);

  public:
    CkDDT_Indexed_Block(int count, int Blength, CkDDT_Aint *ArrDisp, int index, CkDDT_DataType *type);
    CkDDT_Indexed_Block() { };
    ~CkDDT_Indexed_Block() ;
    virtual size_t serialize(char *userdata, char *buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT *ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
};

/*
  The HIndexed_Block type allows one to specify a noncontiguous data
  layout where displacements between
  successive blocks need not be equal.
  Unlike in Indexed_Block type, these displacements are now specified in bytes as ap
  This allows one to gather arbitrary entries from an array
  and make a single buffer out of it.
  All block displacements are measured  in units of oldtype extent.
  The only difference between this Datatype and CkDDT_Indexed is the fact that
    all blockLengths are the same here, so there is no array of BlockLengths

  BlockLength - the length of all blocks
  arrayDisplacements - holds the array of displacements.
*/

class CkDDT_HIndexed_Block : public CkDDT_Indexed_Block
{
  
  private:
    CkDDT_HIndexed_Block(const CkDDT_Indexed_Block &obj);
    CkDDT_HIndexed_Block& operator=(const CkDDT_Indexed_Block &obj);

  public:
    CkDDT_HIndexed_Block(int count, int Blength, CkDDT_Aint *ArrDisp, int index, CkDDT_DataType *type);
    CkDDT_HIndexed_Block() { };
    ~CkDDT_HIndexed_Block() ;
    virtual size_t serialize(char *userdata, char *buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT *ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
};

/*
  CkDDT_Struct is the most general type constructor.
  It further generalizes CkDDT_HIndexed in
  that it allows each block to consist of replications of
  different datatypes.
  The intent is to allow descriptions of arrays of structures,
  as a single datatype.

  arrayBlockLength - array of block lengths
  arrayDisplacements - array of displacements
  arrayDataType - array of DataTypes.
*/

class CkDDT_Struct : public CkDDT_DataType {

  protected:
    int* arrayBlockLength ;
    CkDDT_Aint* arrayDisplacements ;
    int* index;
    CkDDT_DataType** arrayDataType;

  private:
    CkDDT_Struct(const CkDDT_Struct& obj);
    CkDDT_Struct& operator=(const CkDDT_Struct& obj);

  public:
    CkDDT_Struct() { } ;
    CkDDT_Struct(int count, int* arrBlock, CkDDT_Aint* arrDisp, int *index,
               CkDDT_DataType **type);
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], CkDDT_Aint a[], int d[]) const;
};

/*
  This class maintains the typeTable of the derived datatypes.
  First few entries of the table contain primitive datatypes.
  index - holds the current available index in the table where 
          new datatype can be allocated.
  typeTable - holds the table of CkDDT_DataType

  Type_Contiguous - 
  Type_Vector -
  Type_HVector -
  Type_Indexed - 
  Type_HIndexed -
  Type_Struct - builds the new type 
                Contiguous/Vector/Hvector/Indexed/HIndexed/Struct  from the old
                Type provided and stores the new type in the table.
*/

class CkDDT {
  private:
    CkDDT_DataType**  typeTable;
    int*  types; //used for pup
    int max_types;
    int num_types;

  public:

  CkDDT(void*) {} // emulates migration constructor
  CkDDT()
  {
    max_types = CkDDT_MAXTYPE;
    typeTable = new CkDDT_DataType*[max_types];
    types = new int[max_types];
    typeTable[0] = new CkDDT_DataType(CkDDT_DOUBLE);
    types[0] = CkDDT_DOUBLE;
    typeTable[1] = new CkDDT_DataType(CkDDT_INT);
    types[1] = CkDDT_INT;
    typeTable[2] = new CkDDT_DataType(CkDDT_FLOAT);
    types[2] = CkDDT_FLOAT;
    typeTable[3] = new CkDDT_DataType(CkDDT_COMPLEX);
    types[3] = CkDDT_COMPLEX;
    typeTable[4] = new CkDDT_DataType(CkDDT_LOGICAL);
    types[4] = CkDDT_LOGICAL;
    typeTable[5] = new CkDDT_DataType(CkDDT_C_BOOL);
    types[5] = CkDDT_C_BOOL;
    typeTable[6] = new CkDDT_DataType(CkDDT_CHAR);
    types[6] = CkDDT_CHAR;
    typeTable[7] = new CkDDT_DataType(CkDDT_BYTE);
    types[7] = CkDDT_BYTE;
    typeTable[8] = new CkDDT_DataType(CkDDT_PACKED);
    types[8] = CkDDT_PACKED;
    typeTable[9] = new CkDDT_DataType(CkDDT_SHORT);
    types[9] = CkDDT_SHORT;
    typeTable[10] = new CkDDT_DataType(CkDDT_LONG);
    types[10] = CkDDT_LONG;
    typeTable[11] = new CkDDT_DataType(CkDDT_UNSIGNED_CHAR);
    types[11] = CkDDT_UNSIGNED_CHAR;
    typeTable[12] = new CkDDT_DataType(CkDDT_UNSIGNED_SHORT);
    types[12] = CkDDT_UNSIGNED_SHORT;
    typeTable[13] = new CkDDT_DataType(CkDDT_UNSIGNED);
    types[13] = CkDDT_UNSIGNED;
    typeTable[14] = new CkDDT_DataType(CkDDT_UNSIGNED_LONG);
    types[14] = CkDDT_UNSIGNED_LONG;
    typeTable[15] = new CkDDT_DataType(CkDDT_LONG_DOUBLE);
    types[15] = CkDDT_LONG_DOUBLE;
    typeTable[16] = new CkDDT_DataType(CkDDT_FLOAT_INT);
    types[16] = CkDDT_FLOAT_INT;
    typeTable[17] = new CkDDT_DataType(CkDDT_DOUBLE_INT);
    types[17] = CkDDT_DOUBLE_INT;
    typeTable[18] = new CkDDT_DataType(CkDDT_LONG_INT);
    types[18] = CkDDT_LONG_INT;
    typeTable[19] = new CkDDT_DataType(CkDDT_2INT);
    types[19] = CkDDT_2INT;
    typeTable[20] = new CkDDT_DataType(CkDDT_SHORT_INT);
    types[20] = CkDDT_SHORT_INT;
    typeTable[21] = new CkDDT_DataType(CkDDT_LONG_DOUBLE_INT);
    types[21] = CkDDT_LONG_DOUBLE_INT;
    typeTable[22] = new CkDDT_DataType(CkDDT_2FLOAT);
    types[22] = CkDDT_2FLOAT;
    typeTable[23] = new CkDDT_DataType(CkDDT_2DOUBLE);
    types[23] = CkDDT_2DOUBLE;
    typeTable[24] = new CkDDT_DataType(CkDDT_LB);
    types[24] = CkDDT_LB;
    typeTable[25] = new CkDDT_DataType(CkDDT_UB);
    types[25] = CkDDT_UB;
    typeTable[26] = new CkDDT_DataType(CkDDT_LONG_LONG_INT);
    types[26] = CkDDT_LONG_LONG_INT;
    typeTable[27] = new CkDDT_DataType(CkDDT_DOUBLE_COMPLEX);
    types[27] = CkDDT_DOUBLE_COMPLEX;
    typeTable[28] = new CkDDT_DataType(CkDDT_SIGNED_CHAR);
    types[28] = CkDDT_SIGNED_CHAR;
    typeTable[29] = new CkDDT_DataType(CkDDT_UNSIGNED_LONG_LONG);
    types[29] = CkDDT_UNSIGNED_LONG_LONG;
    typeTable[30] = new CkDDT_DataType(CkDDT_WCHAR);
    types[30] = CkDDT_WCHAR;
    typeTable[31] = new CkDDT_DataType(CkDDT_INT8_T);
    types[31] = CkDDT_INT8_T;
    typeTable[32] = new CkDDT_DataType(CkDDT_INT16_T);
    types[32] = CkDDT_INT16_T;
    typeTable[33] = new CkDDT_DataType(CkDDT_INT32_T);
    types[33] = CkDDT_INT32_T;
    typeTable[34] = new CkDDT_DataType(CkDDT_INT64_T);
    types[34] = CkDDT_INT64_T;
    typeTable[35] = new CkDDT_DataType(CkDDT_UINT8_T);
    types[35] = CkDDT_UINT8_T;
    typeTable[36] = new CkDDT_DataType(CkDDT_UINT16_T);
    types[36] = CkDDT_UINT16_T;
    typeTable[37] = new CkDDT_DataType(CkDDT_UINT32_T);
    types[37] = CkDDT_UINT32_T;
    typeTable[38] = new CkDDT_DataType(CkDDT_UINT64_T);
    types[38] = CkDDT_UINT64_T;
    typeTable[39] = new CkDDT_DataType(CkDDT_FLOAT_COMPLEX);
    types[39] = CkDDT_FLOAT_COMPLEX;
    typeTable[40] = new CkDDT_DataType(CkDDT_LONG_DOUBLE_COMPLEX);
    types[40] = CkDDT_LONG_DOUBLE_COMPLEX;
    typeTable[41] = new CkDDT_DataType(CkDDT_AINT);
    types[41] = CkDDT_AINT;
    num_types = 42;

    int i;
    for(i=num_types ; i < max_types; i++)
    {
      typeTable[i] = NULL;
      types[i] = CkDDT_TYPE_NULL ;
    }
  }

  void newContiguous(int count, CkDDT_Type  oldType, CkDDT_Type* newType);
  void newVector(int count, int blocklength, int stride, CkDDT_Type oldtype,
                CkDDT_Type* newtype);
  void newHVector(int count, int blocklength, int stride, CkDDT_Type oldtype,
                 CkDDT_Type* newtype);
  void newIndexed(int count, int* arrbLength, CkDDT_Aint* arrDisp , CkDDT_Type oldtype,
                 CkDDT_Type* newtype);
  void newHIndexed(int count, int* arrbLength, CkDDT_Aint* arrDisp , CkDDT_Type oldtype,
                  CkDDT_Type* newtype);
  void newIndexedBlock(int count, int Blocklength, CkDDT_Aint *arrDisp, CkDDT_Type oldtype,
                      CkDDT_Type *newtype);
  void newHIndexedBlock(int count, int Blocklength, CkDDT_Aint *arrDisp, CkDDT_Type oldtype,
                       CkDDT_Type *newtype);
  void newStruct(int count, int* arrbLength, CkDDT_Aint* arrDisp , CkDDT_Type *oldtype,
                CkDDT_Type* newtype);
  void  freeType(int* index);
  int   getNextFreeIndex(void);
  void  pup(PUP::er &p);
  CkDDT_DataType*  getType(int nIndex) const;

  bool isContig(int nIndex) const;
  int getSize(int nIndex, int count=1) const;
  CkDDT_Aint getExtent(int nIndex) const;
  CkDDT_Aint getLB(int nIndex) const;
  CkDDT_Aint getUB(int nIndex) const;
  int getEnvelope(int nIndex, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) const;
  int getContents(int nIndex, int max_integers, int max_addresses, int max_datatypes,
                 int array_of_integers[], CkDDT_Aint array_of_addresses[], int array_of_datatypes[]) const;
  void setName(int nIndex, const char *name);
  void getName(int nIndex, char *name, int *len) const;
  void createResized(CkDDT_Type oldtype, CkDDT_Aint lb, CkDDT_Aint extent, CkDDT_Type *newtype);
  ~CkDDT() ;
};

#endif
