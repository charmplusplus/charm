#ifndef __CkDDT_H_
#define __CkDDT_H_

#include <string>
#include <vector>
#include "charm++.h"
#include "ampi.h"

//Uncomment for debug print statements
#define DDTDEBUG(...) //CkPrintf(__VA_ARGS__)

using std::vector;
using std::string;

/*
  An MPI basic datatype is a type that corresponds
  to the basic datatypes of the host language
    MPI_Get_elements returns the number of the
    basic types in a more complex datatype such
    as an MPI struct or MPI vector
*/
#define CkDDT_MAX_BASIC_TYPE      27

/*
  CkDDT_MAX_PRIMITIVE_TYPE indicates the highest
  datatype defined in the MPI standard
    Do not free datatypes less than or equal to
    this value
*/
#define CkDDT_MAX_PRIMITIVE_TYPE  41

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

enum CkDDT_Dir : bool {
  PACK   = true,
  UNPACK = false
};

/* Serialize a contiguous chunk of memory */
inline void serializeContig(char* userdata, char* buffer, size_t size, CkDDT_Dir dir)
{
  if (dir == PACK) {
    memcpy(buffer, userdata, size);
  } else {
    memcpy(userdata, buffer, size);
  }
}

/* Helper function to set names (used by AMPI too).
 * Leading whitespaces are significant, trailing spaces are not. */
inline void CkDDT_SetName(string &dst, const char *src)
{
  int end = strlen(src)-1;
  while ((end>0) && (src[end]==' '))
    end--;
  int len = (end==0) ? 0 : end+1;
  if (len > MPI_MAX_OBJECT_NAME) len = MPI_MAX_OBJECT_NAME;
  dst.assign(src, len);
}

class CkDDT;

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

  Methods:

  getSize -  returns the size of the datatype.
  getExtent - returns the extent of the datatype.

  incRefCount - increment the reference count.
  decRefCount - decrement the reference count.

  serialize - This is the function which actually copies the contents from
    user's space to buffer if dir=PACK or reverse if dir=UNPACK
    according to the datatype.

  setName - set the name of datatype
  getName - get the name of datatype
  setAbsolute - tells DDT's serialize methods that we are dealing with absolute addresses

  Reference Counting currently disabled. To add this feature back in, the refcount variable
    cannot be copied when making a duplicate.
*/

class CkDDT_DataType {

  protected:
    bool iscontig;
    bool isAbsolute;
    int size;
    int count;
    int datatype;
    int refCount;
    int baseSize;
    int baseIndex;
    int numContigBlocks;
    int numElements;
    MPI_Aint extent;
    MPI_Aint ub;
    MPI_Aint lb;
    MPI_Aint trueExtent;
    MPI_Aint trueLB;
    MPI_Aint baseExtent;
    CkDDT_DataType *baseType;
    string name;

  public:
    CkDDT_DataType& operator=(const CkDDT_DataType& obj);
    CkDDT_DataType() = default;
    virtual ~CkDDT_DataType() = default;
    CkDDT_DataType(int type);
    CkDDT_DataType(int datatype, int size, MPI_Aint extent, int count, MPI_Aint lb, MPI_Aint ub,
            bool iscontig, int baseSize, MPI_Aint baseExtent, CkDDT_DataType* baseType,
            int numElements, int baseIndex, MPI_Aint trueExtent, MPI_Aint trueLB);
    CkDDT_DataType(const CkDDT_DataType &obj, MPI_Aint _lb, MPI_Aint _extent);
    CkDDT_DataType(const CkDDT_DataType& obj);

    void setSize(MPI_Aint lb, MPI_Aint extent);
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) const;
    virtual int getContents(int max_integers, int max_addresses, int max_datatypes,
                           int array_of_integers[], MPI_Aint array_of_addresses[], int array_of_datatypes[]) const;

    virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const
    {
      DDTDEBUG("CkDDT_Datatype::serialize %s %d objects of type %d (%d bytes)\n",
               (dir==PACK)?"packing":"unpacking", num, datatype, bufSize);
      size_t bufSize = std::min((size_t)num * (size_t)size, (size_t)msgLength);
      if (iscontig) {
        serializeContig(userdata, buffer, bufSize, dir);
        return bufSize;
      }
      else {
        for (int i=0; i<num; i++) {
          if(bufSize < size) {
            break;
          }
          serializeContig(userdata + i*extent, buffer + i*size, size, dir);
          bufSize -= size;
        }
        return msgLength - bufSize;
      }
    }

    virtual bool isContig() const { return iscontig; }
    virtual int getSize(int count=1) const { return count * size; }
    virtual MPI_Aint getExtent() const { return extent; }
    virtual int getBaseSize() const { return baseSize; }
    virtual MPI_Aint getLB() const { return lb; }
    virtual MPI_Aint getUB() const { return ub; }
    virtual MPI_Aint getTrueExtent() const { return trueExtent; }
    virtual MPI_Aint getTrueLB() const { return trueLB; }
    virtual int getBaseIndex() const { return baseIndex; }
    virtual CkDDT_DataType* getBaseType() const { return baseType; }
    virtual MPI_Aint getBaseExtent() const { return baseExtent; }
    virtual int getCount() const { return count; }
    virtual int getType() const { return datatype; }
    virtual int getNumElements() const { return numElements; }
    virtual int getNumBasicElements(int bytes) const;
    virtual int getNumContigBlocks() const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
    virtual void incRefCount() {
      CkAssert(refCount > 0);
      if (datatype > CkDDT_MAX_PRIMITIVE_TYPE)
        refCount++;
    }
    virtual int decRefCount() {
      // Callers of this function should always check its return
      // value and free the type only if it returns 0.
      CkAssert(refCount > 0);
      if (datatype > CkDDT_MAX_PRIMITIVE_TYPE)
        return --refCount;
      return -1;
    }
    inline void setName(const char *src) { CkDDT_SetName(name, src); }
    inline void getName(char *dest, int *len) const {
      int length = *len = name.size();
      memcpy(dest, &name[0], length);
      dest[length] = '\0';
    }
    inline void setAbsolute(bool arg) { isAbsolute = arg; }
};

/*
  This class maintains the type Contiguous.
  It constructs a typemap consisting of the
  replication of a datatype into contiguous locations.
*/

class CkDDT_Contiguous : public CkDDT_DataType {

 private:
  CkDDT_Contiguous& operator=(const CkDDT_Contiguous& obj);

 public:
  CkDDT_Contiguous() { };
  CkDDT_Contiguous(int count, int index, CkDDT_DataType* oldType);
  virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const;
  virtual void pupType(PUP::er &p, CkDDT* ddt) ;
  virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
  virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
  virtual int getNumBasicElements(int bytes) const;
  virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
  virtual size_t getNumMsgBlocks();
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
    CkDDT_Vector& operator=(const CkDDT_Vector& obj);

  public:
    CkDDT_Vector(int count, int blklen, int stride, int index,
                CkDDT_DataType* type);
    CkDDT_Vector(const CkDDT_Vector &obj, MPI_Aint _lb, MPI_Aint _extent);
    CkDDT_Vector() { } ;
    ~CkDDT_Vector() { } ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
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
    CkDDT_HVector& operator=(const CkDDT_HVector& obj);

  public:
    CkDDT_HVector() { } ;
    CkDDT_HVector(int nCount,int blength,int strideLen,int index,
                CkDDT_DataType* type);
    CkDDT_HVector(const CkDDT_HVector &obj, MPI_Aint _lb, MPI_Aint _extent);
    ~CkDDT_HVector() { } ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
    // virtual size_t getMsgAddresses();
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

class CkDDT_HIndexed : public CkDDT_DataType {

  protected:
    vector<int> arrayBlockLength;
    vector<MPI_Aint> arrayDisplacements;

  private:
    CkDDT_HIndexed& operator=(const CkDDT_HIndexed& obj);

  public:
    CkDDT_HIndexed(int count, const int* arrBlock, const MPI_Aint* arrBytesDisp, int index,
                CkDDT_DataType* type);
    // CkDDT_HIndexed(const CkDDT_HIndexed &obj, MPI_Aint _lb, MPI_Aint _extent);
    CkDDT_HIndexed() { } ;
    ~CkDDT_HIndexed() ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
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

class CkDDT_Indexed : public CkDDT_HIndexed {

  private:
    CkDDT_Indexed& operator=(const CkDDT_Indexed& obj) ;

  public:
    CkDDT_Indexed() { } ;
    CkDDT_Indexed(int count, const int* arrBlock, const MPI_Aint* arrBytesDisp,
      const MPI_Aint* arrDisp, int index, CkDDT_DataType* type);
    CkDDT_Indexed(const CkDDT_Indexed &obj, MPI_Aint _lb, MPI_Aint _extent);
    virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
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

class CkDDT_HIndexed_Block : public CkDDT_DataType
{
  protected:
    int BlockLength;
    // The MPI Standard has arrDisp as an array of int's to MPI_Type_create_indexed_block, but
    // as an array of MPI_Aint's to MPI_Type_create_hindexed_block, so we store it as Aint's
    // internally and convert from int to Aint in Indexed_Block's constructor:
    vector<MPI_Aint> arrayDisplacements;

  private:
    CkDDT_HIndexed_Block& operator=(const CkDDT_HIndexed_Block &obj);

  public:
    CkDDT_HIndexed_Block(int count, int Blength, const MPI_Aint *arrBytesDisp, int index, CkDDT_DataType *type);
    CkDDT_HIndexed_Block() { };
    CkDDT_HIndexed_Block(const CkDDT_HIndexed_Block &obj, MPI_Aint _lb, MPI_Aint _extent);
    ~CkDDT_HIndexed_Block() ;
    virtual size_t serialize(char *userdata, char *buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual void pupType(PUP::er &p, CkDDT *ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
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

class CkDDT_Indexed_Block : public CkDDT_HIndexed_Block
{

  private:
    CkDDT_Indexed_Block& operator=(const CkDDT_Indexed_Block &obj);

  public:
    CkDDT_Indexed_Block(int count, int Blength, const MPI_Aint *arrBytesDisp, const int *ArrDisp, int index, CkDDT_DataType *type);
    CkDDT_Indexed_Block() { };
    CkDDT_Indexed_Block(const CkDDT_Indexed_Block &obj, MPI_Aint _lb, MPI_Aint _extent);
    ~CkDDT_Indexed_Block() ;
    virtual size_t serialize(char *userdata, char *buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual void pupType(PUP::er &p, CkDDT *ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
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
    vector<int> arrayBlockLength;
    vector<MPI_Aint> arrayDisplacements;
    vector<int> index;
    vector<CkDDT_DataType*> arrayDataType;

  private:
    CkDDT_Struct& operator=(const CkDDT_Struct& obj);

  public:
    CkDDT_Struct() { } ;
    CkDDT_Struct(int count, const int* arrBlock, const MPI_Aint* arrDisp, const int *index,
               CkDDT_DataType **type);
    CkDDT_Struct(const CkDDT_Struct &obj, MPI_Aint _lb, MPI_Aint _extent);
    virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual const vector<int>& getBaseIndices() const;
    virtual const vector<CkDDT_DataType*>& getBaseTypes() const;
    virtual int getNumBasicElements(int bytes) const;
    virtual size_t getAddresses(char* userdata, char** addresses, int* bLengths) const;
    virtual size_t getNumMsgBlocks();
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
  Type_Indexed_Block -
  Type_HIndexed -
  Type_HIndexed_Block -
  Type_Struct - builds the new type
                Contiguous/Vector/Hvector/Indexed/HIndexed/Struct  from the old
                Type provided and stores the new type in the table.
*/

class CkDDT {
  private:
    vector<CkDDT_DataType*> typeTable;
    vector<int> types; //used for pup

  void addBasic(int type) {
    CkAssert(types.size() > type && types[type] == MPI_DATATYPE_NULL);
    typeTable[type]               = new CkDDT_DataType(type);
    types[type]                   = type;
  }

  void addStruct(const char* name, int type, int val, int idx, int offset) {
    CkAssert(types.size() > type && types[type] == MPI_DATATYPE_NULL);
    const int bLengths[2]           = {1, 1};
    MPI_Datatype bTypes[2]          = {val, idx};
    CkDDT_DataType* nTypes[2]       = {getType(val), getType(idx)};
    MPI_Aint offsets[2]             = {0, offset};
    typeTable[type]                 = new CkDDT_Struct(2, bLengths, offsets, bTypes, nTypes);
    typeTable[type]->setName(name);
    types[type]                     = CkDDT_STRUCT;
  }

  public:

  CkDDT(void*) {} // emulates migration constructor
  CkDDT(void) : typeTable(CkDDT_MAX_PRIMITIVE_TYPE+1, nullptr), types(CkDDT_MAX_PRIMITIVE_TYPE+1, MPI_DATATYPE_NULL)
  {
    addBasic(MPI_DOUBLE);
    addBasic(MPI_INT);
    addBasic(MPI_FLOAT);
    addBasic(MPI_LOGICAL);
    addBasic(MPI_C_BOOL);
    addBasic(MPI_CHAR);
    addBasic(MPI_BYTE);
    addBasic(MPI_PACKED);
    addBasic(MPI_SHORT);
    addBasic(MPI_LONG);
    addBasic(MPI_UNSIGNED_CHAR);
    addBasic(MPI_UNSIGNED_SHORT);
    addBasic(MPI_UNSIGNED);
    addBasic(MPI_UNSIGNED_LONG);
    addBasic(MPI_LONG_DOUBLE);
    addBasic(MPI_LONG_LONG_INT);
    addBasic(MPI_SIGNED_CHAR);
    addBasic(MPI_UNSIGNED_LONG_LONG);
    addBasic(MPI_WCHAR);
    addBasic(MPI_INT8_T);
    addBasic(MPI_INT16_T);
    addBasic(MPI_INT32_T);
    addBasic(MPI_INT64_T);
    addBasic(MPI_UINT8_T);
    addBasic(MPI_UINT16_T);
    addBasic(MPI_UINT32_T);
    addBasic(MPI_UINT64_T);
    addBasic(MPI_AINT);
    addBasic(MPI_LB);
    addBasic(MPI_UB);

    /*
      Following types have multiple elements, for serialize to know where to write data
      the following types must be inserted as structs
    */

    // Contiguous
    typedef struct { int val; int idx; } IntInt;
    addStruct("MPI_2INT", MPI_2INT, MPI_INT, MPI_INT, offsetof(IntInt, idx));

    typedef struct { float val; float idx; } FloatFloat;
    addStruct("MPI_2FLOAT", MPI_2FLOAT, MPI_FLOAT, MPI_FLOAT, offsetof(FloatFloat, idx));

    typedef struct { double val; double idx; } DoubleDouble;
    addStruct("MPI_2DOUBLE", MPI_2DOUBLE, MPI_DOUBLE, MPI_DOUBLE, offsetof(DoubleDouble, idx));

    typedef struct { float val; int idx; } FloatInt;
    addStruct("MPI_FLOAT_INT", MPI_FLOAT_INT, MPI_FLOAT, MPI_INT, offsetof(FloatInt, idx));
    // Not Contiguous

    typedef struct { double val; int idx; } DoubleInt;
    addStruct("MPI_DOUBLE_INT", MPI_DOUBLE_INT, MPI_DOUBLE, MPI_INT, offsetof(DoubleInt, idx));

    typedef struct { long val; int idx; } LongInt;
    addStruct("MPI_LONG_INT", MPI_LONG_INT, MPI_LONG, MPI_INT, offsetof(LongInt, idx));

    typedef struct { short val; int idx; } ShortInt;
    addStruct("MPI_SHORT_INT", MPI_SHORT_INT, MPI_SHORT, MPI_INT, offsetof(ShortInt, idx));

    typedef struct { long double val; int idx; } LongdoubleInt;
    addStruct("MPI_LONG_DOUBLE_INT", MPI_LONG_DOUBLE_INT, MPI_LONG_DOUBLE, MPI_INT, offsetof(LongdoubleInt, idx));

    // Complex datatypes
    typedef struct { float val; float idx; } FloatComplex;
    addStruct("MPI_FLOAT_COMPLEX", MPI_FLOAT_COMPLEX, MPI_FLOAT, MPI_FLOAT, offsetof(FloatComplex, idx));
    addStruct("MPI_COMPLEX", MPI_COMPLEX      , MPI_FLOAT, MPI_FLOAT, offsetof(FloatComplex, idx));

    typedef struct { double val; double idx; } DoubleComplex;
    addStruct("MPI_DOUBLE_COMPLEX", MPI_DOUBLE_COMPLEX, MPI_DOUBLE, MPI_DOUBLE, offsetof(DoubleComplex, idx));

    typedef struct { long double val; long double idx; } LongDoubleComplex;
    addStruct("MPI_LONG_DOUBLE_COMPLEX", MPI_LONG_DOUBLE_COMPLEX, MPI_LONG_DOUBLE, MPI_LONG_DOUBLE, offsetof(LongDoubleComplex, idx));

  }

  void newContiguous(int count, MPI_Datatype  oldType, MPI_Datatype* newType);
  void newVector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                MPI_Datatype* newtype);
  void newHVector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                 MPI_Datatype* newtype);
  void newIndexed(int count, const int* arrbLength, MPI_Aint* arrDisp , MPI_Datatype oldtype,
                 MPI_Datatype* newtype);
  void newHIndexed(int count, const int* arrbLength, const MPI_Aint* arrDisp , MPI_Datatype oldtype,
                  MPI_Datatype* newtype);
  void newIndexedBlock(int count, int Blocklength, const int *arrDisp, MPI_Datatype oldtype,
                      MPI_Datatype *newtype);
  void newHIndexedBlock(int count, int Blocklength, const MPI_Aint *arrDisp, MPI_Datatype oldtype,
                       MPI_Datatype *newtype);
  void newStruct(int count, const int* arrbLength, const MPI_Aint* arrDisp , const MPI_Datatype *oldtype,
                MPI_Datatype* newtype);
  int insertType(CkDDT_DataType* ptr, int type);
  void freeType(int index);
  void pup(PUP::er &p);
  void createDup(int nIndexOld, int *nIndexNew);
  void createResized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype);
  int getEnvelope(int nIndex, int *num_integers, int *num_addresses, int *num_datatypes,
                  int *combiner) const;
  int getContents(int nIndex, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[],
                  MPI_Aint array_of_addresses[], int array_of_datatypes[]);
  ~CkDDT();

  inline CkDDT_DataType* getType(int nIndex) const {
    #if CMK_ERROR_CHECKING
    if (nIndex < 0 || nIndex > typeTable.size())
      CkAbort("AMPI> invalid datatype index passed to getType!");
    #endif
    return typeTable[nIndex];
  }

  inline bool isContig(int nIndex) const { return getType(nIndex)->isContig(); }
  inline int getSize(int nIndex, int count=1) const { return count * getType(nIndex)->getSize(); }
  inline MPI_Aint getExtent(int nIndex) const { return getType(nIndex)->getExtent(); }
  inline MPI_Aint getLB(int nIndex) const { return getType(nIndex)->getLB(); }
  inline MPI_Aint getUB(int nIndex) const { return getType(nIndex)->getUB(); }
  inline MPI_Aint getTrueExtent(int nIndex) const { return getType(nIndex)->getTrueExtent(); }
  inline MPI_Aint getTrueLB(int nIndex) const { return getType(nIndex)->getTrueLB(); }
  inline void setName(int nIndex, const char *name) { getType(nIndex)->setName(name); }
  inline void getName(int nIndex, char *name, int *len) const { getType(nIndex)->getName(name, len); }
};

#endif
