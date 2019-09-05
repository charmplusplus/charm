#ifndef __CkDDT_H_
#define __CkDDT_H_

#include <string>
#include <vector>
#include <array>
#include <queue>
#include <functional>
#include "charm++.h"
#include "ampi.h"

//Uncomment for debug print statements
#define DDTDEBUG(...) //CkPrintf(__VA_ARGS__)

/*
 * An MPI basic datatype is a type that corresponds to the basic
 * datatypes of the host language (C/C++/Fortran).
 *
 * MPI_Get_elements returns the number of the basic types in a
 * more complex datatype such as an MPI struct or vector type.
 */
#define AMPI_MAX_BASIC_TYPE      29

/*
 * AMPI_MAX_PREDEFINED_TYPE indicates the highest
 * datatype defined in the MPI standard.
 *
 * Note: do not free datatypes less than or equal to this value.
 */
#define AMPI_MAX_PREDEFINED_TYPE  41

/*
 * These are the different kinds of MPI derived datatypes
 * supported natively by DDT, in order from least (contiguous)
 * to most complex (struct):
 *
 * Note: DDT doesn't directly implement Subarray and Darray,
 *       which ROMIO implements for us.
 */
#define CkDDT_CONTIGUOUS          42
#define CkDDT_VECTOR              43
#define CkDDT_HVECTOR             44
#define CkDDT_INDEXED_BLOCK       45
#define CkDDT_HINDEXED_BLOCK      46
#define CkDDT_INDEXED             47
#define CkDDT_HINDEXED            48
#define CkDDT_STRUCT              49
#define CkDDT_FIRST_USER_TYPE     50

enum CkDDT_Dir : bool {
  PACK   = true,
  UNPACK = false
};

/* Serialize a contiguous chunk of memory */
inline void serializeContig(char* userdata, char* buffer, size_t size, CkDDT_Dir dir) noexcept
{
  if (dir == PACK) {
    memcpy(buffer, userdata, size);
  }
  else {
    memcpy(userdata, buffer, size);
  }
}

/* Helper function to set names (used by AMPI too).
 * Leading whitespaces are significant, trailing spaces are not. */
inline void CkDDT_SetName(std::string &dst, const char *src) noexcept
{
  int end = strlen(src)-1;
  while ((end>0) && (src[end]==' ')) {
    end--;
  }
  int len = (end==0) ? 0 : end+1;
  if (len > MPI_MAX_OBJECT_NAME) {
    len = MPI_MAX_OBJECT_NAME;
  }
  dst.assign(src, len);
}

class CkDDT;

/*
 * This class maintains the data for all "basic" datatypes, and
 * also acts as the base class for all derived datatypes.
 *
 * The serialize method implements packing and unpacking to/from
 * a contiguous buffer, such as an AMPI message buffer.
 *
 * iscontig - can serialization of this type be optimized for contiguity?
 * isAbsolute - is this typeused for a call with MPI_BOTTOM?
 *
 * size - size of one unit of datatype
 * count -  count of base datatype present in this datatype
 * datatype - used for primitive datatypes for size calculations
 * refCount - to keep track of how many references are present to this type.
 * baseSize - size of base datatype
 * baseIndex - type index of base datatype
 * numElements - number of elements the type is composed of
 *
 * extent - extent is different from size in that it also counts
 *          displacements between blocks
 * ub - upper bound of the type
 * lb - lower bound of the type
 * trueExtent - different from size and extent in that it is the minimum size
 *              of a buffer that can hold a deserialized version of the type
 * trueLB - the lowest/first byte that will actually be serialized of the type
 * baseExtent - extent of base datatype
 *
 * baseType - pointer to the base datatype
 * name - user specified name for datatype
 */
class CkDDT_DataType
{
 protected:
  bool iscontig;
  bool isAbsolute;
  int size;
  int count;
  int datatype;
  int refCount;
  int baseSize;
  int baseIndex;
  int numElements;
  MPI_Aint extent;
  MPI_Aint ub;
  MPI_Aint lb;
  MPI_Aint trueExtent;
  MPI_Aint trueLB;
  MPI_Aint baseExtent;
  CkDDT_DataType *baseType;
  std::unordered_map<int, uintptr_t> attributes;
  std::string name;

 public:
  CkDDT_DataType() = default;
  virtual ~CkDDT_DataType() = default;
  CkDDT_DataType(int type) noexcept;
  CkDDT_DataType(int datatype, int size, MPI_Aint extent, int count, MPI_Aint lb, MPI_Aint ub,
                 bool iscontig, int baseSize, MPI_Aint baseExtent, CkDDT_DataType* baseType,
                 int numElements, int baseIndex, MPI_Aint trueExtent, MPI_Aint trueLB) noexcept;
  CkDDT_DataType(const CkDDT_DataType &obj, MPI_Aint _lb=0, MPI_Aint _extent=0) noexcept;
  CkDDT_DataType& operator=(const CkDDT_DataType& obj) noexcept;

  virtual void pupType(PUP::er &p, CkDDT* ddt) noexcept;
  virtual int getEnvelope(int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) const noexcept;
  virtual int getContents(int max_integers, int max_addresses, int max_datatypes,
                          int array_of_integers[], MPI_Aint array_of_addresses[],
                          int array_of_datatypes[]) const noexcept;

  virtual size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept
  {
    size_t bufSize = std::min((size_t)num * (size_t)size, (size_t)msgLength);
    DDTDEBUG("CkDDT_Datatype::serialize %s %d objects of type %d (%d bytes)\n",
             (dir==PACK)?"packing":"unpacking", num, datatype, bufSize);
    if (iscontig) {
      serializeContig(userdata, buffer, bufSize, dir);
      return bufSize;
    }
    else {
      for (int i=0; i<num; i++) {
        if (bufSize < size) {
          break;
        }
        serializeContig(userdata + i*extent, buffer + i*size, size, dir);
        bufSize -= size;
      }
      return msgLength - bufSize;
    }
  }
  virtual int getNumBasicElements(int bytes) const noexcept;

  void setSize(MPI_Aint lb, MPI_Aint extent) noexcept;
  bool isContig() const noexcept { return iscontig; }
  int getSize(int count=1) const noexcept { return count * size; }
  MPI_Aint getExtent() const noexcept { return extent; }
  int getBaseSize() const noexcept { return baseSize; }
  MPI_Aint getLB() const noexcept { return lb; }
  MPI_Aint getUB() const noexcept { return ub; }
  MPI_Aint getTrueExtent() const noexcept { return trueExtent; }
  MPI_Aint getTrueLB() const noexcept { return trueLB; }
  int getBaseIndex() const noexcept { return baseIndex; }
  CkDDT_DataType* getBaseType() const noexcept { return baseType; }
  MPI_Aint getBaseExtent() const noexcept { return baseExtent; }
  int getCount() const noexcept { return count; }
  int getType() const noexcept { return datatype; }
  int getNumElements() const noexcept { return numElements; }
  void incRefCount() noexcept {
    CkAssert(refCount > 0);
    if (datatype > AMPI_MAX_PREDEFINED_TYPE) {
      refCount++;
    }
  }
  int decRefCount() noexcept {
    // Callers of this function should always check its return
    // value and free the type only if it returns 0.
    CkAssert(refCount > 0);
    if (datatype > AMPI_MAX_PREDEFINED_TYPE) {
      return --refCount;
    }
    return -1;
  }
  std::unordered_map<int, uintptr_t> & getAttributes() noexcept { return attributes; }
  void setName(const char *src) noexcept { CkDDT_SetName(name, src); }
  void getName(char *dest, int *len) const noexcept {
    int length = *len = name.size();
    memcpy(dest, &name[0], length);
    dest[length] = '\0';
  }
  std::string getName() const noexcept {return name;}
  std::string getConfig() const noexcept;
  virtual std::string getTypeMap() const noexcept;
  void setAbsolute(bool arg) noexcept { isAbsolute = arg; }
  bool getAbsolute() const noexcept { return isAbsolute; }
};

/*
 * Contiguous constructs a typemap consisting of the
 * replication of a datatype into contiguous locations.
 */
class CkDDT_Contiguous final : public CkDDT_DataType
{
 public:
  CkDDT_Contiguous() = default;
  ~CkDDT_Contiguous() override = default;
  CkDDT_Contiguous& operator=(const CkDDT_Contiguous& obj) noexcept;
  CkDDT_Contiguous(int count, int index, CkDDT_DataType* oldType) noexcept;
  CkDDT_Contiguous(const CkDDT_Contiguous& obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT* ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
  std::string getTypeMap() const noexcept override;
};

/*
 * Vector allows replication of a datatype into locations that consist of
 * equally spaced blocks. Each block is obtained by concatenating the same
 * number of copies of the old datatype. The spacing between blocks is a
 * multiple of the extent of the old datatype.
 */
class CkDDT_Vector : public CkDDT_DataType
{
 protected:
  int blockLength;
  int strideLength;

 public:
  CkDDT_Vector() = default;
  ~CkDDT_Vector() override = default;
  CkDDT_Vector& operator=(const CkDDT_Vector& obj) noexcept;
  CkDDT_Vector(int count, int blklen, int stride, int index, CkDDT_DataType* type) noexcept;
  CkDDT_Vector(const CkDDT_Vector &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT* ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
  std::string getTypeMap() const noexcept override;
};

/*
 * HVector allows replication of a datatype into locations that consist of
 * equally spaced blocks. Each block is obtained by concatenating the same
 * number of copies of the old datatype. The Vector type assumes that the
 * stride between successive blocks is a multiple of the oldtype's extent,
 * while the HVector type allows a stride which consists of an arbitrary
 * number of bytes. HVector thus derives from Vector.
 */
class CkDDT_HVector final : public CkDDT_Vector
{
 public:
  CkDDT_HVector() = default;
  ~CkDDT_HVector() override = default;
  CkDDT_HVector& operator=(const CkDDT_HVector& obj) noexcept;
  CkDDT_HVector(int nCount, int blength, int strideLen, int index, CkDDT_DataType* type) noexcept;
  CkDDT_HVector(const CkDDT_HVector &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT* ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
  std::string getTypeMap() const noexcept override;
};

/*
 * HIndexed_Block allows one to specify a noncontiguous data layout where
 * displacements between successive blocks need not be equal. This allows one
 * to gather arbitrary entries from an array and make a single buffer out of it.
 * The only difference between this datatype and Indexed is the fact that
 * all block lengths are the same here, so there is no array of block lengths.
 *
 * blockLength - the length of all blocks
 * arrayDisplacements - holds the array of displacements
 */
class CkDDT_HIndexed_Block : public CkDDT_DataType
{
 protected:
  int blockLength;
  // The MPI Standard has arrDisp as an array of int's to MPI_Type_create_indexed_block, but
  // as an array of MPI_Aint's to MPI_Type_create_hindexed_block, so we store it as Aint's
  // internally and convert from int to Aint in Indexed_Block's constructor:
  std::vector<MPI_Aint> arrayDisplacements;

 public:
  CkDDT_HIndexed_Block() = default;
  ~CkDDT_HIndexed_Block() override = default;
  CkDDT_HIndexed_Block& operator=(const CkDDT_HIndexed_Block &obj) noexcept;
  CkDDT_HIndexed_Block(int count, int Blength, const MPI_Aint *arrBytesDisp, int index,
                       CkDDT_DataType *type) noexcept;
  CkDDT_HIndexed_Block(const CkDDT_HIndexed_Block &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char *userdata, char *buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT *ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
  std::string getTypeMap() const noexcept override;
};

/*
 * Indexed_Block allows one to specify a noncontiguous data layout where
 * displacements between successive blocks need not be equal. This allows
 * one to gather arbitrary entries from an array and make a single buffer out
 * of it. All block displacements are measured in units of oldtype extent,
 * otherwise it is the same as HIndexed_Block, so it is derived from HIndexed_Block.
 * The only difference between this datatype and Indexed is the fact that
 * all block lengths are the same here, so there is no array of block lengths.
 */
class CkDDT_Indexed_Block final : public CkDDT_HIndexed_Block
{
 public:
  CkDDT_Indexed_Block() = default;
  ~CkDDT_Indexed_Block() override = default;
  CkDDT_Indexed_Block& operator=(const CkDDT_Indexed_Block &obj) noexcept;
  CkDDT_Indexed_Block(int count, int Blength, const MPI_Aint *arrBytesDisp, const int *ArrDisp,
                      int index, CkDDT_DataType *type) noexcept;
  CkDDT_Indexed_Block(const CkDDT_Indexed_Block &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char *userdata, char *buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT *ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
};

/*
 * HIndexed allows one to specify a noncontiguous data layout where displacements
 * between successive blocks need not be equal. This allows one to gather arbitrary
 * entries from an array and make a single buffer out of it. Unlike Indexed type,
 * block displacements are arbitrary number of bytes.
 *
 * arrayBlockLength - holds the array of block lengths
 * arrayDisplacements - holds the array of displacements
 */
class CkDDT_HIndexed : public CkDDT_DataType
{
 protected:
  std::vector<int> arrayBlockLength;
  std::vector<MPI_Aint> arrayDisplacements;

 public:
  CkDDT_HIndexed() = default;
  ~CkDDT_HIndexed() override = default;
  CkDDT_HIndexed& operator=(const CkDDT_HIndexed& obj) noexcept;
  CkDDT_HIndexed(int count, const int* arrBlock, const MPI_Aint* arrBytesDisp, int index,
                 CkDDT_DataType* type) noexcept;
  CkDDT_HIndexed(const CkDDT_HIndexed &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT* ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
  std::string getTypeMap() const noexcept override;
};

/*
 * Indexed allows one to specify a noncontiguous data layout where displacements
 * between successive blocks need not be equal. This allows one to gather arbitrary
 * entries from an array and make a single buffer out of it. All block displacements
 * are measured in units of oldtype extent, otherwise it is the same as HIndexed,
 * and so it derives from HIndexed.
 */
class CkDDT_Indexed final : public CkDDT_HIndexed
{
 public:
  CkDDT_Indexed() = default;
  ~CkDDT_Indexed() override = default;
  CkDDT_Indexed& operator=(const CkDDT_Indexed& obj) noexcept;
  CkDDT_Indexed(int count, const int* arrBlock, const MPI_Aint* arrBytesDisp,
                const MPI_Aint* arrDisp, int index, CkDDT_DataType* type) noexcept;
  CkDDT_Indexed(const CkDDT_Indexed &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT* ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
};

/*
 * Struct further generalizes CkDDT_HIndexed in that it allows each block to consist of
 * replications of different datatypes. The intent is to allow descriptions of arrays
 * of structures as a single datatype.
 *
 * arrayBlockLength - array of block lengths
 * arrayDisplacements - array of displacements
 * arrayIndex - array of type indices
 * arrayDataType - array of DataTypes
 */
class CkDDT_Struct final : public CkDDT_DataType
{
 protected:
  std::vector<int> arrayBlockLength;
  std::vector<MPI_Aint> arrayDisplacements;
  std::vector<int> index;
  std::vector<CkDDT_DataType *> arrayDataType;

 public:
  CkDDT_Struct() = default;
  ~CkDDT_Struct() override = default;
  CkDDT_Struct& operator=(const CkDDT_Struct& obj) noexcept;
  CkDDT_Struct(int count, const int* arrBlock, const MPI_Aint* arrDisp, const int *index,
               CkDDT_DataType **type, const char* name=nullptr) noexcept;
  CkDDT_Struct(const CkDDT_Struct &obj, MPI_Aint _lb, MPI_Aint _extent) noexcept;

  std::vector<int>& getBaseIndices() noexcept { return index; }
  const std::vector<int>& getBaseIndices() const noexcept { return index; }
  std::vector<CkDDT_DataType *>& getBaseTypes() noexcept { return arrayDataType; }
  const std::vector<CkDDT_DataType *>& getBaseTypes() const noexcept { return arrayDataType; }

  size_t serialize(char* userdata, char* buffer, int num, int msgLength, CkDDT_Dir dir) const noexcept override;
  void pupType(PUP::er &p, CkDDT* ddt) noexcept override;
  int getEnvelope(int *ni, int *na, int *nd, int *combiner) const noexcept override;
  int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const noexcept override;
  int getNumBasicElements(int bytes) const noexcept override;
  std::string getTypeMap() const noexcept override;
};

/*
 * This class maintains the table of all datatypes (predefined and user-defined).
 *
 * predefinedTypeTable - a reference to a const array declared as a static global variable
 *                       (to minimize per-rank memory footprint), which holds the CkDDT_DataType
 *                       object pointers for all predefined types.
 * userTypeTable - a vector that holds the CkDDT_DataType object pointers for all user-defined types
 * freeTypes - a priority queue of freed slot indexes in the userTypeTable, available for reuse.
 */
class CkDDT
{
 private:
  const std::array<const CkDDT_DataType *, AMPI_MAX_PREDEFINED_TYPE+1>& predefinedTypeTable;
  std::vector<CkDDT_DataType *> userTypeTable;
  std::priority_queue<int, std::vector<int>, std::greater<int>> freeTypes;

 public:
  // static methods used by ampi.C for predefined types creation:
  static
  void addBasic(std::array<const CkDDT_DataType *, AMPI_MAX_PREDEFINED_TYPE+1>& predefinedTypeTable_,
                int type) noexcept
  {
    CkAssert(type >= 0);
    CkAssert(type <= AMPI_MAX_BASIC_TYPE);
    CkAssert(type <= AMPI_MAX_PREDEFINED_TYPE);
    predefinedTypeTable_[type] = new CkDDT_DataType(type);
  }

  static
  void addStruct(std::array<const CkDDT_DataType *, AMPI_MAX_PREDEFINED_TYPE+1>& predefinedTypeTable_,
                 const char* name, int type, int val, int idx, int offset) noexcept
  {
    CkAssert(type > AMPI_MAX_BASIC_TYPE);
    CkAssert(type <= AMPI_MAX_PREDEFINED_TYPE);
    const int bLengths[2]      = {1, 1};
    MPI_Datatype bTypes[2]     = {val, idx};
    CkDDT_DataType* nTypes[2]  = {const_cast<CkDDT_DataType *>(predefinedTypeTable_[val]), const_cast<CkDDT_DataType *>(predefinedTypeTable_[idx])};
    MPI_Aint offsets[2]        = {0, offset};
    predefinedTypeTable_[type] = new CkDDT_Struct(2, bLengths, offsets, bTypes, nTypes, name);
  }

  static
  const std::array<const CkDDT_DataType *, AMPI_MAX_PREDEFINED_TYPE+1> createPredefinedTypes() noexcept
  {
    std::array<const CkDDT_DataType *, AMPI_MAX_PREDEFINED_TYPE+1> predefinedTypeTable_;

    addBasic(predefinedTypeTable_, MPI_DOUBLE);
    addBasic(predefinedTypeTable_, MPI_INT);
    addBasic(predefinedTypeTable_, MPI_FLOAT);
    addBasic(predefinedTypeTable_, MPI_LOGICAL);
    addBasic(predefinedTypeTable_, MPI_C_BOOL);
    addBasic(predefinedTypeTable_, MPI_CHAR);
    addBasic(predefinedTypeTable_, MPI_BYTE);
    addBasic(predefinedTypeTable_, MPI_PACKED);
    addBasic(predefinedTypeTable_, MPI_SHORT);
    addBasic(predefinedTypeTable_, MPI_LONG);
    addBasic(predefinedTypeTable_, MPI_UNSIGNED_CHAR);
    addBasic(predefinedTypeTable_, MPI_UNSIGNED_SHORT);
    addBasic(predefinedTypeTable_, MPI_UNSIGNED);
    addBasic(predefinedTypeTable_, MPI_UNSIGNED_LONG);
    addBasic(predefinedTypeTable_, MPI_LONG_DOUBLE);
    addBasic(predefinedTypeTable_, MPI_LONG_LONG_INT);
    addBasic(predefinedTypeTable_, MPI_SIGNED_CHAR);
    addBasic(predefinedTypeTable_, MPI_UNSIGNED_LONG_LONG);
    addBasic(predefinedTypeTable_, MPI_WCHAR);
    addBasic(predefinedTypeTable_, MPI_INT8_T);
    addBasic(predefinedTypeTable_, MPI_INT16_T);
    addBasic(predefinedTypeTable_, MPI_INT32_T);
    addBasic(predefinedTypeTable_, MPI_INT64_T);
    addBasic(predefinedTypeTable_, MPI_UINT8_T);
    addBasic(predefinedTypeTable_, MPI_UINT16_T);
    addBasic(predefinedTypeTable_, MPI_UINT32_T);
    addBasic(predefinedTypeTable_, MPI_UINT64_T);
    addBasic(predefinedTypeTable_, MPI_AINT);
    addBasic(predefinedTypeTable_, MPI_LB);
    addBasic(predefinedTypeTable_, MPI_UB);

    /*
     * The following types have multiple elements, for serialize to know where to write data
     * they must be inserted as CkDDT_Structs:
     */

    // Contiguous:
    typedef struct { int val; int idx; } IntInt;
    addStruct(predefinedTypeTable_, "MPI_2INT", MPI_2INT, MPI_INT, MPI_INT, offsetof(IntInt, idx));

    typedef struct { float val; float idx; } FloatFloat;
    addStruct(predefinedTypeTable_, "MPI_2FLOAT", MPI_2FLOAT, MPI_FLOAT, MPI_FLOAT, offsetof(FloatFloat, idx));

    typedef struct { double val; double idx; } DoubleDouble;
    addStruct(predefinedTypeTable_, "MPI_2DOUBLE", MPI_2DOUBLE, MPI_DOUBLE, MPI_DOUBLE, offsetof(DoubleDouble, idx));

    typedef struct { float val; int idx; } FloatInt;
    addStruct(predefinedTypeTable_, "MPI_FLOAT_INT", MPI_FLOAT_INT, MPI_FLOAT, MPI_INT, offsetof(FloatInt, idx));

    // Non-contiguous:
    typedef struct { double val; int idx; } DoubleInt;
    addStruct(predefinedTypeTable_, "MPI_DOUBLE_INT", MPI_DOUBLE_INT, MPI_DOUBLE, MPI_INT, offsetof(DoubleInt, idx));

    typedef struct { long val; int idx; } LongInt;
    addStruct(predefinedTypeTable_, "MPI_LONG_INT", MPI_LONG_INT, MPI_LONG, MPI_INT, offsetof(LongInt, idx));

    typedef struct { short val; int idx; } ShortInt;
    addStruct(predefinedTypeTable_, "MPI_SHORT_INT", MPI_SHORT_INT, MPI_SHORT, MPI_INT, offsetof(ShortInt, idx));

    typedef struct { long double val; int idx; } LongdoubleInt;
    addStruct(predefinedTypeTable_, "MPI_LONG_DOUBLE_INT", MPI_LONG_DOUBLE_INT, MPI_LONG_DOUBLE, MPI_INT,
              offsetof(LongdoubleInt, idx));

    // Complex datatypes:
    typedef struct { float val; float idx; } FloatComplex;
    addStruct(predefinedTypeTable_, "MPI_FLOAT_COMPLEX", MPI_FLOAT_COMPLEX, MPI_FLOAT, MPI_FLOAT,
              offsetof(FloatComplex, idx));
    addStruct(predefinedTypeTable_, "MPI_COMPLEX", MPI_COMPLEX, MPI_FLOAT, MPI_FLOAT, offsetof(FloatComplex, idx));

    typedef struct { double val; double idx; } DoubleComplex;
    addStruct(predefinedTypeTable_, "MPI_DOUBLE_COMPLEX", MPI_DOUBLE_COMPLEX, MPI_DOUBLE, MPI_DOUBLE,
              offsetof(DoubleComplex, idx));

    typedef struct { long double val; long double idx; } LongDoubleComplex;
    addStruct(predefinedTypeTable_, "MPI_LONG_DOUBLE_COMPLEX", MPI_LONG_DOUBLE_COMPLEX, MPI_LONG_DOUBLE, MPI_LONG_DOUBLE,
              offsetof(LongDoubleComplex, idx));
    return predefinedTypeTable_;
  }

  CkDDT(const std::array<const CkDDT_DataType *, AMPI_MAX_PREDEFINED_TYPE+1>& predefinedTypeTable_) noexcept : predefinedTypeTable(predefinedTypeTable_) {}
  CkDDT& operator=(const CkDDT &obj) = default;
  CkDDT(const CkDDT &obj) = default;
  ~CkDDT() noexcept;

  void newContiguous(int count, MPI_Datatype  oldType, MPI_Datatype* newType) noexcept;
  void newVector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                 MPI_Datatype* newtype) noexcept;
  void newHVector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                  MPI_Datatype* newtype) noexcept;
  void newIndexedBlock(int count, int Blocklength, const int *arrDisp, MPI_Datatype oldtype,
                       MPI_Datatype *newtype) noexcept;
  void newHIndexedBlock(int count, int Blocklength, const MPI_Aint *arrDisp, MPI_Datatype oldtype,
                        MPI_Datatype *newtype) noexcept;
  void newIndexed(int count, const int* arrbLength, MPI_Aint* arrDisp, MPI_Datatype oldtype,
                  MPI_Datatype* newtype) noexcept;
  void newHIndexed(int count, const int* arrbLength, const MPI_Aint* arrDisp, MPI_Datatype oldtype,
                   MPI_Datatype* newtype) noexcept;
  void newStruct(int count, const int* arrbLength, const MPI_Aint* arrDisp,
                 const MPI_Datatype *oldtype, MPI_Datatype* newtype) noexcept;

  int insertType(CkDDT_DataType* ptr, int type) noexcept;
  void freeType(int index) noexcept;
  void pup(PUP::er &p) noexcept;
  void createDup(int nIndexOld, int *nIndexNew) noexcept;
  void createResized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype) noexcept;
  int getEnvelope(int nIndex, int *num_integers, int *num_addresses, int *num_datatypes,
                  int *combiner) const noexcept;
  int getContents(int nIndex, int max_integers, int max_addresses, int max_datatypes,
                  int array_of_integers[], MPI_Aint array_of_addresses[], int array_of_datatypes[]) noexcept;

  CkDDT_DataType* getType(int nIndex) const noexcept {
    if (nIndex <= AMPI_MAX_PREDEFINED_TYPE) {
      CkAssert(nIndex >= 0);
      return const_cast<CkDDT_DataType *>(predefinedTypeTable[nIndex]);
    }
    else {
      CkAssert((nIndex - AMPI_MAX_PREDEFINED_TYPE - 1) < userTypeTable.size());
      return userTypeTable[nIndex - AMPI_MAX_PREDEFINED_TYPE - 1];
    }
  }

  bool isContig(int nIndex) const noexcept { return getType(nIndex)->isContig(); }
  int getSize(int nIndex, int count=1) const noexcept { return count * getType(nIndex)->getSize(); }
  MPI_Aint getExtent(int nIndex) const noexcept { return getType(nIndex)->getExtent(); }
  MPI_Aint getLB(int nIndex) const noexcept { return getType(nIndex)->getLB(); }
  MPI_Aint getUB(int nIndex) const noexcept { return getType(nIndex)->getUB(); }
  MPI_Aint getTrueExtent(int nIndex) const noexcept { return getType(nIndex)->getTrueExtent(); }
  MPI_Aint getTrueLB(int nIndex) const noexcept { return getType(nIndex)->getTrueLB(); }
  void setName(int nIndex, const char *name) noexcept { getType(nIndex)->setName(name); }
  void getName(int nIndex, char *name, int *len) const noexcept { getType(nIndex)->getName(name, len); }
};

#endif
