#ifndef __CkDDT_H_
#define __CkDDT_H_

#include <string>
#include <vector>
#include "charm++.h"
#include "ampi.h"

using std::vector;
using std::string;

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
    user's space to buffer if dir=1 or reverse if dir=-1
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
    CkDDT_DataType() { }
    virtual ~CkDDT_DataType() { }
    CkDDT_DataType(int type);
    CkDDT_DataType(int datatype, int size, MPI_Aint extent, int count, MPI_Aint lb, MPI_Aint ub,
            bool iscontig, int baseSize, MPI_Aint baseExtent, CkDDT_DataType* baseType,
            int numElements, int baseIndex, MPI_Aint trueExtent, MPI_Aint trueLB);
    CkDDT_DataType(const CkDDT_DataType &obj, MPI_Aint _lb, MPI_Aint _extent);
    CkDDT_DataType(const CkDDT_DataType& obj);

    virtual bool isContig(void) const;
    virtual int getSize(int count=1) const;
    virtual MPI_Aint getExtent(void) const;
    virtual int getBaseSize(void) const;
    virtual MPI_Aint getLB(void) const;
    virtual MPI_Aint getUB(void) const;
    virtual MPI_Aint getTrueExtent(void) const;
    virtual MPI_Aint getTrueLB(void) const;
    virtual int getBaseIndex(void) const;
    virtual CkDDT_DataType* getBaseType(void) const;
    virtual MPI_Aint getBaseExtent(void) const;
    virtual int getCount(void) const;
    virtual int getType(void) const;
    virtual int getNumElements(void) const;
    virtual void incRefCount() ;
    virtual int decRefCount();
    virtual void pupType(PUP::er &p, CkDDT* ddt) ;

    virtual int getEnvelope(int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) const;
    virtual int getContents(int max_integers, int max_addresses, int max_datatypes,
                           int array_of_integers[], MPI_Aint array_of_addresses[], int array_of_datatypes[]) const;
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;

    void setName(const char *src);
    void getName(char *dest, int *len) const;
    void setAbsolute(bool arg);

    void setSize(MPI_Aint lb, MPI_Aint extent);
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
  virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
  virtual void pupType(PUP::er &p, CkDDT* ddt) ;
  virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
  virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    vector<int> arrayBlockLength;
    vector<MPI_Aint> arrayDisplacements;

  private:
    CkDDT_Indexed& operator=(const CkDDT_Indexed& obj) ;

  public:
    CkDDT_Indexed(int count, const int* arrBlock, const MPI_Aint* arrDisp, int index,
                CkDDT_DataType* type);
    CkDDT_Indexed() { } ;
    ~CkDDT_Indexed() ;
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    CkDDT_HIndexed& operator=(const CkDDT_HIndexed& obj);

  public:
    CkDDT_HIndexed() { } ;
    CkDDT_HIndexed(int count, const int* arrBlock, const MPI_Aint* arrDisp, int index,
                 CkDDT_DataType* type);
    CkDDT_HIndexed(const CkDDT_HIndexed &obj, MPI_Aint _lb, MPI_Aint _extent);
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    // The MPI Standard has arrDisp as an array of int's to MPI_Type_create_indexed_block, but
    // as an array of MPI_Aint's to MPI_Type_create_hindexed_block, so we store it as Aint's
    // internally and convert from int to Aint in Indexed_Block's constructor:
    vector<MPI_Aint> arrayDisplacements;

  private:
    CkDDT_Indexed_Block& operator=(const CkDDT_Indexed_Block &obj);

  public:
    CkDDT_Indexed_Block(int count, int Blength, const MPI_Aint *ArrDisp, int index, CkDDT_DataType *type);
    CkDDT_Indexed_Block() { };
    CkDDT_Indexed_Block(const CkDDT_Indexed_Block &obj, MPI_Aint _lb, MPI_Aint _extent);
    ~CkDDT_Indexed_Block() ;
    virtual size_t serialize(char *userdata, char *buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT *ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    CkDDT_HIndexed_Block& operator=(const CkDDT_HIndexed_Block &obj);

  public:
    CkDDT_HIndexed_Block(int count, int Blength, const MPI_Aint *ArrDisp, int index, CkDDT_DataType *type);
    CkDDT_HIndexed_Block() { };
    CkDDT_HIndexed_Block(const CkDDT_HIndexed_Block &obj, MPI_Aint _lb, MPI_Aint _extent);
    ~CkDDT_HIndexed_Block() ;
    virtual size_t serialize(char *userdata, char *buffer, int num, int dir) const;
    virtual void pupType(PUP::er &p, CkDDT *ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
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
    virtual size_t serialize(char* userdata, char* buffer, int num, int dir) const;
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner) const;
    virtual int getContents(int ni, int na, int nd, int i[], MPI_Aint a[], int d[]) const;
    virtual const vector<int>& getBaseIndices() const;
    virtual const vector<CkDDT_DataType*>& getBaseTypes() const;
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
    vector<CkDDT_DataType*> typeTable;
    vector<int> types; //used for pup

  public:

  CkDDT(void*) {} // emulates migration constructor
  CkDDT(void) : typeTable(CkDDT_MAX_PRIMITIVE_TYPE+1, nullptr), types(CkDDT_MAX_PRIMITIVE_TYPE+1, MPI_DATATYPE_NULL)
  {
    typeTable[0] = new CkDDT_DataType(MPI_DOUBLE);
    types[0] = MPI_DOUBLE;
    typeTable[1] = new CkDDT_DataType(MPI_INT);
    types[1] = MPI_INT;
    typeTable[2] = new CkDDT_DataType(MPI_FLOAT);
    types[2] = MPI_FLOAT;
    typeTable[3] = new CkDDT_DataType(MPI_COMPLEX);
    types[3] = MPI_COMPLEX;
    typeTable[4] = new CkDDT_DataType(MPI_LOGICAL);
    types[4] = MPI_LOGICAL;
    typeTable[5] = new CkDDT_DataType(MPI_C_BOOL);
    types[5] = MPI_C_BOOL;
    typeTable[6] = new CkDDT_DataType(MPI_CHAR);
    types[6] = MPI_CHAR;
    typeTable[7] = new CkDDT_DataType(MPI_BYTE);
    types[7] = MPI_BYTE;
    typeTable[8] = new CkDDT_DataType(MPI_PACKED);
    types[8] = MPI_PACKED;
    typeTable[9] = new CkDDT_DataType(MPI_SHORT);
    types[9] = MPI_SHORT;
    typeTable[10] = new CkDDT_DataType(MPI_LONG);
    types[10] = MPI_LONG;
    typeTable[11] = new CkDDT_DataType(MPI_UNSIGNED_CHAR);
    types[11] = MPI_UNSIGNED_CHAR;
    typeTable[12] = new CkDDT_DataType(MPI_UNSIGNED_SHORT);
    types[12] = MPI_UNSIGNED_SHORT;
    typeTable[13] = new CkDDT_DataType(MPI_UNSIGNED);
    types[13] = MPI_UNSIGNED;
    typeTable[14] = new CkDDT_DataType(MPI_UNSIGNED_LONG);
    types[14] = MPI_UNSIGNED_LONG;
    typeTable[15] = new CkDDT_DataType(MPI_LONG_DOUBLE);
    types[15] = MPI_LONG_DOUBLE;
    typeTable[16] = new CkDDT_DataType(MPI_FLOAT_INT);
    types[16] = MPI_FLOAT_INT;
    typeTable[17] = new CkDDT_DataType(MPI_DOUBLE_INT);
    types[17] = MPI_DOUBLE_INT;
    typeTable[18] = new CkDDT_DataType(MPI_LONG_INT);
    types[18] = MPI_LONG_INT;
    typeTable[19] = new CkDDT_DataType(MPI_2INT);
    types[19] = MPI_2INT;
    typeTable[20] = new CkDDT_DataType(MPI_SHORT_INT);
    types[20] = MPI_SHORT_INT;
    typeTable[21] = new CkDDT_DataType(MPI_LONG_DOUBLE_INT);
    types[21] = MPI_LONG_DOUBLE_INT;
    typeTable[22] = new CkDDT_DataType(MPI_2FLOAT);
    types[22] = MPI_2FLOAT;
    typeTable[23] = new CkDDT_DataType(MPI_2DOUBLE);
    types[23] = MPI_2DOUBLE;
    typeTable[24] = new CkDDT_DataType(MPI_LB);
    types[24] = MPI_LB;
    typeTable[25] = new CkDDT_DataType(MPI_UB);
    types[25] = MPI_UB;
    typeTable[26] = new CkDDT_DataType(MPI_LONG_LONG_INT);
    types[26] = MPI_LONG_LONG_INT;
    typeTable[27] = new CkDDT_DataType(MPI_DOUBLE_COMPLEX);
    types[27] = MPI_DOUBLE_COMPLEX;
    typeTable[28] = new CkDDT_DataType(MPI_SIGNED_CHAR);
    types[28] = MPI_SIGNED_CHAR;
    typeTable[29] = new CkDDT_DataType(MPI_UNSIGNED_LONG_LONG);
    types[29] = MPI_UNSIGNED_LONG_LONG;
    typeTable[30] = new CkDDT_DataType(MPI_WCHAR);
    types[30] = MPI_WCHAR;
    typeTable[31] = new CkDDT_DataType(MPI_INT8_T);
    types[31] = MPI_INT8_T;
    typeTable[32] = new CkDDT_DataType(MPI_INT16_T);
    types[32] = MPI_INT16_T;
    typeTable[33] = new CkDDT_DataType(MPI_INT32_T);
    types[33] = MPI_INT32_T;
    typeTable[34] = new CkDDT_DataType(MPI_INT64_T);
    types[34] = MPI_INT64_T;
    typeTable[35] = new CkDDT_DataType(MPI_UINT8_T);
    types[35] = MPI_UINT8_T;
    typeTable[36] = new CkDDT_DataType(MPI_UINT16_T);
    types[36] = MPI_UINT16_T;
    typeTable[37] = new CkDDT_DataType(MPI_UINT32_T);
    types[37] = MPI_UINT32_T;
    typeTable[38] = new CkDDT_DataType(MPI_UINT64_T);
    types[38] = MPI_UINT64_T;
    typeTable[39] = new CkDDT_DataType(MPI_FLOAT_COMPLEX);
    types[39] = MPI_FLOAT_COMPLEX;
    typeTable[40] = new CkDDT_DataType(MPI_LONG_DOUBLE_COMPLEX);
    types[40] = MPI_LONG_DOUBLE_COMPLEX;
    typeTable[41] = new CkDDT_DataType(MPI_AINT);
    types[41] = MPI_AINT;
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
  void  pup(PUP::er &p);
  CkDDT_DataType*  getType(int nIndex) const;

  bool isContig(int nIndex) const;
  int getSize(int nIndex, int count=1) const;
  MPI_Aint getExtent(int nIndex) const;
  MPI_Aint getLB(int nIndex) const;
  MPI_Aint getUB(int nIndex) const;
  MPI_Aint getTrueExtent(int nIndex) const;
  MPI_Aint getTrueLB(int nIndex) const;
  void createDup(int nIndexOld, int *nIndexNew);
  int getEnvelope(int nIndex, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner) const;
  int getContents(int nIndex, int max_integers, int max_addresses, int max_datatypes,
                 int array_of_integers[], MPI_Aint array_of_addresses[], int array_of_datatypes[]);
  void setName(int nIndex, const char *name);
  void getName(int nIndex, char *name, int *len) const;
  void createResized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype);
  ~CkDDT() ;
};

#endif
