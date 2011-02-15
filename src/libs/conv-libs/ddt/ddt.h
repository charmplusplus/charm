#ifndef __CkDDT_H_
#define __CkDDT_H_

#include "pup.h"

#define CkDDT_MAXTYPE         100

#define CkDDT_TYPE_NULL      -1
#define CkDDT_DOUBLE          0
#define CkDDT_INT             1
#define CkDDT_FLOAT           2
#define CkDDT_COMPLEX         3
#define CkDDT_LOGICAL         4
#define CkDDT_CHAR            5
#define CkDDT_BYTE            6
#define CkDDT_PACKED          7
#define CkDDT_SHORT           8
#define CkDDT_LONG            9
#define CkDDT_UNSIGNED_CHAR   10
#define CkDDT_UNSIGNED_SHORT  11
#define CkDDT_UNSIGNED        12
#define CkDDT_UNSIGNED_LONG   13
#define CkDDT_LONG_DOUBLE     14
#define CkDDT_FLOAT_INT       15
#define CkDDT_DOUBLE_INT      16
#define CkDDT_LONG_INT        17
#define CkDDT_2INT            18
#define CkDDT_SHORT_INT       19
#define CkDDT_LONG_DOUBLE_INT 20
#define CkDDT_2FLOAT          21
#define CkDDT_2DOUBLE         22
#define CkDDT_LB              23
#define CkDDT_UB              24
#define CkDDT_LONG_LONG_INT   25
#define CkDDT_DOUBLE_COMPLEX  26

#define CkDDT_CONTIGUOUS      36
#define CkDDT_VECTOR          37
#define CkDDT_HVECTOR         38
#define CkDDT_INDEXED         39
#define CkDDT_HINDEXED        40
#define CkDDT_STRUCT          41

/* for the datatype decoders */
#define CkDDT_COMBINER_NAMED         1
#define CkDDT_COMBINER_CONTIGUOUS    2
#define CkDDT_COMBINER_VECTOR        3
#define CkDDT_COMBINER_HVECTOR       4
#define CkDDT_COMBINER_INDEXED       5
#define CkDDT_COMBINER_HINDEXED      6
#define CkDDT_COMBINER_STRUCT        7

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

  Methods:

  getSize -  returns the size of the datatype. 
  getExtent - returns the extent of the datatype.

  inrRefCount - increament the reference count.
  getRefCount - returns the RefCount

  serialize - This is the function which actually copies the contents from
    user's space to buffer if dir=1 or reverse if dir=0
    according to the datatype.
*/

class CkDDT_DataType {

  protected:
    int datatype;
    int refCount;

  protected:
    int size;
    int extent;
    int count;
    int lb;
    int ub;
    int iscontig;
    int baseSize;
    int baseExtent;
    CkDDT_DataType *baseType;
    int baseIndex;

  public:
    CkDDT_DataType() { }
    virtual ~CkDDT_DataType() { }
    CkDDT_DataType(int type);
    CkDDT_DataType(int datatype, int size, int extent, int count, int lb, int ub,
            int iscontig, int baseSize, int baseExtent,
            CkDDT_DataType* baseType, int baseIndex);
    CkDDT_DataType(const CkDDT_DataType& obj);
    CkDDT_DataType& operator=(const CkDDT_DataType& obj);

    virtual int isContig(void);
    virtual int getSize(int count=1);
    virtual int getExtent(void);
    virtual int getBaseSize(void);
    virtual int getLB(void);
    virtual int getUB(void);
    virtual int getType(void);
    virtual void inrRefCount(void) ;
    virtual int getRefCount(void) ;
    virtual void pupType(PUP::er &p, CkDDT* ddt) ;

    virtual int getEnvelope(int *num_integers, int *num_addresses, int *num_datatypes, int *combiner);
    virtual int getContents(int max_integers, int max_addresses, int max_datatypes,
                           int array_of_integers[], int array_of_addresses[], int array_of_datatypes[]);
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
};

/*
  This class maintains the type Contiguous.
  It constructs a typemap consisting of the
  replication of a datatype into contiguous locations.
*/

class CkDDT_Contiguous : public CkDDT_DataType {
 public:
  CkDDT_Contiguous() { };
  CkDDT_Contiguous(int count, int index, CkDDT_DataType* oldType);
  CkDDT_Contiguous(const CkDDT_Contiguous& obj) ;
  CkDDT_Contiguous& operator=(const CkDDT_Contiguous& obj);
  virtual int serialize(char* userdata, char* buffer, int num, int dir);
  virtual void pupType(PUP::er &p, CkDDT* ddt) ;
  virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner);
  virtual int getContents(int ni, int na, int nd, int i[], int a[], int d[]);
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
  public:
    CkDDT_Vector(int count, int blklen, int stride, int index,
                CkDDT_DataType* type);
    CkDDT_Vector(const CkDDT_Vector& obj) ;
    CkDDT_Vector& operator=(const CkDDT_Vector& obj);
    CkDDT_Vector() { } ;
    ~CkDDT_Vector() { } ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner);
    virtual int getContents(int ni, int na, int nd, int i[], int a[], int d[]);
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

  public:
    CkDDT_HVector() { } ;
    CkDDT_HVector(int nCount,int blength,int strideLen,int index,
                CkDDT_DataType* type);
    ~CkDDT_HVector() { } ;
    CkDDT_HVector(const CkDDT_HVector& obj) ;
    CkDDT_HVector& operator=(const CkDDT_HVector& obj);
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner);
    virtual int getContents(int ni, int na, int nd, int i[], int a[], int d[]);
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
    int* arrayDisplacements ;

  public:

    CkDDT_Indexed(int count, int* arrBlock, int* arrDisp, int index,
                CkDDT_DataType* type);
    CkDDT_Indexed(const CkDDT_Indexed& obj);
    CkDDT_Indexed& operator=(const CkDDT_Indexed& obj) ;
    CkDDT_Indexed() { } ;
    ~CkDDT_Indexed() ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner);
    virtual int getContents(int ni, int na, int nd, int i[], int a[], int d[]);
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

  public:
    CkDDT_HIndexed() { } ;
    CkDDT_HIndexed(int count, int* arrBlock, int* arrDisp, int index, 
                 CkDDT_DataType* type);
    CkDDT_HIndexed(const CkDDT_HIndexed& obj);
    CkDDT_HIndexed& operator=(const CkDDT_HIndexed& obj) ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual void pupType(PUP::er &p, CkDDT* ddt);
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner);
    virtual int getContents(int ni, int na, int nd, int i[], int a[], int d[]);
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
    int* arrayDisplacements ;
    int* index;
    CkDDT_DataType** arrayDataType;

  public:
    CkDDT_Struct() { } ;
    CkDDT_Struct(int count, int* arrBlock, int* arrDisp, int *index,
               CkDDT_DataType **type);
    CkDDT_Struct(const CkDDT_Struct& obj);
    CkDDT_Struct& operator=(const CkDDT_Struct& obj) ;
    virtual int serialize(char* userdata, char* buffer, int num, int dir);
    virtual  void pupType(PUP::er &p, CkDDT* ddt) ;
    virtual int getEnvelope(int *ni, int *na, int *nd, int *combiner);
    virtual int getContents(int ni, int na, int nd, int i[], int a[], int d[]);
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
    typeTable[5] = new CkDDT_DataType(CkDDT_CHAR);
    types[5] = CkDDT_CHAR;
    typeTable[6] = new CkDDT_DataType(CkDDT_BYTE);
    types[6] = CkDDT_BYTE;
    typeTable[7] = new CkDDT_DataType(CkDDT_PACKED);
    types[7] = CkDDT_PACKED;
    typeTable[8] = new CkDDT_DataType(CkDDT_SHORT);
    types[8] = CkDDT_SHORT;
    typeTable[9] = new CkDDT_DataType(CkDDT_LONG);
    types[9] = CkDDT_LONG;
    typeTable[10] = new CkDDT_DataType(CkDDT_UNSIGNED_CHAR);
    types[10] = CkDDT_UNSIGNED_CHAR;
    typeTable[11] = new CkDDT_DataType(CkDDT_UNSIGNED_SHORT);
    types[11] = CkDDT_UNSIGNED_SHORT;
    typeTable[12] = new CkDDT_DataType(CkDDT_UNSIGNED);
    types[12] = CkDDT_UNSIGNED;
    typeTable[13] = new CkDDT_DataType(CkDDT_UNSIGNED_LONG);
    types[13] = CkDDT_UNSIGNED_LONG;
    typeTable[14] = new CkDDT_DataType(CkDDT_LONG_DOUBLE);
    types[14] = CkDDT_LONG_DOUBLE;
    typeTable[15] = new CkDDT_DataType(CkDDT_FLOAT_INT);
    types[15] = CkDDT_FLOAT_INT;
    typeTable[16] = new CkDDT_DataType(CkDDT_DOUBLE_INT);
    types[16] = CkDDT_DOUBLE_INT;
    typeTable[17] = new CkDDT_DataType(CkDDT_LONG_INT);
    types[17] = CkDDT_LONG_INT;
    typeTable[18] = new CkDDT_DataType(CkDDT_2INT);
    types[18] = CkDDT_2INT;
    typeTable[19] = new CkDDT_DataType(CkDDT_SHORT_INT);
    types[19] = CkDDT_SHORT_INT;
    typeTable[20] = new CkDDT_DataType(CkDDT_LONG_DOUBLE_INT);
    types[20] = CkDDT_LONG_DOUBLE_INT;
    typeTable[21] = new CkDDT_DataType(CkDDT_2FLOAT);
    types[21] = CkDDT_2FLOAT;
    typeTable[22] = new CkDDT_DataType(CkDDT_2DOUBLE);
    types[22] = CkDDT_2DOUBLE;
    typeTable[23] = new CkDDT_DataType(CkDDT_LB);
    types[23] = CkDDT_LB;
    typeTable[24] = new CkDDT_DataType(CkDDT_UB);
    types[24] = CkDDT_UB;
    typeTable[25] = new CkDDT_DataType(CkDDT_LONG_LONG_INT);
    types[25] = CkDDT_LONG_LONG_INT;    
    typeTable[26] = new CkDDT_DataType(CkDDT_DOUBLE_COMPLEX);
    types[26] = CkDDT_DOUBLE_COMPLEX;
    num_types = 27;

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
  void newIndexed(int count, int* arrbLength, int* arrDisp , CkDDT_Type oldtype,
                 CkDDT_Type* newtype);
  void newHIndexed(int count, int* arrbLength, int* arrDisp , CkDDT_Type oldtype,
                  CkDDT_Type* newtype);
  void newStruct(int count, int* arrbLength, int* arrDisp , CkDDT_Type *oldtype,
                CkDDT_Type* newtype);
  void  freeType(int* index);
  int   getNextFreeIndex(void) ;
  void  pup(PUP::er &p);
  CkDDT_DataType*  getType(int nIndex);

  int isContig(int nIndex);
  int getSize(int nIndex, int count=1);
  int getExtent(int nIndex);
  int getLB(int nIndex);
  int getUB(int nIndex);
  int getEnvelope(int nIndex, int *num_integers, int *num_addresses, int *num_datatypes, int *combiner);
  int getContents(int nIndex, int max_integers, int max_addresses, int max_datatypes,
                 int array_of_integers[], int array_of_addresses[], int array_of_datatypes[]);
  ~CkDDT() ;
};

#endif
