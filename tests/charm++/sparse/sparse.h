#include "sparse.decl.h"

class Main : public CBase_Main {
  private:
    int expected; 
  public:
    Main_SDAG_CODE
    Main(CkArgMsg*);
    void checkTest(int);
    void setExpected(CkArrayIndex);
    void setExpected(CkArrayIndex, CkArrayIndex, CkArrayIndex);
    void test1D(CkArrayOptions options);
    void test2D(CkArrayOptions options);
    void test3D(CkArrayOptions options);
    void test4D(CkArrayOptions options);
    void test5D(CkArrayOptions options);
    void test6D(CkArrayOptions options);
};

class Array1D : public CBase_Array1D {
  public:
    Array1D();
    Array1D(CkMigrateMessage* msg) {}
    void ping();
};

class Array2D : public CBase_Array2D {
  public:
    Array2D();
    Array2D(CkMigrateMessage* msg) {}
    void ping();
};

class Array3D : public CBase_Array3D {
  public:
    Array3D();
    Array3D(CkMigrateMessage* msg) {}
    void ping();
};

class Array4D : public CBase_Array4D {
  public:
    Array4D();
    Array4D(CkMigrateMessage* msg) {}
    void ping();
};

class Array5D : public CBase_Array5D {
  public:
    Array5D();
    Array5D(CkMigrateMessage* msg) {}
    void ping();
};

class Array6D : public CBase_Array6D {
  public:
    Array6D();
    Array6D(CkMigrateMessage* msg) {}
    void ping();
};
