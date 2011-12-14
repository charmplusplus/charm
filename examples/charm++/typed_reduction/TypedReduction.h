#include "TypedReduction.decl.h"
class Driver : public CBase_Driver {
    public: 
        Driver(CkArgMsg*);
        void untyped_done(CkReductionMsg* m);
        void typed_done(int x);
        void typed_array_done(int* results, int n);
        void typed_array_done2(int x, int y, int z);
        void typed_array_done3(int n, double* results);
    private:
        CProxy_Worker w;
        int count;
};

class Worker : public CBase_Worker {
    public:
        Worker(void);
        Worker(CkMigrateMessage* m) {}
        void reduce();
        void reduce_array();
        void reduce_array_doubles();
};

