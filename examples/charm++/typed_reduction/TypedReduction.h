#include "TypedReduction.decl.h"
class Driver : public CBase_Driver {
    public: 
        Driver(CkArgMsg*);
        void untyped_done(CkReductionMsg* m);
        void typed_done(int x);
        void typed_array_done(int* results, int n);
        void typed_array_done2(int x, int y, int z);
    private:
        CProxy_Worker w;
};

class Worker : public CBase_Worker {
    public:
        Worker(void);
        Worker(CkMigrateMessage* m) {}
        void reduce();
        void reduce_array();
};

