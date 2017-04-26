#include "TypedReduction.decl.h"
class Driver : public CBase_Driver {
    public: 
        Driver(CkArgMsg*);
        void untyped_done(CkReductionMsg* m);
        void typed_done(int x);
        void typed_array_ints_done(int* results, int n);
        void typed_indiv_ints_done(int x, int y, int z);
        void typed_array_doubles_done(int n, double* results);
        void set_done(CkReductionMsg *msg);
        void tuple_reducer_done(CkReductionMsg* msg);
    private:
        CProxy_Worker w;
};

class Worker : public CBase_Worker {
    public:
        Worker(void);
        Worker(CkMigrateMessage* m) {}
        void untyped_reduce();
        void typed_reduce();
        void reduce_array_ints();
        void reduce_indiv_ints();
        void reduce_array_doubles();
        void reduce_set();
        void reduce_tuple();
};

