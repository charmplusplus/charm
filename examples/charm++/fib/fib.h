#include "fib.decl.h"

class main : public CBase_main
{
    public:
        main(CkMigrateMessage *m) {}
        main(CkArgMsg *m);
};

class fib : public CBase_fib
{
    private:
        int result, count, IamRoot;
        CProxy_fib parent;
    public:
        fib(CkMigrateMessage *m) {}
        fib(int amIRoot, int n, CProxy_fib parent);
        void response(int fibValue);
        void processResult();
};

