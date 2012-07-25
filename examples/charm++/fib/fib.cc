#include "fib.h"
#define THRESHOLD 10

main::main(CkArgMsg * m)
{ 
    if(m->argc < 2) CmiAbort("./pgm N.");
    int n = atoi(m->argv[1]); 
    CProxy_fib::ckNew(1, n, thishandle); 
}

int seqFib(int n) {
    if (n<2) return n;
    else return (seqFib(n-1) + seqFib(n-2));
}

fib::fib(int AmIRoot, int n, CProxy_fib parent){ 
    CkPrintf("in fib::fib. n=%d\n", n);
    IamRoot = AmIRoot;
    this->parent = parent;
    if (n< THRESHOLD) {
        result =seqFib(n);
        processResult();
    } else {
        CProxy_fib::ckNew(0,n-1, thishandle); 
        CProxy_fib::ckNew(0,n-2, thishandle); 
        result = 0;
        count = 2;
    }
}

void fib::response(int fibValue) {
    result += fibValue;
    if (--count == 0)
        processResult();
}

void fib::processResult()
{
    CkPrintf("result:%d\n", result);
    if (IamRoot) {
        CkPrintf("The requested Fibonacci number is : %d\n", result);
        CkExit();
    } else {
        parent.response(result);
    }
    delete this; /*this chare has no more work to do.*/ 
}

#include "fib.def.h"

