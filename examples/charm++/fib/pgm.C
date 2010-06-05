#include "pgm.h"
#define THRESHOLD 10

main::main(CkArgMsg * m)
{ 
  if(m->argc < 2) CmiAbort("./pgm N.");
  int n = atoi(m->argv[1]); 
  CProxy_fib::ckNew(1, n, thishandle); 

  CProxy_SingleProvider a = CProxy_SingleProvider::ckNew();
  CProxy_SetSimpleInterface b = CProxy_SetSimpleInterface::ckNew();
  a.provider_set_s(b, a, 2, "w");
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
    processResult();}
  else {
    CProxy_fib::ckNew(0,n-1, thishandle); 
    CProxy_fib::ckNew(0,n-2, thishandle); 
    result = 0;
    count = 2;  }
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
//    CkExit();
  }
  else parent.response(result);
  delete this; /*this chare has no more work to do.*/ 
}

class SimpleInterface: public Chare {
public:
SimpleInterface() {}
};


class SetSimpleInterface: public Chare {
public:
SetSimpleInterface() {}
void set_si(CProxy_SimpleInterface& pssi, int n, char *name){
printf("HERE\n");
}
};

class SingleProvider : virtual public CBase_SingleProvider {
public:
SingleProvider() {}
void provider_set_s(CProxy_SetSimpleInterface& pssi, CProxy_SimpleInterface& psi, int n, char *name){
printf("FIRST\n");
   pssi.set_si(thishandle,n,name);
}
    
};


#include "pgm.def.h"
