#include "fib.h"

#define NUMCHARES 20
#define GRAIN 10
 
void fib_init(void){
  CProxy_fib_main::ckNew(0);
}

void fib_moduleinit(void){} 

fib_main::fib_main (void)
{
  fib_Range *pMsg;
  
  result = 0;
  pMsg = new fib_Range;
  pMsg->n = NUMCHARES;
  pMsg->parent = thishandle;
  CProxy_fib_fibFunction::ckNew(pMsg, NULL, CK_PE_ANY);
}  

void fib_main::results(fib_DataMsg *msg)
{
  result = msg->x;
  delete msg;
  if (result != 6765)
    CkAbort("fib test failed! \n");
  megatest_finish();
}      

                                                                                   
fib_fibFunction::fib_fibFunction(fib_Range *m)
{
  sum = 0;
  count = 0;
  parent = m->parent;
  root = m->n;
  if (m->n <= GRAIN){ 
    fib_DataMsg *sub = new fib_DataMsg;
    sub->x = sequent(m->n); 
    CProxy_fib_fibFunction fibproxy(parent);
    fibproxy.sendParent (sub);
  } else {
    //if not then recursively call for f(n-1) and f(n-2) 
    
    fib_Range *pMsg1 = new fib_Range; //create two new chares
    fib_Range *pMsg2 = new fib_Range;

    pMsg1->n = m->n - 1;
    pMsg1->parent = thishandle;
   
    pMsg2->n = m->n - 2;
    pMsg2->parent = thishandle;
   
    CProxy_fib_fibFunction::ckNew(pMsg1, NULL, CK_PE_ANY);
    CProxy_fib_fibFunction::ckNew(pMsg2, NULL, CK_PE_ANY);
  }
  delete m;
}

void fib_fibFunction::sendParent(fib_DataMsg *msg){
  sum += msg->x;
  delete msg;
  count++;
  if (count == 2) {
    fib_DataMsg *subtotal = new fib_DataMsg;
    subtotal->x = sum;
    count = 0;
    if (root == NUMCHARES){
      CProxy_fib_main mainproxy (parent);
      mainproxy.results(subtotal);
    } else {
      CProxy_fib_fibFunction fibproxy(parent);
      fibproxy.sendParent (subtotal);
    }
    delete this;
  }
} 

int fib_fibFunction::sequent(int i)
{
  int sum1 = 0;
  if (i == 0 || i == 1) {
      return i;
  } else {
    sum1 = sequent(i-1) + sequent(i-2);
    return sum1;
  }
}

MEGATEST_REGISTER_TEST(fib,"jackie",1)
#include "fib.def.h"
