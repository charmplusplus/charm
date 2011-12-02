# include "megatest.h"
# include "fib.decl.h"

#define NUMCHARES 20
#define GRAIN 10

class fib_DataMsg: public CMessage_fib_DataMsg { 
 public:
    int x;
};

class fib_Range :public CMessage_fib_Range{
 public:
  int n;
  CkChareID parent;
};

class fib_main : public CBase_fib_main {
 private:
  int result;
 public:
  fib_main(void);
  fib_main(CkMigrateMessage *m) {}
  void results(fib_DataMsg * msg);
};

class fib_fibFunction: public Chare {
 private:
  int sum;
  CkChareID parent;
  int count;
  int root;
  int sequent(int i);
 public:
  fib_fibFunction(fib_Range * m);
  fib_fibFunction(CkMigrateMessage *m) {}
  void sendParent(fib_DataMsg * msg);
};
