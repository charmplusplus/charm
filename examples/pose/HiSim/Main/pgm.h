#include "Pgm.decl.h"

CkChareID mainhandle;
int numWth;
roarray<char, 1024>  appname;
unsigned int netLength;
unsigned int netWidth;                
unsigned int netHeight;  

class main : public Chare {
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {};
};
