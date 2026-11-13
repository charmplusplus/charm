#ifndef __MAIN_H__
#define __MAIN_H__

extern /* readonly */ CProxy_Cell cellArray;
extern /* readonly */ CProxy_Compute computeArray;

extern /* readonly */ int cellArrayDimX;
extern /* readonly */ int cellArrayDimY;
extern /* readonly */ int cellArrayDimZ;

//central controller chare
class Main : public CBase_Main {
  private:
    Main_CORO_CODE
    long long checksum;   // received by done()

  public:
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);
    void pup(PUP::er &p);
    ck::coroutine run();
};
#endif
