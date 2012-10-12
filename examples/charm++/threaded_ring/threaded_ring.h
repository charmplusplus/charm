#include "threaded_ring.decl.h"

int nElems;

class Main : public CBase_Main
{
public:
    Main(CkArgMsg*);
    Main(CkMigrateMessage*) {}
};

class Ring : public CBase_Ring
{
public:
    Ring() {
        threadWaiting = false;
        dataHere =  0;
    }
    Ring(CkMigrateMessage*) {}
    void run();
    void getData();
    void waitFor();
private:
    CthThread t;
    int dataHere;
    bool threadWaiting;
};
