#ifndef _MAIN_H_
#define _MAIN_H_
/* readonly */ int mask;
/* readonly */ int numQueens; 
/* readonly */ int grainsize;
/* readonly */ CkGroupID counterGroup;
/* readonly */ CkChareID mainhandle;


class Main : public CBase_Main{

public:
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);
    void Quiescence1(DUMMYMSG *msg);

private:
    double starttimer;
};

#endif
