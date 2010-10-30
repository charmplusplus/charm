#ifndef _MAIN_H_
#define _MAIN_H_

/* readonly */ int grainsize;


class Main : public CBase_Main{

private:
    double starttimer;
    double readfiletimer;
public:
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);

    void done(CkVec<int>);
};
#endif
