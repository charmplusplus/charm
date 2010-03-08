#ifndef _MAIN_H_
#define _MAIN_H_

class Main : public CBase_Main{

private:
    double starttimer;
    double readfiletimer;
public:
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);

    void done();
};
#endif
