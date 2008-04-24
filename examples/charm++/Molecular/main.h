/*
 University of Illinois at Urbana-Champaign
 Department of Computer Science
 Parallel Programming Lab
 2008
*/

#ifndef __MAIN_H__
#define __MAIN_H__

// Main class
class Main : public CBase_Main {

  private:
    int checkInCount; // Count to terminate

  public:

    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);

    void checkIn();
};

#endif
