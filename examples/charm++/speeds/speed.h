#include "charm++.h"
#include "speed.decl.h"

class test : public Group
{
  private:
    int nchunks;
    int numrecd;
    int *speeds;
    void distrib(void);
  public:
    test(CkMigrateMessage *m) {}
    test(void) { speeds = new int[CkNumPes()]; numrecd = 0; }
    void measure(int nc);
    void recv(int pe, int speed);
};

class main : public Chare
{
  public:
    main(CkMigrateMessage *m) {}
    main(CkArgMsg *);
};

