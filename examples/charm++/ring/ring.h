#include "charm++.h"
#include "ring.decl.h"

class Msg : public CMessage_Msg
{
  public:
    int value;
    int hop;
    int *list1;
    int listsize;
    static void *pack(Msg *);
    static Msg  *unpack(void *);
};

class Btest : public Group
{
  private:
    int nexthop;
  public:
    Btest(CkMigrateMessage *m) {}
    Btest(Msg *);
    void recv_msg(Msg *);
};

class main : public Chare
{
  public:
    static CkChareID mainhandle;
    main(CkMigrateMessage *m) {}
    main(CkArgMsg *);
    void quit_when_done(Msg *);
};

