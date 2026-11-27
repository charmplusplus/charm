// Post API tags of an application next to CkIO reads on the same PE (#4304).
// Elements on PE 0: element 0 matches the small tag 0 and posts only after a
// CkIO read has started, element 2 does the same with a tag from CkNewPostTag(),
// and element 1 posts the small tag 1 before a second read. Each read must
// return the file, and each element its own data, received in its own buffer
// after it posted.
#include <cstdio>
#include <cstring>
#include "iopost.decl.h"

CProxy_Main mainProxy;
CProxy_Recv recvProxy;
CProxy_Sender senderProxy;

static const size_t N = 1024;
static const char* fname = "iopost.dat";

static void check(const char* p, char c, const char* what)
{
  for (size_t i = 0; i < N; i++)
    if (p[i] != c)
      CkAbort("%s: byte %zu is '%c', expected '%c'", what, i, p[i], c);
}

class PeZeroMap : public CkArrayMap
{
public:
  PeZeroMap() {}
  int procNum(int, const CkArrayIndex&) override { return 0; }
};

class Sender : public CBase_Sender
{
public:
  Sender() {}
  void send(int to, char fill)
  {
    char* src = new char[N];
    memset(src, fill, N);
    recvProxy[to].recv(CkSendBuffer(src), N);
  }
};

class Main : public CBase_Main
{
  Ck::IO::Session session;
  char readBuf[2][N];
  int reads = 0, matched_ = 0, received_ = 0;

public:
  Main(CkArgMsg* m)
  {
    delete m;
    mainProxy = thisProxy;
    senderProxy = CProxy_Sender::ckNew();
    CkArrayOptions opts(3);
    opts.setMap(CProxy_PeZeroMap::ckNew());
    recvProxy = CProxy_Recv::ckNew(opts);
    FILE* f = fopen(fname, "wb");
    if (!f) CkAbort("cannot create %s", fname);
    for (size_t i = 0; i < N; i++) fputc('F', f);
    fclose(f);
    Ck::IO::Options o;
    o.numReaders = 1;
    Ck::IO::open(fname, CkCallback(CkIndex_Main::fileReady(nullptr), thisProxy), o);
  }

  void fileReady(Ck::IO::FileReadyMsg* m)
  {
    Ck::IO::startReadSession(m->file, N, 0,
                             CkCallback(CkIndex_Main::sessionReady(nullptr), thisProxy));
    delete m;
  }

  void sessionReady(Ck::IO::SessionReadyMsg* m)
  {
    session = m->session;
    delete m;
    senderProxy[CkNumPes() - 1].send(0, 'A');
    senderProxy[CkNumPes() - 1].send(2, 'C');
  }

  void startRead()
  {
    memset(readBuf[reads], '.', N);
    Ck::IO::read(session, N, 0, readBuf[reads],
                 CkCallback(CkIndex_Main::readDone(nullptr), thisProxy));
  }

  void matched()
  {
    if (++matched_ < 2) return;
    startRead();
    recvProxy[0].post();
    recvProxy[2].post();
  }

  void posted() { startRead(); }

  void readDone(Ck::IO::ReadCompleteMsg* m)
  {
    delete m;
    check(readBuf[reads], 'F', "CkIO read buffer");
    if (++reads == 1)
      recvProxy[1].post();
    else
      senderProxy[CkNumPes() - 1].send(1, 'B');
    finish();
  }

  void received()
  {
    received_++;
    finish();
  }

  void finish()
  {
    if (reads == 2 && received_ == 3)
    {
      CkPrintf("io_post_tag: CkIO reads and application posts kept apart\n");
      CkExit();
    }
  }
};

class Recv : public CBase_Recv
{
  char buf[N];
  int tag;
  bool hasPosted = false;

public:
  Recv() : tag(thisIndex) { memset(buf, '.', N); }

  void recv(char* data, size_t n, CkNcpyBufferPost* ncpyPost)
  {
    if (thisIndex == 2)
    {
      tag = CkNewPostTag();
      if (tag >= -1) CkAbort("CkNewPostTag returned %d", tag);
    }
    CkMatchBuffer(ncpyPost, 0, tag);
    if (thisIndex != 1) mainProxy.matched();
  }

  void post()
  {
    hasPosted = true;
    CkPostBuffer(buf, N, tag);
    if (thisIndex == 1) mainProxy.posted();
  }

  void recv(char* data, size_t n)
  {
    if (!hasPosted) CkAbort("element %d received data before it posted", thisIndex);
    if (data != buf) CkAbort("element %d received into another buffer", thisIndex);
    check(buf, "ABC"[thisIndex], "element buffer");
    mainProxy.received();
  }
};

#include "iopost.def.h"
