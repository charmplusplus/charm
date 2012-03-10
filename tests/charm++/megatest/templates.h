#ifndef _TEMPLATES_H
#define _TEMPLATES_H

#include "templates.decl.h"
#include "megatest.h"

extern readonly<CkGroupID> templates_redid;

extern int BINSIZE;

template <class dtype>
class templates_Single : public CMessage_templates_Single<dtype> {
  public:
    dtype data;
    templates_Single() { data = 0; }
    templates_Single(dtype _data) { data = _data; }
};

class templates_ClientMsg: public CMessage_templates_ClientMsg {
  public:
    CkChareID cid;
    templates_ClientMsg(CkChareID _cid) : cid(_cid) {}
};

template <class dtype>
class templates_Array : public CBase_templates_Array<dtype> {
  private:
    dtype data;
  public:
    templates_Array(void) { data = 0; }
    templates_Array(CkMigrateMessage *m) {}
    void remoteRecv(templates_Single<dtype> *);
    void marshalled(int len,dtype *arr);
};

template <class dtype>
class templates_Reduction : public CBase_templates_Reduction<dtype> {
  private:
    dtype data;
    int nreported;
    CkChareID cid;
  public:
    templates_Reduction(void) { nreported = 0; data = 0; }
    templates_Reduction(CkMigrateMessage *m) {}
    void submit(templates_Single<dtype> *msg);
    void Register(templates_ClientMsg *msg);
    void remoteRecv(templates_Single<dtype> *msg);
};

template <class dtype>
class templates_Collector : public CBase_templates_Collector<dtype> {
  public:
    templates_Collector(void);
    templates_Collector(CkMigrateMessage *m) {}
    void collect(templates_Single<dtype> *msg);
};

#define CK_TEMPLATES_ONLY
#include "templates.def.h"
#undef CK_TEMPLATES_ONLY

#endif
