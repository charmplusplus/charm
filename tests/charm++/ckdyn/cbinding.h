/*
 * cbinding: a minimal CkDyn "binding" written in C++ (DESIGN.md 3.16).
 *
 * A class is a CbClass: a name, a kind, a method table (methods[0] is the
 * constructor) and a factory for its per-object state. An object is an Obj.
 * The hook table that CkDyn calls is built by cbHooks(); the test program
 * installs it with CkDynInit from an initnode and registers its classes with
 * cbRegister. Nothing here includes a Charm++ header: only ckdyn.h.
 */
#ifndef CBINDING_H
#define CBINDING_H

#include "ckdyn.h"
#include <cstdint>
#include <functional>
#include <vector>

struct Obj;
struct CbClass;

/* A method. Its return value is sent to the message's FUTURE slot when the
 * message carries CKDYN_MSG_BLOCK; otherwise it is ignored. */
typedef int64_t (*CbMethodFn)(Obj *o, const CkDynMsgView *v);

struct CbMethod {
  const char *name;     /* "Class.method", a string literal (kept by pointer) */
  uint32_t    flags;    /* CKDYN_EP_* */
  CbMethodFn  fn;
};

struct CbClass {
  const char *name;
  CkDynKind   kind;
  int         ndims;
  std::vector<CbMethod> methods;   /* methods[0]: constructor */
  int chareIdx;       /* filled by cbRegister; no default initializers: charmc compiles C++11 aggregates */
  int firstEp;
  int ep(int method) const { return firstEp + method; }
};

/* The per-object state. Plain old data, so that pack/unpack is a memcpy. */
struct ObjState {
  int64_t counter;
  int64_t fid;        /* a future id an object reports to later */
  int64_t origPe;     /* migration test: the PE before AtSync */
  int64_t replies;    /* ping-pong */
  int64_t n;          /* ping-pong rounds */
  int64_t mode;
  int64_t expGot;     /* priority test: the expedited message arrived */
  int64_t joined;     /* section_joined calls */
  double  t0;
  CkDynSectionID heldSid;   /* DMain: a section id held across checkpoint and restart */
};

struct Obj {
  const CbClass *cls;
  CkDynTarget t;
  ObjState s;
  std::vector<double> vec;     /* migration test */
  std::vector<int64_t> order;  /* priority test */
};

/* ---- registration ------------------------------------------------------- */

const CkDynHooks *cbHooks(void);
void cbRegister(CbClass *c);      /* CkDynRegisterClass; fills chareIdx, firstEp */
void cbRegisterLate(CbClass *c);  /* the same after start-up, with CkDynRegisterClassLate */
void cbInitPe(void);              /* per-PE state; call from an initproc */

/* ---- futures ------------------------------------------------------------ */

typedef std::function<void(const CkDynMsgView *)> CbFutureFn;
/* A new future on this PE; fn runs once when its value arrives. Ids stay
 * below 65536 so that a 16-bit CMK_REFNUM_TYPE build carries them. */
uint32_t cbNewFuture(CbFutureFn fn);

/* ---- per-PE counters the tests read ------------------------------------- */

int64_t cbUnpacks(void);              /* unpack calls on this PE (each checked the target's chare_idx) */
int64_t cbDeliverManyCalls(int ep);   /* deliver_many calls on this PE for ep */
bool    cbInDeliverMany(void);        /* true while deliver_many is delivering */
int64_t cbMaxDeliverManyN(int ep);    /* the largest n of those calls */

/* ---- targets ------------------------------------------------------------ */

CkDynTarget cbArrayTarget1(int aid, int chareIdx, int i);
CkDynTarget cbGroupTarget(int gid, int chareIdx, int pe);

/* ---- callbacks ---------------------------------------------------------- */

CkDynCallback cbFutureCb(int pe, uint32_t id);
CkDynCallback cbTargetCb(const CkDynTarget &t, int ep, int bcast, uint32_t refnum);
CkDynCallback cbSectionCb(const CkDynSectionID &sid, int ep, uint32_t refnum);

/* ---- message building --------------------------------------------------- */

struct MsgB {
  CkDynMsgBuilder *b;
  explicit MsgB(int prio_bits = 0, int group_dep = -1) { b = CkDynMsgBegin(4, 64, 0, prio_bits, group_dep); }
  MsgB &flags(uint16_t f) { CkDynMsgSetFlags(b, f); return *this; }
  MsgB &user(uint32_t u) { CkDynMsgSetUserId(b, u); return *this; }
  MsgB &i64(int64_t v) { CkDynMsgAddI64(b, v); return *this; }
  MsgB &f64(double v) { CkDynMsgAddF64(b, v); return *this; }
  MsgB &fut(int pe, uint32_t id) { CkDynMsgAddFuture(b, pe, id); return *this; }
  MsgB &cb(const CkDynCallback &c) { CkDynMsgAddCallback(b, &c); return *this; }
  MsgB &sec(const CkDynSectionID &s) { CkDynMsgAddSection(b, &s); return *this; }
  CkDynMsg *finish(int ep) { return CkDynMsgFinish(b, ep); }
  CkDynMcastMsg *finishMcast(int ep) { return CkDynMsgFinishMcast(b, ep); }
};

/* ---- reading views ------------------------------------------------------ */

int64_t cbI64(const CkDynMsgView *v, int slot);     /* aborts unless slot is I64 */
double  cbF64(const CkDynMsgView *v, int slot);
bool    cbFindFuture(const CkDynMsgView *v, int *pe, uint32_t *id);
int     cbFindSlot(const CkDynMsgView *v, int type);    /* first slot of a type, or -1 */
/* The reduction data of a REDRESULT view (slot 0, a BUFFER), and its length. */
const void *cbRedData(const CkDynMsgView *v, uint32_t *len);
/* Bytes of a CkDyn payload: header, slots, data region and tail. */
uint32_t cbPayloadLen(const CkDynMsgView *v);

/* The binding reducer id of the "sum of I64 slot 0" foreign reducer. */
enum { CB_REDUCER_SUM_I64 = 7 };

void cbFail(const char *fmt, ...);   /* print and abort */

#endif
