/*
 * cbinding: the hook table of the C++ test binding (see cbinding.h).
 * Per-PE state lives in a Ckpv: on reconverse several PEs share a process.
 */
#include "charm++.h"
#include "cbinding.h"
#include <cstdarg>
#include <cstring>
#include <map>
#include <unordered_map>

struct PeState {
  uint32_t nextFid = 1;
  std::unordered_map<uint32_t, CbFutureFn> futures;
  std::map<int, int64_t> manyCalls;   /* ep -> deliver_many calls */
  std::map<int, int64_t> manyMaxN;    /* ep -> largest n */
  bool inMany = false;
  int64_t unpacks = 0;                /* unpack calls whose target carried the class's chare index */
};
CkpvStaticDeclare(PeState *, cbPe);

void cbInitPe(void) {
  CkpvInitialize(PeState *, cbPe);
  CkpvAccess(cbPe) = new PeState;
}

void cbFail(const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  CkAbort("FAIL [PE %d] %s", CkMyPe(), buf);
}

/* ---- views -------------------------------------------------------------- */

int64_t cbI64(const CkDynMsgView *v, int slot) {
  if (slot >= v->nslots || v->slots[slot].type != CKDYN_SLOT_I64)
    cbFail("slot %d is not I64 (nslots %d)", slot, v->nslots);
  return v->slots[slot].v.i64;
}

double cbF64(const CkDynMsgView *v, int slot) {
  if (slot >= v->nslots || v->slots[slot].type != CKDYN_SLOT_F64)
    cbFail("slot %d is not F64 (nslots %d)", slot, v->nslots);
  return v->slots[slot].v.f64;
}

int cbFindSlot(const CkDynMsgView *v, int type) {
  for (int i = 0; i < v->nslots; i++) if (v->slots[i].type == type) return i;
  return -1;
}

bool cbFindFuture(const CkDynMsgView *v, int *pe, uint32_t *id) {
  int i = cbFindSlot(v, CKDYN_SLOT_FUTURE);
  if (i < 0) return false;
  *pe = (int)(v->slots[i].v.u64 >> 32);
  *id = (uint32_t)(v->slots[i].v.u64 & 0xffffffffu);
  return true;
}

const void *cbRedData(const CkDynMsgView *v, uint32_t *len) {
  if (v->kind != CKDYN_MSGKIND_REDRESULT) cbFail("expected a REDRESULT view, got kind %d", (int)v->kind);
  if (v->nslots < 1 || v->slots[0].type != CKDYN_SLOT_BUFFER) cbFail("REDRESULT view without a BUFFER slot 0");
  *len = v->slots[0].len;
  return CkDynViewBufferPtr(v, 0, nullptr);
}

uint32_t cbPayloadLen(const CkDynMsgView *v) {
  const CkDynHdr *h = v->hdr;
  return (uint32_t)(sizeof(CkDynHdr) + h->nslots * sizeof(CkDynSlot) + h->data_len + h->tail_len);
}

/* ---- targets and callbacks ---------------------------------------------- */

CkDynTarget cbArrayTarget1(int aid, int chareIdx, int i) {
  CkDynTarget t;
  memset(&t, 0, sizeof t);
  t.kind = CKDYN_ARRAY; t.chare_idx = chareIdx;
  t.u.array.aid = aid; t.u.array.ndims = 1; t.u.array.index[0] = i;
  return t;
}

CkDynTarget cbGroupTarget(int gid, int chareIdx, int pe) {
  CkDynTarget t;
  memset(&t, 0, sizeof t);
  t.kind = CKDYN_GROUP; t.chare_idx = chareIdx;
  t.u.group.gid = gid; t.u.group.pe_or_node = pe;
  return t;
}

CkDynCallback cbFutureCb(int pe, uint32_t id) {
  CkDynCallback c;
  memset(&c, 0, sizeof c);
  c.kind = CKDYN_CB_FUTURE; c.pe = pe; c.id = id;
  return c;
}

CkDynCallback cbTargetCb(const CkDynTarget &t, int ep, int bcast, uint32_t refnum) {
  CkDynCallback c;
  memset(&c, 0, sizeof c);
  c.kind = CKDYN_CB_TARGET; c.t = t; c.ep = ep; c.bcast = bcast; c.refnum = refnum;
  return c;
}

CkDynCallback cbSectionCb(const CkDynSectionID &sid, int ep, uint32_t refnum) {
  CkDynCallback c;
  memset(&c, 0, sizeof c);
  c.kind = CKDYN_CB_SECTION; c.sid = sid; c.ep = ep; c.refnum = refnum;
  return c;
}

/* ---- futures ------------------------------------------------------------ */

uint32_t cbNewFuture(CbFutureFn fn) {
  PeState *s = CkpvAccess(cbPe);
  uint32_t id;
  do {
    id = s->nextFid++;
    if (s->nextFid >= 65536) s->nextFid = 1;
  } while (s->futures.count(id));
  s->futures[id] = std::move(fn);
  return id;
}

int64_t cbUnpacks(void) { return CkpvAccess(cbPe)->unpacks; }
int64_t cbDeliverManyCalls(int ep) { auto &m = CkpvAccess(cbPe)->manyCalls; auto it = m.find(ep); return it == m.end() ? 0 : it->second; }
int64_t cbMaxDeliverManyN(int ep) { auto &m = CkpvAccess(cbPe)->manyMaxN; auto it = m.find(ep); return it == m.end() ? 0 : it->second; }
bool cbInDeliverMany(void) { return CkpvAccess(cbPe)->inMany; }

/* ---- hooks -------------------------------------------------------------- */

static void *hConstruct(void *bd, const CkDynTarget *t, const CkDynMsgView *v) {
  const CbClass *c = (const CbClass *)bd;
  Obj *o = new Obj;
  o->cls = c;
  o->t = *t;
  memset(&o->s, 0, sizeof o->s);
  if (c->methods[0].fn) c->methods[0].fn(o, v);
  return o;
}

/* The main chare's constructor method runs with a NULL view (there is no
 * CkDyn message, only argc/argv). */
static void hMainchare(void *bd, void **handle_out, const CkDynTarget *t, int, char **) {
  const CbClass *c = (const CbClass *)bd;
  Obj *o = new Obj;
  o->cls = c;
  o->t = *t;
  memset(&o->s, 0, sizeof o->s);
  *handle_out = o;
  if (c->methods[0].fn) c->methods[0].fn(o, nullptr);
}

static void hDestruct(void *, void *h) { delete (Obj *)h; }

static void hDeliver(void *bd, void *h, int ep, uint32_t user_id, uint32_t, const CkDynMsgView *v) {
  const CbClass *c = (const CbClass *)bd;
  if (user_id >= c->methods.size() || ep != c->ep((int)user_id))
    cbFail("%s: delivery to ep %d with user_id %u does not match the class table (first ep %d)",
           c->name, ep, user_id, c->firstEp);
  Obj *o = (Obj *)h;
  if (o == nullptr) cbFail("%s: delivery to a NULL handle", c->name);
  int64_t r = c->methods[user_id].fn(o, v);
  if (v->hdr && (v->hdr->flags & CKDYN_MSG_BLOCK)) {
    int pe; uint32_t id;
    if (!cbFindFuture(v, &pe, &id)) cbFail("%s: CKDYN_MSG_BLOCK without a FUTURE slot", c->methods[user_id].name);
    CkDynFutureSet(pe, id, MsgB().i64(r).finish(0));
  }
}

static void hDeliverMany(void *bd, void **hs, int n, int ep, uint32_t user_id, uint32_t f, const CkDynMsgView *v) {
  PeState *s = CkpvAccess(cbPe);
  s->manyCalls[ep]++;
  if (n > s->manyMaxN[ep]) s->manyMaxN[ep] = n;
  s->inMany = true;
  for (int i = 0; i < n; i++) hDeliver(bd, hs[i], ep, user_id, f, v);
  s->inMany = false;
}

/* pack layout: ObjState, uint64 nvec, nvec doubles, uint64 norder, norder int64 */
static size_t hPackSize(void *, void *h, int) {
  Obj *o = (Obj *)h;
  return sizeof(ObjState) + 8 + 8 * o->vec.size() + 8 + 8 * o->order.size();
}

static void hPack(void *bd, void *h, int reason, void *buf, size_t len) {
  Obj *o = (Obj *)h;
  if (len != hPackSize(bd, h, reason)) cbFail("pack: length %zu, expected %zu", len, hPackSize(bd, h, reason));
  char *p = (char *)buf;
  memcpy(p, &o->s, sizeof o->s); p += sizeof o->s;
  uint64_t n = o->vec.size(); memcpy(p, &n, 8); p += 8;
  if (n) { memcpy(p, o->vec.data(), 8 * n); p += 8 * n; }
  n = o->order.size(); memcpy(p, &n, 8); p += 8;
  if (n) memcpy(p, o->order.data(), 8 * n);
}

static void *hUnpack(void *bd, const CkDynTarget *t, int, const void *buf, size_t len) {
  Obj *o = new Obj;
  o->cls = (const CbClass *)bd;
  o->t = *t;
  if (t->chare_idx != o->cls->chareIdx)
    cbFail("unpack: %s target has chare_idx %d, expected the registered %d", o->cls->name, t->chare_idx, o->cls->chareIdx);
  if (t->kind == CKDYN_ARRAY) {
    CkDynTarget e = cbArrayTarget1(t->u.array.aid, t->chare_idx, t->u.array.index[0]);
    if (t->u.array.ndims != 1 || memcmp(t, &e, sizeof e) != 0)
      cbFail("unpack: the target CkDyn built differs in its bytes from the binding-built one");
  }
  CkpvAccess(cbPe)->unpacks++;
  const char *p = (const char *)buf, *end = p + len;
  if (len < sizeof(ObjState) + 16) cbFail("unpack: %zu bytes is too short", len);
  memcpy(&o->s, p, sizeof o->s); p += sizeof o->s;
  uint64_t n; memcpy(&n, p, 8); p += 8;
  o->vec.resize(n); if (n) { memcpy(o->vec.data(), p, 8 * n); p += 8 * n; }
  memcpy(&n, p, 8); p += 8;
  o->order.resize(n); if (n) { memcpy(o->order.data(), p, 8 * n); p += 8 * n; }
  if (p != end) cbFail("unpack: %td bytes left over", end - p);
  return o;
}

/* The reduce hook's second slot: a BUFFER of sum, sum+1, ... built in a
   stack array that is overwritten right after CkDynMsgAddBuffer returns, so a
   builder that read it later would deliver 0xab bytes. */
static void addReduceCheckBuffer(CkDynMsgBuilder *result, int64_t sum) {
  int64_t tmp[CB_RBUF_N];
  for (int k = 0; k < CB_RBUF_N; k++) tmp[k] = sum + k;
  static const int64_t shape[1] = { CB_RBUF_N };
  CkDynBufDesc d;
  memset(&d, 0, sizeof d);
  d.ptr = tmp; d.len = sizeof tmp; d.elem = CKDYN_ELEM_I64; d.ndim = 1; d.shape = shape;
  CkDynMsgAddBuffer(result, &d);
  volatile unsigned char *q = (volatile unsigned char *)tmp;   /* volatile: the overwrite is kept */
  for (size_t i = 0; i < sizeof tmp; i++) q[i] = 0xab;
}

static void hReduce(void *, uint32_t rid, const CkDynMsgView **contribs, int n, CkDynMsgBuilder *result) {
  if (rid != CB_REDUCER_SUM_I64) cbFail("reduce: unknown reducer id %u", rid);
  int64_t sum = 0;
  for (int i = 0; i < n; i++) {
    if (contribs[i]->hdr == nullptr || contribs[i]->hdr->user_id != rid)
      cbFail("reduce: contribution %d has reducer id %u, expected %u", i,
             contribs[i]->hdr ? contribs[i]->hdr->user_id : 0u, rid);
    sum += cbI64(contribs[i], 0);
  }
  CkDynMsgAddI64(result, sum);
  addReduceCheckBuffer(result, sum);
}

int cbCheckReduceBuffer(const CkDynMsgView *v) {
  if (v->nslots < 2) return 0;
  if (v->slots[1].type != CKDYN_SLOT_BUFFER || v->slots[1].len != CB_RBUF_N * sizeof(int64_t) ||
      v->slots[1].elem != CKDYN_ELEM_I64)
    cbFail("reduce result: slot 1 is not the %d-int64 BUFFER the reduce hook added", CB_RBUF_N);
  const int64_t *p = (const int64_t *)CkDynViewBufferPtr(v, 1, nullptr);
  int64_t sum = cbI64(v, 0);
  for (int k = 0; k < CB_RBUF_N; k++)
    if (p[k] != sum + k)
      cbFail("reduce result: BUFFER element %d is %lld, expected %lld (the hook's temporary was not copied at "
             "CkDynMsgAddBuffer)", k, (long long)p[k], (long long)(sum + k));
  return 1;
}

static void hFuture(void *, int pe, uint32_t id, const CkDynMsgView *v) {
  if (pe != CkMyPe()) cbFail("future %u: hook called for PE %d on PE %d", id, pe, CkMyPe());
  PeState *s = CkpvAccess(cbPe);
  auto it = s->futures.find(id);
  if (it == s->futures.end()) cbFail("future %u: no such future on this PE", id);
  CbFutureFn fn = std::move(it->second);
  s->futures.erase(it);
  fn(v);
}

static int hMapProcNum(void *, void *, int, int ndims, const int *index) {
  int64_t lin = 0;
  for (int i = 0; i < ndims; i++) lin = lin * 1009 + index[i];
  return (int)(lin % CkNumPes());
}

/* ResumeFromSync: the class's method named "*.resumed" (by convention the
 * last method) is sent to the element itself. */
static void hResumeFromSync(void *bd, void *h) {
  const CbClass *c = (const CbClass *)bd;
  Obj *o = (Obj *)h;
  int m = -1;
  for (size_t i = 0; i < c->methods.size(); i++)
    if (strstr(c->methods[i].name, ".resumed")) m = (int)i;
  if (m < 0) cbFail("%s: resume_from_sync without a resumed method", c->name);
  CkDynSendTarget(&o->t, c->ep(m), MsgB().finish(c->ep(m)), nullptr);
}

/* section_joined: the member counts the sections it belongs to */
static void hSectionJoined(void *bd, void *h, const CkDynSectionID *sid) {
  Obj *o = (Obj *)h;
  if (o->cls != (const CbClass *)bd) cbFail("section_joined: binding data of another class");
  if (sid->root_pe < 0 || sid->root_pe >= CkNumPes() || sid->gen == 0) cbFail("section_joined: malformed section id");
  o->s.joined++;
}

static void hError(void *, const char *what) {
  CkPrintf("[PE %d] cbinding: CkDyn error: %s\n", CkMyPe(), what);
}

static CkDynHooks g_hooks;

const CkDynHooks *cbHooks(void) {
  memset(&g_hooks, 0, sizeof g_hooks);
  g_hooks.version = CKDYN_API_VERSION;
  g_hooks.size = sizeof(CkDynHooks);
  g_hooks.mainchare = hMainchare;
  g_hooks.construct = hConstruct;
  g_hooks.destruct = hDestruct;
  g_hooks.deliver = hDeliver;
  g_hooks.deliver_many = hDeliverMany;
  g_hooks.pack_size = hPackSize;
  g_hooks.pack = hPack;
  g_hooks.unpack = hUnpack;
  g_hooks.reduce = hReduce;
  g_hooks.future = hFuture;
  g_hooks.map_proc_num = hMapProcNum;
  g_hooks.resume_from_sync = hResumeFromSync;
  g_hooks.section_joined = hSectionJoined;
  g_hooks.error = hError;
  return &g_hooks;
}

static void registerClass(CbClass *c, bool late) {
  int n = (int)c->methods.size();
  CkDynEntryDesc *e = new CkDynEntryDesc[n];   /* leaked: registration keeps it for the run */
  for (int i = 0; i < n; i++) {
    e[i].name = c->methods[i].name;
    e[i].flags = c->methods[i].flags;
    e[i].user_id = (uint32_t)i;
  }
  CkDynClassDesc d;
  memset(&d, 0, sizeof d);
  d.name = c->name; d.kind = c->kind; d.ndims = c->ndims;
  d.num_entries = n; d.entries = e;
  d.binding_data = c;
  d.hooks = nullptr;
  c->chareIdx = late ? CkDynRegisterClassLate(&d, &c->firstEp) : CkDynRegisterClass(&d, &c->firstEp);
}

void cbRegister(CbClass *c) { registerClass(c, false); }
void cbRegisterLate(CbClass *c) { registerClass(c, true); }
