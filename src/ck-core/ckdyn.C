/*
 * CkDyn: dynamic chares for language bindings. See ckdyn.h for the API and
 * doc/internals/ckdyn.md for how the pieces fit.
 */
#include "ckdyn-impl.h"
#include "register.h"
#include "ckcheckpoint.h"
#include "LBManager.h"
#include "spanningTree.h"
#include "conv-ccs.h"
#include "trace.h"
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>

CkpvExtern(int, _currentChareType);
CkpvExtern(std::vector<void *>, chare_objs);

/* readonly */ CkGroupID _ckdynMgrGID;

/* The structs a binding copies as bytes into messages keep their sizes; a
   change here is an ABI change (bump CKDYN_API_VERSION). */
static_assert(sizeof(CkDynTarget) == 40, "CkDynTarget layout changed");
static_assert(sizeof(CkDynSectionID) == 16, "CkDynSectionID layout changed");
static_assert(sizeof(CkDynCallback) == 88, "CkDynCallback layout changed");
static_assert(sizeof(CkDynHdr) == 24, "CkDynHdr layout changed");
static_assert(sizeof(CkDynSlot) == 24, "CkDynSlot layout changed");

/* ====================================================================== */
/* Tables                                                                  */
/* ====================================================================== */

struct ClassRec {
  CkDynClassDesc desc;                 /* names copied; entries points at `entries` */
  std::vector<CkDynEntryDesc> entries;
  const CkDynHooks *hooks;
  int chareIdx = -1;
  int ctorEp = -1;
  int firstEp = -1;
  int migEp = -1;
  int relayFirst = -1;   /* main chares: the CkDynMgr relay ep of entries[1] */
};

struct EpRec {
  ClassRec *cls = nullptr;
  uint32_t flags = 0;
  uint32_t user_id = 0;
  int relayTo = -1;      /* a main chare relay ep: the main chare entry it delivers to */
};

static CkDynHooks g_hooks;
static bool g_hooksSet = false;
static bool g_lateAllowed = false;
static std::unordered_map<int, ClassRec *> g_classes;   /* chareIdx -> record */
static std::vector<EpRec> g_eps;                        /* indexed by Charm++ ep */
static CkReduction::reducerType g_foreignType = CkReduction::invalid;

static inline ClassRec *classOf(int chareIdx) {
  auto it = g_classes.find(chareIdx);
  if (it == g_classes.end()) CkAbort("CkDyn: chare index %d is not a dynamic class", chareIdx);
  return it->second;
}

static inline const EpRec &epRec(int ep) {
  if (ep < 0 || (size_t)ep >= g_eps.size() || g_eps[ep].cls == nullptr)
    CkAbort("CkDyn: entry index %d is not a dynamic entry method", ep);
  return g_eps[ep];
}

static void bindingError(const CkDynHooks *H, const char *what) {
  if (H && H->error) H->error(H->global, what);
  CkAbort("CkDyn: %s", what);
}

/* ====================================================================== */
/* Indices and local object lookup                                         */
/* ====================================================================== */

static CkArrayIndex mkIndex(int ndims, const int *index) {
  int tmp[CKDYN_MAX_DIMS];
  if (ndims < 1 || ndims > CKDYN_MAX_DIMS) CkAbort("CkDyn: array index with %d dimensions", ndims);
  for (int i = 0; i < ndims; i++) {
    tmp[i] = index[i];
    if (ndims > 3 && (index[i] < -32768 || index[i] > 32767))   /* 4-6 dimensions are stored as shorts */
      CkAbort("CkDyn: index component %d does not fit in 16 bits (%d-dimensional array)", index[i], ndims);
  }
  return CkArrayIndex(ndims, tmp);
}

static void fillIndex(const CkArrayIndex &ix, int *out, int *ndims) {
  int d = ix.getDimension();
  *ndims = d;
  if (d <= 3) for (int i = 0; i < d; i++) out[i] = ix.data()[i];
  else        for (int i = 0; i < d; i++) out[i] = ix.shortData()[i];
}

static inline CkGroupID gidOf(int idx) { CkGroupID g; g.idx = idx; return g; }

void ArrayElemDyn::fillTarget(CkDynTarget *t) const {
  t->kind = CKDYN_ARRAY;
  t->chare_idx = d.chareIdx;
  t->u.array.aid = ((CkGroupID)thisArrayID).idx;
  fillIndex(thisIndexMax, t->u.array.index, &t->u.array.ndims);
}

static ArrayElement *localArrayElem(int aid, int ndims, const int *index) {
  CProxyElement_ArrayBase p(CkArrayID(gidOf(aid)), mkIndex(ndims, index));
  return p.ckLocal();
}

/* The local C++ object a target names, or NULL. */
static Chare *localChare(const CkDynTarget *t) {
  switch (t->kind) {
    case CKDYN_MAINCHARE:
      /* main chares are not in chare_objs (init.C builds them with chare index -1) */
      if (t->u.chare.pe != CkMyPe()) return nullptr;
      return (Chare *)_mainTable[_chareTable[t->chare_idx]->mainChareType()]->getObj();
    case CKDYN_CHARE:
      if (t->u.chare.pe != CkMyPe()) return nullptr;
#ifndef CMK_CHARE_USE_PTR
      return (Chare *)CkpvAccess(chare_objs)[(CmiIntPtr)t->u.chare.ptr];
#else
      return (Chare *)t->u.chare.ptr;
#endif
    case CKDYN_GROUP:
    case CKDYN_ARRAYMAP:
      return (Chare *)CkLocalBranch(gidOf(t->u.group.gid));
    case CKDYN_NODEGROUP:
      return (Chare *)CkLocalNodeBranch(gidOf(t->u.group.gid));
    case CKDYN_ARRAY:
      return localArrayElem(t->u.array.aid, t->u.array.ndims, t->u.array.index);
  }
  return nullptr;
}

/* ====================================================================== */
/* Registration                                                            */
/* ====================================================================== */

static int ckFlagsOf(uint32_t f) {
  int r = CK_EP_NOKEEP;
  if (f & CKDYN_EP_INLINE)    r |= CK_EP_INLINE;
  if (f & CKDYN_EP_IMMEDIATE) r |= CK_EP_IMMEDIATE;
  if (f & CKDYN_EP_NOTRACE)   r |= CK_EP_TRACEDISABLE;
  if (f & CKDYN_EP_APPWORK)   r |= CK_EP_APPWORK;
  return r;
}

static void noteEp(int ep, ClassRec *c, uint32_t flags, uint32_t user_id) {
  if ((size_t)ep >= g_eps.size()) g_eps.resize(ep + 1);
  g_eps[ep].cls = c;
  g_eps[ep].flags = flags;
  g_eps[ep].user_id = user_id;
}

static int registerClass(const CkDynClassDesc *desc, int *first_ep) {
  if (desc->num_entries < 1 || desc->entries == nullptr)
    CkAbort("CkDyn: class %s has no constructor entry", desc->name ? desc->name : "?");
  if (desc->hooks && (desc->hooks->version != CKDYN_API_VERSION || desc->hooks->size != sizeof(CkDynHooks)))
    CkAbort("CkDyn: class %s: hook table built for API version %u with size %u; this libcharm has version %d, size %zu",
            desc->name ? desc->name : "?", desc->hooks->version, desc->hooks->size, CKDYN_API_VERSION, sizeof(CkDynHooks));
  ClassRec *c = new ClassRec;
  c->desc = *desc;
  c->desc.name = strdup(desc->name);
  c->entries.resize(desc->num_entries);
  for (int i = 0; i < desc->num_entries; i++) {
    c->entries[i] = desc->entries[i];
    c->entries[i].name = strdup(desc->entries[i].name ? desc->entries[i].name : "");
  }
  c->desc.entries = c->entries.data();
  c->hooks = desc->hooks ? desc->hooks : &g_hooks;
  const char *name = c->desc.name;
  int idx = -1, ctor = -1;
  CkCallFnPtr entryFn = nullptr;

  switch (desc->kind) {
    case CKDYN_MAINCHARE:
      idx = CkRegisterChare(name, sizeof(MainchareDyn), TypeMainChare);
      CkRegisterBase(idx, CkIndex_Chare::__idx);
      ctor = CkRegisterEp(c->entries[0].name, MainchareDyn::callCtor, CMessage_CkArgMsg::__idx, idx, 0);
      CkRegisterMessagePupFn(ctor, (CkMessagePupFn)CkArgMsg::ckDebugPup);
      CkRegisterMainChare(idx, ctor);
      entryFn = MainchareDyn::callEntry;
      break;
    case CKDYN_CHARE:
      idx = CkRegisterChare(name, sizeof(ChareDyn), TypeChare);
      CkRegisterBase(idx, CkIndex_Chare::__idx);
      ctor = CkRegisterEp(c->entries[0].name, ChareDyn::callCtor, CkDynMsg::__idx, idx, ckFlagsOf(c->entries[0].flags));
      CkRegisterDefaultCtor(idx, ctor);
      entryFn = ChareDyn::callEntry;
      break;
    case CKDYN_GROUP:
      idx = CkRegisterChare(name, sizeof(GroupDyn), TypeGroup);
      CkRegisterBase(idx, CkIndex_IrrGroup::__idx);
      ctor = CkRegisterEp(c->entries[0].name, GroupDyn::callCtor, CkDynMsg::__idx, idx, ckFlagsOf(c->entries[0].flags));
      CkRegisterDefaultCtor(idx, ctor);
      entryFn = GroupDyn::callEntry;
      break;
    case CKDYN_NODEGROUP:
      idx = CkRegisterChare(name, sizeof(NodeGroupDyn), TypeNodeGroup);
      CkRegisterBase(idx, CkIndex_NodeGroup::__idx);
      ctor = CkRegisterEp(c->entries[0].name, NodeGroupDyn::callCtor, CkDynMsg::__idx, idx, ckFlagsOf(c->entries[0].flags));
      CkRegisterDefaultCtor(idx, ctor);
      entryFn = NodeGroupDyn::callEntry;
      break;
    case CKDYN_ARRAY:
      /* ndims 0: any dimension count; each creation or insert gives its own */
      if (desc->ndims < 0 || desc->ndims > CKDYN_MAX_DIMS)
        CkAbort("CkDyn: array class %s has %d dimensions", name, desc->ndims);
      idx = CkRegisterChare(name, sizeof(ArrayElemDyn), TypeArray);
      CkRegisterBase(idx, CkIndex_ArrayElement::__idx);
      CkRegisterArrayDimensions(idx, desc->ndims);
      ctor = CkRegisterEp(c->entries[0].name, ArrayElemDyn::callCtor, CkDynMsg::__idx, idx, ckFlagsOf(c->entries[0].flags));
      CkRegisterDefaultCtor(idx, ctor);
      entryFn = ArrayElemDyn::callEntry;
      break;
    case CKDYN_ARRAYMAP:
      idx = CkRegisterChare(name, sizeof(ArrayMapDyn), TypeGroup);
      CkRegisterBase(idx, CkIndex_IrrGroup::__idx);
      ctor = CkRegisterEp(c->entries[0].name, ArrayMapDyn::callCtor, CkDynMsg::__idx, idx, ckFlagsOf(c->entries[0].flags));
      CkRegisterDefaultCtor(idx, ctor);
      entryFn = ArrayMapDyn::callEntry;
      break;
    default:
      CkAbort("CkDyn: class %s has an unknown kind %d", name, (int)desc->kind);
  }
  c->chareIdx = idx;
  c->ctorEp = ctor;
  c->firstEp = ctor;
  /* the main chare constructor deletes its CkArgMsg itself (flags 0); every other entry is NOKEEP */
  noteEp(ctor, c, desc->kind == CKDYN_MAINCHARE ? c->entries[0].flags : (c->entries[0].flags | CKDYN_EP_NOKEEP),
         c->entries[0].user_id);

  /* user entries: ep == first_ep + i */
  for (int i = 1; i < desc->num_entries; i++) {
    int ep = CkRegisterEp(c->entries[i].name, entryFn, CkDynMsg::__idx, idx, ckFlagsOf(c->entries[i].flags));
    if (ep != ctor + i) CkAbort("CkDyn: entry indices of class %s are not contiguous", name);
    noteEp(ep, c, c->entries[i].flags | CKDYN_EP_NOKEEP, c->entries[i].user_id);
  }

  /* migration constructors, after the user entries so the contiguity holds */
  switch (desc->kind) {
    case CKDYN_MAINCHARE:
      c->migEp = CkRegisterEp(name, MainchareDyn::callMigrate, 0, idx, 0);
      CkRegisterMigCtor(idx, c->migEp);
      break;
    case CKDYN_CHARE:   /* for checkpoint restart (ckcheckpoint.C aborts on a chare without one) */
      c->migEp = CkRegisterEp(name, ChareDyn::callMigrate, 0, idx, 0);
      CkRegisterMigCtor(idx, c->migEp);
      break;
    case CKDYN_GROUP:
      c->migEp = CkRegisterEp(name, GroupDyn::callMigrate, 0, idx, 0);
      CkRegisterMigCtor(idx, c->migEp);
      break;
    case CKDYN_NODEGROUP:
      c->migEp = CkRegisterEp(name, NodeGroupDyn::callMigrate, 0, idx, 0);
      CkRegisterMigCtor(idx, c->migEp);
      break;
    case CKDYN_ARRAY:
      c->migEp = CkRegisterEp(name, ArrayElemDyn::callMigrate, 0, idx, 0);
      CkRegisterMigCtor(idx, c->migEp);
      break;
    case CKDYN_ARRAYMAP:
      c->migEp = CkRegisterEp(name, ArrayMapDyn::callMigrate, 0, idx, 0);
      CkRegisterMigCtor(idx, c->migEp);
      break;
    default:
      break;
  }
  /* main chares: one relay entry per user entry, delivered by CkDynMgr on the
     main chare's PE, so that a callback names the main chare by chare index
     rather than by its object pointer (which a restarted process does not share) */
  if (desc->kind == CKDYN_MAINCHARE) {
    for (int i = 1; i < desc->num_entries; i++) {
      int rep = CkRegisterEp(c->entries[i].name, MainchareDyn::callRelay, CkDynMsg::__idx, idx,
                             ckFlagsOf(c->entries[i].flags));
      if (i == 1) c->relayFirst = rep;
      else if (rep != c->relayFirst + i - 1) CkAbort("CkDyn: relay entry indices of class %s are not contiguous", name);
      noteEp(rep, c, c->entries[i].flags | CKDYN_EP_NOKEEP, c->entries[i].user_id);
      g_eps[rep].relayTo = ctor + i;
    }
  }
  _chareTable[idx]->isDynamic = true;
  g_classes[idx] = c;
  if (first_ep) *first_ep = ctor;
  return idx;
}

extern "C" int CkDynRegisterClass(const CkDynClassDesc *desc, int *first_ep) {
  if (CkMyRank() != 0) CkAbort("CkDyn: CkDynRegisterClass must run on rank 0 of each process");
  return registerClass(desc, first_ep);
}

extern "C" void CkDynAllowLateRegistration(void) { g_lateAllowed = true; }

extern "C" int CkDynRegisterClassLate(const CkDynClassDesc *desc, int *first_ep) {
  if (!g_lateAllowed) CkAbort("CkDyn: late registration not enabled (CkDynAllowLateRegistration)");
  if (CkMyNodeSize() != 1)
    CkAbort("CkDyn: late registration needs one PE per process (this process has %d)", CkMyNodeSize());
  return registerClass(desc, first_ep);
}

extern "C" const CkDynClassDesc *CkDynClassOf(int chare_idx) {
  auto it = g_classes.find(chare_idx);
  return it == g_classes.end() ? nullptr : &it->second->desc;
}

bool CkDynIsDynamicChare(int chareIdx) {
  return chareIdx >= 0 && (size_t)chareIdx < _chareTable.size() && _chareTable[chareIdx]->isDynamic;
}

/* ====================================================================== */
/* Message builder                                                         */
/* ====================================================================== */

/* The data region is assembled at Finish from pieces: small ones (bytes,
 * shapes, callbacks, sections, proxies) are copied into `inl` when added,
 * large ones (buffers) are recorded by pointer and copied straight into the
 * message, so a buffer is written once. */
struct Piece {
  uint64_t off;        /* offset in the data region */
  uint64_t len;
  const void *src;     /* a caller's buffer (valid until Finish or Abandon), or NULL */
  uint64_t inlOff;     /* offset into inl when src is NULL */
};

struct CkDynMsgBuilder {
  std::vector<CkDynSlot> slots;
  std::vector<Piece> pieces;
  std::vector<char> inl;
  std::vector<char> tail;
  uint64_t dataLen = 0;       /* bytes of the data region so far */
  uint16_t flags = 0;
  uint32_t user_id = 0;
  int prio_bits = 0;
  int group_dep = -1;
  bool inUse = false;
  void reset(int nslots_hint, size_t data_hint, size_t tail_hint) {
    slots.clear(); pieces.clear(); inl.clear(); tail.clear(); dataLen = 0;
    slots.reserve(nslots_hint); inl.reserve(data_hint); tail.reserve(tail_hint);
    flags = 0; user_id = 0; prio_bits = 0; group_dep = -1; inUse = true;
  }
};

CkpvStaticDeclare(CkDynMsgBuilder *, _ckdynBuilder);
#if CMK_TRACE_ENABLED
CkpvStaticDeclare(std::vector<bool>, _ckdynCoroTraced);   /* per open CkDynCoroResume: an event was begun */
#endif

static inline size_t align8(size_t n) { return (n + 7) & ~(size_t)7; }

/* Copy n bytes into the builder now; returns their offset in the data region. */
static uint64_t appendData(CkDynMsgBuilder *b, const void *p, size_t n) {
  uint64_t off = align8(b->dataLen);
  if (n) {
    uint64_t io = b->inl.size();
    b->inl.insert(b->inl.end(), (const char *)p, (const char *)p + n);
    b->pieces.push_back(Piece{off, n, nullptr, io});
  }
  b->dataLen = off + n;
  return off;
}

/* Record n bytes to be copied into the message at Finish; the source must stay
 * valid until then. Returns their offset in the data region. */
static uint64_t deferData(CkDynMsgBuilder *b, const void *p, size_t n) {
  uint64_t off = b->dataLen;    /* immediately after the caller's previous append */
  if (n) b->pieces.push_back(Piece{off, n, p, 0});
  b->dataLen = off + n;
  return off;
}

static CkDynSlot &newSlot(CkDynMsgBuilder *b, uint8_t type) {
  b->slots.emplace_back();
  CkDynSlot &s = b->slots.back();
  memset(&s, 0, sizeof s);
  s.type = type;
  return s;
}

extern "C" CkDynMsgBuilder *CkDynMsgBegin(int nslots_hint, size_t data_hint, size_t tail_hint, int prio_bits, int group_dep) {
  CkDynMsgBuilder *b = CkpvAccess(_ckdynBuilder);
  if (b == nullptr || b->inUse) b = new CkDynMsgBuilder;   /* the cached one is busy: a nested build */
  if (CkpvAccess(_ckdynBuilder) == nullptr) CkpvAccess(_ckdynBuilder) = b;
  b->reset(nslots_hint, data_hint, tail_hint);
  b->prio_bits = prio_bits;
  b->group_dep = group_dep;
  return b;
}

static void releaseBuilder(CkDynMsgBuilder *b) {
  b->inUse = false;
  if (b != CkpvAccess(_ckdynBuilder)) delete b;
}

extern "C" void CkDynMsgSetFlags(CkDynMsgBuilder *b, uint16_t flags) { b->flags = flags; }
extern "C" void CkDynMsgSetUserId(CkDynMsgBuilder *b, uint32_t user_id) { b->user_id = user_id; }
extern "C" void CkDynMsgAddNone(CkDynMsgBuilder *b) { newSlot(b, CKDYN_SLOT_NONE); }
extern "C" void CkDynMsgAddBool(CkDynMsgBuilder *b, int v) { newSlot(b, CKDYN_SLOT_BOOL).v.i64 = v ? 1 : 0; }
extern "C" void CkDynMsgAddI64(CkDynMsgBuilder *b, int64_t v) { newSlot(b, CKDYN_SLOT_I64).v.i64 = v; }
extern "C" void CkDynMsgAddU64(CkDynMsgBuilder *b, uint64_t v) { newSlot(b, CKDYN_SLOT_U64).v.u64 = v; }
extern "C" void CkDynMsgAddF64(CkDynMsgBuilder *b, double v) { newSlot(b, CKDYN_SLOT_F64).v.f64 = v; }

static void checkLen(uint64_t len) {
  if (len > 0xffffffffull) CkAbort("CkDyn: a slot of %llu bytes exceeds the 32-bit slot length", (unsigned long long)len);
}

extern "C" void CkDynMsgAddBytes(CkDynMsgBuilder *b, const void *p, uint32_t len, uint16_t tag) {
  uint64_t off = appendData(b, p, len);
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_BYTES);
  s.len = len; s.tag = tag; s.v.off = off;
}

/* True when strides describe the contiguous layout of shape in the given order. */
static bool contiguous(const CkDynBufDesc *d, size_t itemsize) {
  if (d->ndim == 0 || d->strides == nullptr) return true;
  int64_t expect = (int64_t)itemsize;
  if (d->bflags & CKDYN_BUF_FORTRAN) {
    for (int i = 0; i < d->ndim; i++) { if (d->shape[i] > 1 && d->strides[i] != expect) return false; expect *= d->shape[i]; }
  } else {
    for (int i = d->ndim - 1; i >= 0; i--) { if (d->shape[i] > 1 && d->strides[i] != expect) return false; expect *= d->shape[i]; }
  }
  return true;
}

static size_t elemSize(uint8_t elem) {
  switch (elem) {
    case CKDYN_ELEM_I8: case CKDYN_ELEM_U8: case CKDYN_ELEM_BOOL: case CKDYN_ELEM_BYTE: return 1;
    case CKDYN_ELEM_I16: case CKDYN_ELEM_U16: case CKDYN_ELEM_F16: return 2;
    case CKDYN_ELEM_I32: case CKDYN_ELEM_U32: case CKDYN_ELEM_F32: return 4;
    case CKDYN_ELEM_I64: case CKDYN_ELEM_U64: case CKDYN_ELEM_F64: case CKDYN_ELEM_C64: return 8;
    case CKDYN_ELEM_C128: return 16;
  }
  return 1;
}

extern "C" void CkDynMsgAddBuffer(CkDynMsgBuilder *b, const CkDynBufDesc *d) {
  if (d->ndim > CKDYN_MAX_BUF_DIMS) CkAbort("CkDyn: buffer with %d dimensions", (int)d->ndim);
  if (!contiguous(d, elemSize(d->elem))) CkAbort("CkDyn: non-contiguous buffer passed to CkDynMsgAddBuffer");
  checkLen(d->len);
  /* the shape (copied now) and then the bytes (copied into the message at Finish) */
  uint64_t off = d->ndim ? appendData(b, d->shape, 8 * d->ndim) : align8(b->dataLen);
  if (!d->ndim) b->dataLen = off;
  deferData(b, d->ptr, d->len);
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_BUFFER);
  s.elem = d->elem; s.ndim = d->ndim; s.bflags = d->bflags; s.tag = d->tag;
  s.len = (uint32_t)d->len; s.v.off = off;
}

extern "C" void CkDynMsgAddFuture(CkDynMsgBuilder *b, int pe, uint32_t id) {
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_FUTURE);
  s.v.u64 = ((uint64_t)(uint32_t)pe << 32) | id;
}

extern "C" void CkDynMsgAddCallback(CkDynMsgBuilder *b, const CkDynCallback *cb) {
  uint64_t off = appendData(b, cb, sizeof *cb);
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_CALLBACK);
  s.len = sizeof *cb; s.v.off = off;
}

extern "C" void CkDynMsgAddSection(CkDynMsgBuilder *b, const CkDynSectionID *sid) {
  uint64_t off = appendData(b, sid, sizeof *sid);
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_SECTION);
  s.len = sizeof *sid; s.v.off = off;
}

extern "C" void CkDynMsgAddProxy(CkDynMsgBuilder *b, const CkDynTarget *t) {
  uint64_t off = appendData(b, t, sizeof *t);
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_PROXY);
  s.len = sizeof *t; s.v.off = off;
}

extern "C" void *CkDynMsgReserveTail(CkDynMsgBuilder *b, uint32_t len) {
  b->tail.resize(len);
  CkDynSlot &s = newSlot(b, CKDYN_SLOT_TAIL);
  s.len = len;
  return b->tail.data();
}

/* Lay the payload out into buf (which has payloadSize(b) bytes). */
static size_t payloadSize(const CkDynMsgBuilder *b) {
  return sizeof(CkDynHdr) + b->slots.size() * sizeof(CkDynSlot) + align8(b->dataLen) + b->tail.size();
}

static void writePayload(const CkDynMsgBuilder *b, char *buf, int ep) {
  CkDynHdr *h = (CkDynHdr *)buf;
  memset(h, 0, sizeof *h);
  h->magic_version = CKDYN_MAGIC;
  h->ep = (uint16_t)ep;
  h->flags = b->flags;
  h->nslots = (uint16_t)b->slots.size();
  h->user_id = b->user_id;
  h->tail_len = (uint32_t)b->tail.size();
  h->data_len = (uint32_t)align8(b->dataLen);
  char *p = buf + sizeof(CkDynHdr);
  if (!b->slots.empty()) memcpy(p, b->slots.data(), b->slots.size() * sizeof(CkDynSlot));
  p += b->slots.size() * sizeof(CkDynSlot);
  /* the data region: each piece once, alignment gaps zeroed */
  uint64_t cur = 0;
  for (const Piece &pc : b->pieces) {
    if (pc.off > cur) memset(p + cur, 0, pc.off - cur);
    memcpy(p + pc.off, pc.src ? pc.src : b->inl.data() + pc.inlOff, pc.len);
    cur = pc.off + pc.len;
  }
  if (h->data_len > cur) memset(p + cur, 0, h->data_len - cur);
  p += h->data_len;
  if (!b->tail.empty()) memcpy(p, b->tail.data(), b->tail.size());
}

extern "C" CkDynMsg *CkDynMsgFinish(CkDynMsgBuilder *b, int ep) {
  size_t psize = payloadSize(b);
  if (psize > 0x7fffffffu) CkAbort("CkDyn: message payload of %zu bytes exceeds the 2 GB message limit", psize);
  if (b->slots.size() > 0xffff) CkAbort("CkDyn: %zu slots exceed the 65535 slot limit", b->slots.size());
  int total = (int)psize;
  CkDynMsg *m;
  if (b->group_dep >= 0) {
    m = new (total, b->prio_bits, GroupDepNum{1}) CkDynMsg;
    UsrToEnv(m)->setGroupDep(gidOf(b->group_dep));
  } else {
    m = new (total, b->prio_bits) CkDynMsg;
  }
  setMemoryTypeMessage(UsrToEnv(m));
  UsrToEnv(m)->setRef(0);   /* the envelope's ref is not initialized by allocation */
  writePayload(b, m->buf, ep);
  releaseBuilder(b);
  return m;
}

extern "C" void CkDynMsgAbandon(CkDynMsgBuilder *b) { releaseBuilder(b); }
extern "C" void *CkDynMsgPriorityPtr(CkDynMsg *m) { return CkPriorityPtr(m); }
extern "C" void CkDynMsgSetQueueing(CkDynMsg *m, int queueing) { CkSetQueueing(m, queueing); }
extern "C" void CkDynMsgFree(CkDynMsg *m) { CkFreeMsg(m); }
extern "C" const void *CkDynMsgPayload(CkDynMsg *m, uint32_t *len) {
  const CkDynHdr *h = (const CkDynHdr *)m->buf;
  *len = (uint32_t)(sizeof(CkDynHdr) + h->nslots * sizeof(CkDynSlot) + h->data_len + h->tail_len);
  return m->buf;
}

static inline void stampEp(CkDynMsg *m, int ep) { ((CkDynHdr *)m->buf)->ep = (uint16_t)ep; }

/* ====================================================================== */
/* Views                                                                   */
/* ====================================================================== */

static uint32_t refOf(void *msg);

static void viewPayload(CkDynMsgView &v, const char *buf, void *msg, CkDynMsgKind kind) {
  const CkDynHdr *h = (const CkDynHdr *)buf;
  if (h->magic_version != CKDYN_MAGIC) CkAbort("CkDyn: message without a CkDyn header (magic 0x%x)", h->magic_version);
  v.kind = kind;
  v.hdr = h;
  v.slots = (const CkDynSlot *)(buf + sizeof(CkDynHdr));
  v.nslots = h->nslots;
  v.data = buf + sizeof(CkDynHdr) + h->nslots * sizeof(CkDynSlot);
  v.tail_len = h->tail_len;
  v.tail = h->tail_len ? v.data + h->data_len : nullptr;
  v.refnum = msg ? refOf(msg) : 0;
  v.section = nullptr;
  v.msg = msg;
}

static uint8_t elemOfReducer(int type);

/* The envelope's reference number as a future id: 0 when none. A reduction
 * result carries the reduction's user flag, -1 by default (ckreduction.C
 * CkSetRefNum(result, getUserFlag())), unless the callback set one; that
 * all-ones value means none here, so a future id must stay below it. */
static uint32_t refOf(void *msg) {
  CMK_REFNUM_TYPE r = UsrToEnv(msg)->getRef();
  if (r == (CMK_REFNUM_TYPE)~(CMK_REFNUM_TYPE)0) return 0;
  return (uint32_t)r;
}

/* A reduction result: one BUFFER slot over the reduction data, plus a FUTURE
 * slot when the envelope carries a reference number. tmp must hold 2 slots. */
static void viewReduction(CkDynMsgView &v, CkDynSlot *tmp, CkReductionMsg *rm) {
  if (g_foreignType != CkReduction::invalid && rm->getReducer() == g_foreignType) {
    viewPayload(v, (const char *)rm->getData(), rm, CKDYN_MSGKIND_DYN);
    return;
  }
  memset(tmp, 0, 2 * sizeof(CkDynSlot));
  tmp[0].type = CKDYN_SLOT_BUFFER;
  tmp[0].elem = elemOfReducer(rm->getReducer());
  tmp[0].len = rm->getSize();
  tmp[0].v.off = 0;
  int n = 1;
  uint32_t ref = refOf(rm);
  if (ref) { tmp[1].type = CKDYN_SLOT_FUTURE; tmp[1].v.u64 = ((uint64_t)(uint32_t)CkMyPe() << 32) | ref; n = 2; }
  v.kind = CKDYN_MSGKIND_REDRESULT;
  v.hdr = nullptr;
  v.slots = tmp; v.nslots = n;
  v.data = (const char *)rm->getData();
  v.tail = nullptr; v.tail_len = 0;
  v.refnum = ref;
  v.section = nullptr;
  v.msg = rm;
}

static void viewSys(CkDynMsgView &v, CkDynSlot *tmp, void *msg) {
  memset(tmp, 0, sizeof(CkDynSlot));
  uint32_t ref = refOf(msg);
  int n = 0;
  if (ref) { tmp[0].type = CKDYN_SLOT_FUTURE; tmp[0].v.u64 = ((uint64_t)(uint32_t)CkMyPe() << 32) | ref; n = 1; }
  v.kind = CKDYN_MSGKIND_SYS;
  v.hdr = nullptr;
  v.slots = tmp; v.nslots = n;
  v.data = nullptr; v.tail = nullptr; v.tail_len = 0;
  v.refnum = ref;
  v.section = nullptr;
  v.msg = msg;
}

/* View any message that reaches a CkDyn entry; returns the entry index. */
static int viewAny(CkDynMsgView &v, CkDynSlot *tmp, void *msg) {
  envelope *env = UsrToEnv(msg);
  int mi = env->getMsgIdx();
  if (mi == CkDynMsg::__idx) {
    viewPayload(v, ((CkDynMsg *)msg)->buf, msg, CKDYN_MSGKIND_DYN);
    return v.hdr->ep;
  }
  if (mi == CkDynMcastMsg::__idx) {
    viewPayload(v, ((CkDynMcastMsg *)msg)->buf, msg, CKDYN_MSGKIND_MCAST);
    return v.hdr->ep;
  }
  if (mi == CkReductionMsg::__idx) {
    viewReduction(v, tmp, (CkReductionMsg *)msg);
    return CkMessageToEpIdx(msg);
  }
  viewSys(v, tmp, msg);
  return CkMessageToEpIdx(msg);
}

static int epOf(void *msg) {
  envelope *env = UsrToEnv(msg);
  int mi = env->getMsgIdx();
  if (mi == CkDynMsg::__idx) return ((const CkDynHdr *)((CkDynMsg *)msg)->buf)->ep;
  if (mi == CkDynMcastMsg::__idx) return ((const CkDynHdr *)((CkDynMcastMsg *)msg)->buf)->ep;
  return CkMessageToEpIdx(msg);
}

static inline const CkDynSlot *slotAt(const CkDynMsgView *v, int i) {
  if (i < 0 || i >= v->nslots) CkAbort("CkDyn: slot %d out of range (%d slots)", i, v->nslots);
  return &v->slots[i];
}

extern "C" const void *CkDynViewBufferPtr(const CkDynMsgView *v, int slot, const int64_t **shape_out) {
  const CkDynSlot *s = slotAt(v, slot);
  if (s->type != CKDYN_SLOT_BUFFER) CkAbort("CkDyn: slot %d is not a BUFFER", slot);
  const char *p = v->data + s->v.off;
  if (shape_out) *shape_out = s->ndim ? (const int64_t *)p : nullptr;
  return p + 8 * s->ndim;
}

extern "C" const void *CkDynViewBytesPtr(const CkDynMsgView *v, int slot) {
  const CkDynSlot *s = slotAt(v, slot);
  if (s->type != CKDYN_SLOT_BYTES) CkAbort("CkDyn: slot %d is not BYTES", slot);
  return v->data + s->v.off;
}

extern "C" void CkDynViewCallback(const CkDynMsgView *v, int slot, CkDynCallback *out) {
  const CkDynSlot *s = slotAt(v, slot);
  if (s->type != CKDYN_SLOT_CALLBACK) CkAbort("CkDyn: slot %d is not a CALLBACK", slot);
  memcpy(out, v->data + s->v.off, sizeof *out);
}

extern "C" void CkDynViewSection(const CkDynMsgView *v, int slot, CkDynSectionID *out) {
  const CkDynSlot *s = slotAt(v, slot);
  if (s->type != CKDYN_SLOT_SECTION) CkAbort("CkDyn: slot %d is not a SECTION", slot);
  memcpy(out, v->data + s->v.off, sizeof *out);
}

extern "C" void CkDynViewProxy(const CkDynMsgView *v, int slot, CkDynTarget *out) {
  const CkDynSlot *s = slotAt(v, slot);
  if (s->type != CKDYN_SLOT_PROXY) CkAbort("CkDyn: slot %d is not a PROXY", slot);
  memcpy(out, v->data + s->v.off, sizeof *out);
}

extern "C" void CkDynMsgRetain(const CkDynMsgView *v) { CmiReference(UsrToEnv(v->msg)); }
extern "C" void CkDynMsgRelease(void *msg) { CmiFree(UsrToEnv(msg)); }

/* The first FUTURE slot of a view, or false. */
static bool findFuture(const CkDynMsgView &v, int *pe, uint32_t *id) {
  for (int i = 0; i < v.nslots; i++)
    if (v.slots[i].type == CKDYN_SLOT_FUTURE) {
      *pe = (int)(v.slots[i].v.u64 >> 32);
      *id = (uint32_t)(v.slots[i].v.u64 & 0xffffffffu);
      return true;
    }
  return false;
}

static CkDynMsg *emptyMsg() {
  CkDynMsgBuilder *b = CkDynMsgBegin(0, 0, 0, 0, -1);
  return CkDynMsgFinish(b, 0);
}

/* ====================================================================== */
/* Delivery                                                                */
/* ====================================================================== */

static void dispatch(CkDynObj &d, void *msg) {
  ClassRec *c = classOf(d.chareIdx);
  CkDynMsgView v;
  CkDynSlot tmp[2];
  int ep = viewAny(v, tmp, msg);
  const EpRec &r = epRec(ep);
  if (r.relayTo >= 0) ep = r.relayTo;   /* a non-CkDyn message through a main chare relay */
  if (d.handle == nullptr) {
    /* a branch outside a constrained group: broadcasts and callback results
       reach every branch and are dropped here; a point-to-point send is an error */
    if (v.kind != CKDYN_MSGKIND_DYN || (v.hdr->flags & CKDYN_MSG_BCAST)) return;
    bindingError(c->hooks, "entry method invoked on a branch that was not constructed (outside a constrained group?)");
  }
  c->hooks->deliver(c->desc.binding_data, d.handle, ep, r.user_id, r.flags, &v);
}

void CkDynDeliverBroadcast(void *bcastMsg, ArrayElemDyn **elems, int n) {
  if (n <= 0) return;
  ClassRec *c = classOf(elems[0]->d.chareIdx);
  CkDynMsgView v;
  CkDynSlot tmp[2];
  int ep = viewAny(v, tmp, bcastMsg);
  const EpRec &r = epRec(ep);
  if (c->hooks->deliver_many) {
    std::vector<void *> handles(n);
    for (int i = 0; i < n; i++) handles[i] = elems[i]->d.handle;
    c->hooks->deliver_many(c->desc.binding_data, handles.data(), n, ep, r.user_id, r.flags, &v);
  } else {
    /* a handler may destroy another local element: look each one up again by index */
    CkArray *arr = CkArrayID(elems[0]->ckGetArrayID()).ckLocalBranch();
    std::vector<CkArrayIndex> idx(n);
    for (int i = 0; i < n; i++) idx[i] = elems[i]->ckGetArrayIndex();
#if CMK_TRACE_ENABLED
    /* one execute event per element, as CkLocRec::invokeEntry records for a
       charmxi element; CkArray::recvBroadcast is not traced, so none is open */
    envelope *env = UsrToEnv(bcastMsg);
    const bool traced = CpvAccess(traceOn) && _entryTable[ep]->traceEnabled;
    const int locMgr = arr->getLocMgr()->ckGetGroupID().idx;
#endif
    for (int i = 0; i < n; i++) {
      ArrayElemDyn *el = static_cast<ArrayElemDyn *>(arr->lookup(idx[i]));
      if (el == nullptr || el->d.handle == nullptr) continue;
      CkCallstackPush(el);
#if CMK_TRACE_ENABLED
      if (traced) {
        CmiObjId pid = idx[i].getProjectionID();
        _TRACE_SET_OBJECT(locMgr, el->CkMigratable::ckGetID());
        _TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(), ForChareMsg, ep, env->getSrcPe(),
                                      env->getTotalsize(), &pid, el);
      }
#endif
      c->hooks->deliver(c->desc.binding_data, el->d.handle, ep, r.user_id, r.flags, &v);
#if CMK_TRACE_ENABLED
      if (traced) { _TRACE_END_EXECUTE(); _TRACE_CLEAR_OBJECT(); }
#endif
      CkCallstackPop(el);
    }
  }
}

/* ---- MainchareDyn ------------------------------------------------------- */

/* Once per program: a program with several main chares may call it from each. */
extern "C" void CkDynCreateManagers(void) {
  if (!_ckdynMgrGID.isZero()) return;
  _ckdynMgrGID = CProxy_CkDynMgr::ckNew();
}

MainchareDyn::MainchareDyn(CkArgMsg *m) {
  d.chareIdx = CkpvAccess(_currentChareType);
  ClassRec *c = classOf(d.chareIdx);
  CkDynCreateManagers();
  CkDynTarget t;
  t.kind = CKDYN_MAINCHARE; t.chare_idx = d.chareIdx;
  t.u.chare.pe = thishandle.onPE; t.u.chare.ptr = thishandle.objPtr;
  if (c->hooks->mainchare == nullptr) CkAbort("CkDyn: no mainchare hook");
  c->hooks->mainchare(c->desc.binding_data, &d.handle, &t, m->argc, m->argv);
  delete m;
}

MainchareDyn::MainchareDyn(CkMigrateMessage *m) : Chare(m) {}
MainchareDyn::~MainchareDyn() {
  if (d.handle) { ClassRec *c = classOf(d.chareIdx); if (c->hooks->destruct) c->hooks->destruct(c->desc.binding_data, d.handle); }
}
void MainchareDyn::callCtor(void *msg, void *obj) { new (obj) MainchareDyn((CkArgMsg *)msg); }
void MainchareDyn::callMigrate(void *msg, void *obj) { new (obj) MainchareDyn((CkMigrateMessage *)msg); }
void MainchareDyn::callEntry(void *msg, void *obj) { dispatch(((MainchareDyn *)obj)->d, msg); }
/* A relay entry: runs on the CkDynMgr branch (obj) of the main chare's PE and
   delivers to the main chare of the entry's class. */
void MainchareDyn::callRelay(void *msg, void *) {
  const EpRec &r = epRec(UsrToEnv(msg)->getEpIdx());
  MainchareDyn *m = (MainchareDyn *)_mainTable[_chareTable[r.cls->chareIdx]->mainChareType()]->getObj();
  if (m == nullptr) CkAbort("CkDyn: a callback reached main chare %s on PE %d, where it does not exist", r.cls->desc.name, CkMyPe());
  CkCallstackPush(m);
  dispatch(m->d, msg);
  CkCallstackPop(m);
}

/* ---- ChareDyn ----------------------------------------------------------- */

static void pupObj(PUP::er &p, CkDynObj &d, CkDynTarget t);

ChareDyn::ChareDyn(CkDynMsg *m) {
  /* _currentChareType is set only for main chares (init.C); a singleton's class
     comes from the constructor entry stamped in the message header */
  CkDynMsgView v;
  viewPayload(v, m->buf, m, CKDYN_MSGKIND_DYN);
  ClassRec *c = epRec(v.hdr->ep).cls;
  d.chareIdx = c->chareIdx;
  CkDynTarget t;
  t.kind = CKDYN_CHARE; t.chare_idx = d.chareIdx;
  t.u.chare.pe = thishandle.onPE; t.u.chare.ptr = thishandle.objPtr;
  d.handle = c->hooks->construct(c->desc.binding_data, &t, &v);
  int pe; uint32_t id;
  if ((v.hdr->flags & CKDYN_MSG_CREATION) && findFuture(v, &pe, &id))
    CkDynFutureSet(pe, id, emptyMsg());
}

ChareDyn::ChareDyn(CkMigrateMessage *m) : Chare(m) {}
void ChareDyn::callMigrate(void *msg, void *obj) { new (obj) ChareDyn((CkMigrateMessage *)msg); }
void ChareDyn::pup(PUP::er &p) {
  Chare::pup(p);
  CkDynTarget t;
  t.kind = CKDYN_CHARE; t.chare_idx = d.chareIdx;
  t.u.chare.pe = thishandle.onPE; t.u.chare.ptr = thishandle.objPtr;
  pupObj(p, d, t);
}
ChareDyn::~ChareDyn() {
  if (d.handle) { ClassRec *c = classOf(d.chareIdx); if (c->hooks->destruct) c->hooks->destruct(c->desc.binding_data, d.handle); }
}
void ChareDyn::callCtor(void *msg, void *obj) { new (obj) ChareDyn((CkDynMsg *)msg); }
void ChareDyn::callEntry(void *msg, void *obj) { dispatch(((ChareDyn *)obj)->d, msg); }

/* ---- pup shared by the migratable kinds --------------------------------- */

static void pupObj(PUP::er &p, CkDynObj &d, CkDynTarget t) {
  p | d.chareIdx;
  t.chare_idx = d.chareIdx;   /* the caller's copy was filled before d.chareIdx was restored */
  bool hasHandle = d.handle != nullptr;   /* false on a branch outside a constrained group */
  p | hasHandle;
  if (!hasHandle) return;
  ClassRec *c = classOf(d.chareIdx);
  const CkDynHooks *H = c->hooks;
  int reason = p.isCheckpoint() ? CKDYN_PUP_CHECKPOINT : CKDYN_PUP_MIGRATE;
  PUP::mem *mem = dynamic_cast<PUP::mem *>(&p);
  if (p.isUnpacking()) {
    size_t n; p | n;
    if (H->unpack == nullptr) CkAbort("CkDyn: no unpack hook");
    if (mem) {
      d.handle = H->unpack(c->desc.binding_data, &t, reason, mem->get_current_pointer(), n);
      mem->advance(n);
    } else {
      std::vector<char> tmp(n);
      p(tmp.data(), n);
      d.handle = H->unpack(c->desc.binding_data, &t, reason, tmp.data(), n);
    }
  } else if (p.isSizing()) {
    if (H->pack_size == nullptr) CkAbort("CkDyn: no pack_size hook");
    size_t n = H->pack_size(c->desc.binding_data, d.handle, reason | CKDYN_PUP_SIZING);
    if (n == CKDYN_PACK_REFUSED) bindingError(H, "the object cannot be packed in its current state");
    d.sizedLen = n;
    p | n;
    p((char *)nullptr, n);
  } else {
    size_t n = H->pack_size(c->desc.binding_data, d.handle, reason);
    if (n == CKDYN_PACK_REFUSED) bindingError(H, "the object cannot be packed in its current state");
    if (mem && n != d.sizedLen)
      CkAbort("CkDyn: pack_size returned %zu bytes on the packing pass but %zu on the sizing pass", n, d.sizedLen);
    p | n;
    if (mem) {
      H->pack(c->desc.binding_data, d.handle, reason, mem->get_current_pointer(), n);
      mem->advance(n);
    } else {
      std::vector<char> tmp(n);
      H->pack(c->desc.binding_data, d.handle, reason, tmp.data(), n);
      p(tmp.data(), n);
    }
  }
}

void MainchareDyn::pup(PUP::er &p) {
  Chare::pup(p);
  CkDynTarget t;
  t.kind = CKDYN_MAINCHARE; t.chare_idx = d.chareIdx;
  t.u.chare.pe = thishandle.onPE; t.u.chare.ptr = thishandle.objPtr;
  pupObj(p, d, t);
}

/* ---- GroupDyn ----------------------------------------------------------- */

/* Constructors take their class from the constructor entry stamped in the
   message header by CkDynCreate*: IrrGroup::ckGetChareType reads the per-PE
   group table, which does not hold nodegroups, and _currentChareType is set
   only for main chares. */
GroupDyn::GroupDyn(CkDynMsg *m) {
  CkDynMsgView v;
  viewPayload(v, m->buf, m, CKDYN_MSGKIND_DYN);
  ClassRec *c = epRec(v.hdr->ep).cls;
  d.chareIdx = c->chareIdx;
  bool member = true;
  if (v.hdr->flags & CKDYN_MSG_CONSTRAINED) {
    /* the last slot is a BYTES slot of int32 PE numbers */
    member = false;
    if (v.nslots > 0 && v.slots[v.nslots - 1].type == CKDYN_SLOT_BYTES) {
      const CkDynSlot &s = v.slots[v.nslots - 1];
      const int32_t *pes = (const int32_t *)(v.data + s.v.off);
      for (uint32_t i = 0; i < s.len / 4; i++) if (pes[i] == CkMyPe()) { member = true; break; }
    }
  }
  if (member) {
    CkDynTarget t;
    t.kind = CKDYN_GROUP; t.chare_idx = d.chareIdx;
    t.u.group.gid = thisgroup.idx; t.u.group.pe_or_node = CkMyPe();
    d.handle = c->hooks->construct(c->desc.binding_data, &t, &v);
  }
  int pe; uint32_t id;
  if ((v.hdr->flags & CKDYN_MSG_CREATION) && findFuture(v, &pe, &id)) {
    CkDynCallback cb; memset(&cb, 0, sizeof cb);
    cb.kind = CKDYN_CB_FUTURE; cb.pe = pe; cb.id = id;
    contribute(CkDynMakeCallback(&cb));
  }
}

GroupDyn::GroupDyn(CkMigrateMessage *m) : Group(m) {}
GroupDyn::~GroupDyn() {
  if (d.handle) { ClassRec *c = classOf(d.chareIdx); if (c->hooks->destruct) c->hooks->destruct(c->desc.binding_data, d.handle); }
}
void GroupDyn::callCtor(void *msg, void *obj) { new (obj) GroupDyn((CkDynMsg *)msg); }
void GroupDyn::callMigrate(void *msg, void *obj) { new (obj) GroupDyn((CkMigrateMessage *)msg); }
void GroupDyn::callEntry(void *msg, void *obj) { dispatch(((GroupDyn *)obj)->d, msg); }
void GroupDyn::pup(PUP::er &p) {
  Group::pup(p);
  CkDynTarget t;
  t.kind = CKDYN_GROUP; t.chare_idx = d.chareIdx;
  t.u.group.gid = thisgroup.idx; t.u.group.pe_or_node = CkMyPe();
  pupObj(p, d, t);
}

/* ---- NodeGroupDyn ------------------------------------------------------- */

NodeGroupDyn::NodeGroupDyn(CkDynMsg *m) {
  CkDynMsgView v;
  viewPayload(v, m->buf, m, CKDYN_MSGKIND_DYN);
  ClassRec *c = epRec(v.hdr->ep).cls;
  d.chareIdx = c->chareIdx;
  CkDynTarget t;
  t.kind = CKDYN_NODEGROUP; t.chare_idx = d.chareIdx;
  t.u.group.gid = thisgroup.idx; t.u.group.pe_or_node = CkMyNode();
  d.handle = c->hooks->construct(c->desc.binding_data, &t, &v);
  int pe; uint32_t id;
  if ((v.hdr->flags & CKDYN_MSG_CREATION) && findFuture(v, &pe, &id)) {
    CkDynCallback cb; memset(&cb, 0, sizeof cb);
    cb.kind = CKDYN_CB_FUTURE; cb.pe = pe; cb.id = id;
    contribute(CkDynMakeCallback(&cb));
  }
}

NodeGroupDyn::NodeGroupDyn(CkMigrateMessage *m) : NodeGroup(m) {}
NodeGroupDyn::~NodeGroupDyn() {
  if (d.handle) { ClassRec *c = classOf(d.chareIdx); if (c->hooks->destruct) c->hooks->destruct(c->desc.binding_data, d.handle); }
}
void NodeGroupDyn::callCtor(void *msg, void *obj) { new (obj) NodeGroupDyn((CkDynMsg *)msg); }
void NodeGroupDyn::callMigrate(void *msg, void *obj) { new (obj) NodeGroupDyn((CkMigrateMessage *)msg); }

void NodeGroupDyn::callEntry(void *msg, void *obj) {
  NodeGroupDyn *g = (NodeGroupDyn *)obj;
  int ep = epOf(msg);
  const EpRec &r = epRec(ep);
  if (r.flags & CKDYN_EP_EXCLUSIVE) {
    /* the same protocol charmxi generates for [exclusive]: try the node lock,
       and if another PE holds it, re-send a copy to this node */
    if (CmiTryLock(g->__nodelock) != 0) {
      void *copy = CkCopyMsg(&msg);
      CkSendMsgNodeBranch(ep, copy, CkMyNode(), g->thisgroup);
      return;
    }
    dispatch(g->d, msg);
    CmiUnlock(g->__nodelock);
  } else {
    dispatch(g->d, msg);
  }
}

void NodeGroupDyn::pup(PUP::er &p) {
  NodeGroup::pup(p);
  CkDynTarget t;
  t.kind = CKDYN_NODEGROUP; t.chare_idx = d.chareIdx;
  t.u.group.gid = thisgroup.idx; t.u.group.pe_or_node = CkMyNode();
  pupObj(p, d, t);
}

/* ---- ArrayElemDyn ------------------------------------------------------- */

/* A class registered with ndims 0 takes the dimension count of the first
   index this process sees for it, so that the .sts file and the projections
   log record that many index fields, as for a charmxi array. Only the trace
   modules read _chareTable[]->ndims. */
static void noteDims(int chareIdx, int nd) {
  ChareInfo *ci = _chareTable[chareIdx];
  if (ci->ndims == 0 && classOf(chareIdx)->desc.ndims == 0) ci->ndims = nd;
}

ArrayElemDyn::ArrayElemDyn(CkDynMsg *m) {
  CkDynMsgView v;
  viewPayload(v, m->buf, m, CKDYN_MSGKIND_DYN);
  ClassRec *c = epRec(v.hdr->ep).cls;
  d.chareIdx = c->chareIdx;
  usesAtSync = (v.hdr->flags & CKDYN_MSG_USESATSYNC) != 0;
  CkDynTarget t;
  fillTarget(&t);
  noteDims(d.chareIdx, t.u.array.ndims);
  d.handle = c->hooks->construct(c->desc.binding_data, &t, &v);
  int pe; uint32_t id;
  if ((v.hdr->flags & CKDYN_MSG_CREATION) && findFuture(v, &pe, &id)) {
    CkDynCallback cb; memset(&cb, 0, sizeof cb);
    cb.kind = CKDYN_CB_FUTURE; cb.pe = pe; cb.id = id;
    contribute(CkDynMakeCallback(&cb));
  }
}

ArrayElemDyn::ArrayElemDyn(CkMigrateMessage *m) : ArrayElement(m) {}
ArrayElemDyn::~ArrayElemDyn() {
  if (d.handle) { ClassRec *c = classOf(d.chareIdx); if (c->hooks->destruct) c->hooks->destruct(c->desc.binding_data, d.handle); }
}
void ArrayElemDyn::callCtor(void *msg, void *obj) { new (obj) ArrayElemDyn((CkDynMsg *)msg); }
void ArrayElemDyn::callMigrate(void *msg, void *obj) { new (obj) ArrayElemDyn((CkMigrateMessage *)msg); }
void ArrayElemDyn::callEntry(void *msg, void *obj) { dispatch(((ArrayElemDyn *)obj)->d, msg); }

void ArrayElemDyn::ResumeFromSync() {
  if (!usesAtSync) return;   /* the LB framework calls this on every element every period */
  ClassRec *c = classOf(d.chareIdx);
  if (c->hooks->resume_from_sync) c->hooks->resume_from_sync(c->desc.binding_data, d.handle);
}

void ArrayElemDyn::pup(PUP::er &p) {
  ArrayElement::pup(p);
  CkDynTarget t;
  fillTarget(&t);
  pupObj(p, d, t);
  if (p.isUnpacking()) noteDims(d.chareIdx, t.u.array.ndims);
}

/* ---- ArrayMapDyn -------------------------------------------------------- */

ArrayMapDyn::ArrayMapDyn(CkDynMsg *m) {
  CkDynMsgView v;
  viewPayload(v, m->buf, m, CKDYN_MSGKIND_DYN);
  ClassRec *c = epRec(v.hdr->ep).cls;
  d.chareIdx = c->chareIdx;
  CkDynTarget t;
  t.kind = CKDYN_ARRAYMAP; t.chare_idx = d.chareIdx;
  t.u.group.gid = thisgroup.idx; t.u.group.pe_or_node = CkMyPe();
  d.handle = c->hooks->construct(c->desc.binding_data, &t, &v);
}

ArrayMapDyn::ArrayMapDyn(CkMigrateMessage *m) : CkArrayMap(m) {}
void ArrayMapDyn::callMigrate(void *msg, void *obj) { new (obj) ArrayMapDyn((CkMigrateMessage *)msg); }
void ArrayMapDyn::pup(PUP::er &p) {
  CkArrayMap::pup(p);
  CkDynTarget t;
  t.kind = CKDYN_ARRAYMAP; t.chare_idx = d.chareIdx;
  t.u.group.gid = thisgroup.idx; t.u.group.pe_or_node = CkMyPe();
  pupObj(p, d, t);
}
ArrayMapDyn::~ArrayMapDyn() {
  if (d.handle) { ClassRec *c = classOf(d.chareIdx); if (c->hooks->destruct) c->hooks->destruct(c->desc.binding_data, d.handle); }
}
void ArrayMapDyn::callCtor(void *msg, void *obj) { new (obj) ArrayMapDyn((CkDynMsg *)msg); }
void ArrayMapDyn::callEntry(void *msg, void *obj) { dispatch(((ArrayMapDyn *)obj)->d, msg); }

int ArrayMapDyn::procNum(int arrayHdl, const CkArrayIndex &idx) {
  ClassRec *c = classOf(d.chareIdx);
  if (c->hooks->map_proc_num == nullptr) CkAbort("CkDyn: no map_proc_num hook");
  int index[CKDYN_MAX_DIMS], nd;
  fillIndex(idx, index, &nd);
  return c->hooks->map_proc_num(c->desc.binding_data, d.handle, arrayHdl, nd, index);
}

/* ====================================================================== */
/* Sends                                                                   */
/* ====================================================================== */

static int sendFlags(int ep, const CkDynSendOpts *o) {
  const EpRec &r = epRec(ep);
  int f = 0;
  if (r.flags & CKDYN_EP_EXPEDITED) f |= CK_MSG_EXPEDITED;
  if (r.flags & CKDYN_EP_INLINE)    f |= CK_MSG_INLINE;
  if (r.flags & CKDYN_EP_IMMEDIATE) f |= CK_MSG_IMMEDIATE;
  if (o) {
    if (o->flags & CKDYN_SEND_EXPEDITED) f |= CK_MSG_EXPEDITED;
    if (o->flags & CKDYN_SEND_INLINE)    f |= CK_MSG_INLINE;
    if (o->flags & CKDYN_SEND_IMMEDIATE) f |= CK_MSG_IMMEDIATE;
    if (o->flags & CKDYN_SEND_NOTRACE)   f |= CK_MSG_LB_NOTRACE;
  }
  return f;
}

extern "C" void CkDynSendChare(const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o) {
  int f = sendFlags(ep, o);
  stampEp(m, ep);
  CkChareID cid; cid.onPE = t->u.chare.pe; cid.objPtr = t->u.chare.ptr;
  if (f & CK_MSG_INLINE) CkSendMsgInline(ep, m, &cid, f);
  else CkSendMsg(ep, m, &cid, f);
}

extern "C" void CkDynSendGroup(int gid, int pe, int ep, CkDynMsg *m, const CkDynSendOpts *o) {
  int f = sendFlags(ep, o);
  stampEp(m, ep);
  if (pe < 0) { ((CkDynHdr *)m->buf)->flags |= CKDYN_MSG_BCAST; CkBroadcastMsgBranch(ep, m, gidOf(gid), f); }
  else if (f & CK_MSG_INLINE) CkSendMsgBranchInline(ep, m, pe, gidOf(gid), f);
  else CkSendMsgBranch(ep, m, pe, gidOf(gid), f);
}

extern "C" void CkDynSendGroupMulti(int gid, int npes, const int *pes, int ep, CkDynMsg *m, const CkDynSendOpts *o) {
  int f = sendFlags(ep, o);
  stampEp(m, ep);
  CkSendMsgBranchMulti(ep, m, gidOf(gid), npes, pes, f);
}

extern "C" void CkDynSendNodeGroup(int gid, int node, int ep, CkDynMsg *m, const CkDynSendOpts *o) {
  int f = sendFlags(ep, o);
  stampEp(m, ep);
  if (node < 0) { ((CkDynHdr *)m->buf)->flags |= CKDYN_MSG_BCAST; CkBroadcastMsgNodeBranch(ep, m, gidOf(gid), f); }
  else if (f & CK_MSG_INLINE) CkSendMsgNodeBranchInline(ep, m, node, gidOf(gid), f);
  else CkSendMsgNodeBranch(ep, m, node, gidOf(gid), f);
}

extern "C" void CkDynSendArray(int aid, int ndims, const int *index, int ep, CkDynMsg *m, const CkDynSendOpts *o) {
  int f = sendFlags(ep, o);
  stampEp(m, ep);
  UsrToEnv(m)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *am = (CkArrayMessage *)m;
  am->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkArrayID a(gidOf(aid));
  if (index == nullptr) {
    ((CkDynHdr *)m->buf)->flags |= CKDYN_MSG_BCAST;
    CkBroadcastMsgArray(ep, am, a, f);
  } else {
    CkArrayIndex ix = mkIndex(ndims, index);
    if (f & CK_MSG_INLINE) CkSendMsgArrayInline(ep, am, a, ix, f);
    else CProxyElement_ArrayBase::ckSendWrapper(a, ix, am, ep, f);
  }
}

extern "C" void CkDynSendSection(const CkDynSectionID *, int, CkDynMcastMsg *, const CkDynSendOpts *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}

extern "C" void CkDynSendTarget(const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o) {
  switch (t->kind) {
    case CKDYN_MAINCHARE: case CKDYN_CHARE: CkDynSendChare(t, ep, m, o); break;
    case CKDYN_GROUP: case CKDYN_ARRAYMAP: CkDynSendGroup(t->u.group.gid, t->u.group.pe_or_node, ep, m, o); break;
    case CKDYN_NODEGROUP: CkDynSendNodeGroup(t->u.group.gid, t->u.group.pe_or_node, ep, m, o); break;
    case CKDYN_ARRAY: CkDynSendArray(t->u.array.aid, t->u.array.ndims, t->u.array.index, ep, m, o); break;
  }
}

/* ====================================================================== */
/* Creation                                                                */
/* ====================================================================== */

extern "C" void CkDynCreateChare(int chare_idx, int on_pe, CkDynMsg *ctor, CkDynTarget *out) {
  ClassRec *c = classOf(chare_idx);
  if (c->desc.kind != CKDYN_CHARE) CkAbort("CkDyn: CkDynCreateChare on a class that is not a chare");
  stampEp(ctor, c->ctorEp);
  CkChareID cid;
  CkCreateChare(chare_idx, c->ctorEp, ctor, &cid, on_pe < 0 ? CK_PE_ANY : on_pe);
  if (out) {
    out->kind = CKDYN_CHARE; out->chare_idx = chare_idx;
    out->u.chare.pe = cid.onPE; out->u.chare.ptr = cid.objPtr;
  }
}

extern "C" int CkDynCreateGroup(int chare_idx, CkDynMsg *ctor) {
  ClassRec *c = classOf(chare_idx);
  if (c->desc.kind != CKDYN_GROUP && c->desc.kind != CKDYN_ARRAYMAP)
    CkAbort("CkDyn: CkDynCreateGroup on a class that is not a group or array map");
  stampEp(ctor, c->ctorEp);
  return CkCreateGroup(chare_idx, c->ctorEp, ctor).idx;
}

extern "C" int CkDynCreateNodeGroup(int chare_idx, CkDynMsg *ctor) {
  ClassRec *c = classOf(chare_idx);
  if (c->desc.kind != CKDYN_NODEGROUP) CkAbort("CkDyn: CkDynCreateNodeGroup on a class that is not a nodegroup");
  stampEp(ctor, c->ctorEp);
  return CkCreateNodeGroup(chare_idx, c->ctorEp, ctor).idx;
}

extern "C" int CkDynCreateArray(int chare_idx, CkDynMsg *ctor, const CkDynArrayOpts *o) {
  ClassRec *c = classOf(chare_idx);
  if (c->desc.kind != CKDYN_ARRAY) CkAbort("CkDyn: CkDynCreateArray on a class that is not an array");
  stampEp(ctor, c->ctorEp);
  CkArrayOptions opts;
  if (o->dims != nullptr) {
    int dims[CKDYN_MAX_DIMS];
    if (c->desc.ndims != 0 && o->ndims != c->desc.ndims) CkAbort("CkDyn: array %s has %d dimensions, creation gave %d", c->desc.name, c->desc.ndims, o->ndims);
    if (o->ndims < 1 || o->ndims > CKDYN_MAX_DIMS) CkAbort("CkDyn: array %s created with %d dimensions", c->desc.name, o->ndims);
    for (int i = 0; i < o->ndims; i++) dims[i] = o->dims[i];
    opts = CkArrayOptions(o->ndims, dims);
    noteDims(chare_idx, o->ndims);
  }
  if (o->map_gid >= 0) opts.setMap(gidOf(o->map_gid));
  if (o->bind_to_aid >= 0) opts.bindTo(CkArrayID(gidOf(o->bind_to_aid)));
  if (o->anytime_migration) opts.setAnytimeMigration(true);
  if (o->static_insertion) opts.setStaticInsertion(true);
  UsrToEnv(ctor)->setMsgtype(ArrayEltInitMsg);
  CkGroupID g = CProxy_ArrayBase::ckCreateArray((CkArrayMessage *)ctor, c->ctorEp, opts);
  return g.idx;
}

extern "C" void CkDynArrayInsert(int aid, int ndims, const int *index, int on_pe, CkDynMsg *ctor) {
  /* the class is the array's: look it up through the ctor ep in the header,
     which the binding stamped by building the message for that class */
  int ctorEp = ((const CkDynHdr *)ctor->buf)->ep;
  const EpRec &r = epRec(ctorEp);
  if (r.cls->ctorEp != ctorEp || r.cls->desc.kind != CKDYN_ARRAY)
    CkAbort("CkDyn: CkDynArrayInsert: the message was not finished with an array class's constructor entry");
  noteDims(r.cls->chareIdx, ndims);
  UsrToEnv(ctor)->setMsgtype(ArrayEltInitMsg);   /* ckInsertIdx buffers the insert if the array is not here yet */
  CProxy_ArrayBase(CkArrayID(gidOf(aid))).ckInsertIdx((CkArrayMessage *)ctor, ctorEp, on_pe, mkIndex(ndims, index));
}

extern "C" void CkDynArrayDoneInserting(int aid) {
  CProxy_ArrayBase(CkArrayID(gidOf(aid))).doneInserting();
}

extern "C" void CkDynArrayDestroyElement(int aid, int ndims, const int *index) {
  ArrayElement *el = localArrayElem(aid, ndims, index);
  if (el == nullptr) CkAbort("CkDyn: CkDynArrayDestroyElement on an element that is not local");
  el->ckDestroy();
}

/* ====================================================================== */
/* Callbacks and futures                                                   */
/* ====================================================================== */

/* The future id travels in the envelope's reference number; a libcharm built
   with the default 16-bit CMK_REFNUM_TYPE cannot carry ids above 65535
   (DESIGN.md 3.7: build with --with-refnum-type="unsigned int"). */
static CMK_REFNUM_TYPE checkedRefnum(uint32_t id) {
  if (id == 0 || id >= (uint32_t)(CMK_REFNUM_TYPE)~(CMK_REFNUM_TYPE)0)
    CkAbort("CkDyn: future id %u is not usable as this build's %d-byte reference number (0 and the all-ones value are reserved)", id, (int)sizeof(CMK_REFNUM_TYPE));
  return (CMK_REFNUM_TYPE)id;
}

CkCallback CkDynMakeCallback(const CkDynCallback *cb) {
  switch (cb->kind) {
    case CKDYN_CB_IGNORE: return CkCallback(CkCallback::ignore);
    case CKDYN_CB_EXIT:   return CkCallback(CkCallback::ckExit);
    case CKDYN_CB_FUTURE: {
      CkCallback c(CkIndex_CkDynMgr::deposit(0), cb->pe, _ckdynMgrGID);
      c.setRefnum(checkedRefnum(cb->id));
      return c;
    }
    case CKDYN_CB_SECTION:
      CkAbort("CkDyn: section callbacks are not implemented yet (step 2)");
    case CKDYN_CB_TARGET: {
      CkCallback c;
      const CkDynTarget &t = cb->t;
      switch (t.kind) {
        case CKDYN_MAINCHARE: {
          /* by chare index through the relay entry on CkDynMgr (see registerClass) */
          ClassRec *mc = classOf(t.chare_idx);
          int i = cb->ep - mc->firstEp;
          if (i < 1 || i >= mc->desc.num_entries)
            CkAbort("CkDyn: callback to entry %d, which is not a method of main chare %s", cb->ep, mc->desc.name);
          c = CkCallback(mc->relayFirst + i - 1, t.u.chare.pe, _ckdynMgrGID);
          break;
        }
        case CKDYN_CHARE: {
          CkChareID cid; cid.onPE = t.u.chare.pe; cid.objPtr = t.u.chare.ptr;
          c = CkCallback(cb->ep, cid);
          break;
        }
        case CKDYN_GROUP: case CKDYN_ARRAYMAP:
          c = cb->bcast ? CkCallback(cb->ep, gidOf(t.u.group.gid))
                        : CkCallback(cb->ep, t.u.group.pe_or_node, gidOf(t.u.group.gid));
          break;
        case CKDYN_NODEGROUP:
          c = cb->bcast ? CkCallback(cb->ep, gidOf(t.u.group.gid), true)
                        : CkCallback(cb->ep, t.u.group.pe_or_node, gidOf(t.u.group.gid), false, true);
          break;
        case CKDYN_ARRAY:
          c = cb->bcast ? CkCallback(cb->ep, CkArrayID(gidOf(t.u.array.aid)))
                        : CkCallback(cb->ep, mkIndex(t.u.array.ndims, t.u.array.index), CkArrayID(gidOf(t.u.array.aid)));
          break;
      }
      if (cb->refnum) c.setRefnum(checkedRefnum(cb->refnum));
      return c;
    }
  }
  CkAbort("CkDyn: unknown callback kind %d", (int)cb->kind);
}

extern "C" void CkDynCallbackSend(const CkDynCallback *cb, CkDynMsg *m) {
  if (cb->kind == CKDYN_CB_TARGET) stampEp(m, cb->ep);
  CkDynMakeCallback(cb).send(m);
}

extern "C" void CkDynFutureSet(int pe, uint32_t id, CkDynMsg *value) {
  CkSetRefNum(value, checkedRefnum(id));
  CProxy_CkDynMgr(_ckdynMgrGID)[pe].deposit(value);
}

extern "C" int CkDynGetReductionNumber(const CkDynTarget *t) {
  Chare *c = localChare(t);
  if (c == nullptr) CkAbort("CkDyn: CkDynGetReductionNumber on an object that is not local");
  switch (t->kind) {
    case CKDYN_GROUP: return ((Group *)c)->getRedNo();
    case CKDYN_ARRAY: return ((ArrayElement *)c)->getRedNo();
    default: CkAbort("CkDyn: CkDynGetReductionNumber: only groups and array elements expose a reduction number");
  }
}

/* ---- CkDynMgr ----------------------------------------------------------- */

CkDynMgr::CkDynMgr() {}
CkDynMgr::CkDynMgr(CkMigrateMessage *m) : CBase_CkDynMgr(m) {}
void CkDynMgr::pup(PUP::er &p) { CBase_CkDynMgr::pup(p); }

void CkDynMgr::deposit(CkDynMsg *m) {
  CkDynMsgView v;
  CkDynSlot tmp[2];
  viewAny(v, tmp, m);
  uint32_t id = refOf(m);
  if (g_hooks.future == nullptr) CkAbort("CkDyn: a future value arrived but the binding has no future hook");
  g_hooks.future(g_hooks.global, CkMyPe(), id, &v);
  CkFreeMsg(m);
}

void CkDynMgr::relay(CkDynMsg *m) {
  CkFreeMsg(m);
  CkAbort("CkDyn: section relay is not implemented yet (step 2)");
}

/* ====================================================================== */
/* Reductions                                                              */
/* ====================================================================== */

#define R(name, elem) { #name, CkReduction::name, elem }
#define RFAM(op) \
  R(op##_char, CKDYN_ELEM_I8), R(op##_short, CKDYN_ELEM_I16), R(op##_int, CKDYN_ELEM_I32), \
  R(op##_long, CKDYN_ELEM_I64), R(op##_long_long, CKDYN_ELEM_I64), \
  R(op##_uchar, CKDYN_ELEM_U8), R(op##_ushort, CKDYN_ELEM_U16), R(op##_uint, CKDYN_ELEM_U32), \
  R(op##_ulong, CKDYN_ELEM_U64), R(op##_ulong_long, CKDYN_ELEM_U64), \
  R(op##_float, CKDYN_ELEM_F32), R(op##_double, CKDYN_ELEM_F64)

static const CkDynReducerInfo g_reducerTable[] = {
  R(nop, CKDYN_ELEM_NONE),
  RFAM(sum), RFAM(product), RFAM(max), RFAM(min),
  R(logical_and, CKDYN_ELEM_I32), R(logical_and_int, CKDYN_ELEM_I32), R(logical_and_bool, CKDYN_ELEM_BOOL),
  R(logical_or, CKDYN_ELEM_I32), R(logical_or_int, CKDYN_ELEM_I32), R(logical_or_bool, CKDYN_ELEM_BOOL),
  R(logical_xor_int, CKDYN_ELEM_I32), R(logical_xor_bool, CKDYN_ELEM_BOOL),
  R(bitvec_and, CKDYN_ELEM_I32), R(bitvec_and_int, CKDYN_ELEM_I32), R(bitvec_and_bool, CKDYN_ELEM_BOOL),
  R(bitvec_or, CKDYN_ELEM_I32), R(bitvec_or_int, CKDYN_ELEM_I32), R(bitvec_or_bool, CKDYN_ELEM_BOOL),
  R(bitvec_xor, CKDYN_ELEM_I32), R(bitvec_xor_int, CKDYN_ELEM_I32), R(bitvec_xor_bool, CKDYN_ELEM_BOOL),
  R(random, CKDYN_ELEM_I32),
  R(concat, CKDYN_ELEM_BYTE), R(set, CKDYN_ELEM_BYTE), R(statistics, CKDYN_ELEM_BYTE), R(tuple, CKDYN_ELEM_BYTE),
};
#undef R
#undef RFAM

static const int g_numReducers = (int)(sizeof(g_reducerTable) / sizeof(g_reducerTable[0]));

static uint8_t elemOfReducer(int type) {
  for (int i = 0; i < g_numReducers; i++) if (g_reducerTable[i].type == type) return g_reducerTable[i].elem;
  return CKDYN_ELEM_BYTE;
}

extern "C" int CkDynReducers(const CkDynReducerInfo **out) { *out = g_reducerTable; return g_numReducers; }
extern "C" int CkDynForeignReducerType(void) { return (int)g_foreignType; }

/* The one Charm++ reducer for binding-defined reducers: every contribution is
 * a CkDyn payload; hooks.reduce combines them into a new payload. */
static CkReductionMsg *ckdynForeignReduce(int n, CkReductionMsg **msgs) {
  if (g_hooks.reduce == nullptr) CkAbort("CkDyn: foreign reduction without a reduce hook");
  std::vector<CkDynMsgView> views(n);
  std::vector<const CkDynMsgView *> ptrs(n);
  for (int i = 0; i < n; i++) {
    viewPayload(views[i], (const char *)msgs[i]->getData(), nullptr, CKDYN_MSGKIND_DYN);
    views[i].msg = msgs[i];
    ptrs[i] = &views[i];
  }
  uint32_t rid = views[0].hdr->user_id;
  CkDynMsgBuilder *b = CkDynMsgBegin(2, 64, 0, 0, -1);
  g_hooks.reduce(g_hooks.global, rid, ptrs.data(), n, b);
  b->user_id = rid;
  std::vector<char> out(payloadSize(b));
  writePayload(b, out.data(), 0);
  releaseBuilder(b);
  return CkReductionMsg::buildNew((int)out.size(), out.data(), g_foreignType);
}

extern "C" void CkDynContribute(const CkDynTarget *me, const CkDynRedData *d, const CkDynCallback *cb) {
  Chare *c = localChare(me);
  if (c == nullptr) CkAbort("CkDyn: contribute from an object that is not local");
  int len = (int)d->len;
  const void *data = d->data;
  if (d->type == CkReduction::nop) { len = 0; data = nullptr; }
  CkCallback ck = CkDynMakeCallback(cb);
  switch (me->kind) {
    case CKDYN_GROUP:     ((Group *)c)->contribute(len, data, (CkReduction::reducerType)d->type, ck); break;
    case CKDYN_NODEGROUP: ((NodeGroup *)c)->contribute(len, data, (CkReduction::reducerType)d->type, ck); break;
    case CKDYN_ARRAY:     ((ArrayElement *)c)->contribute(len, data, (CkReduction::reducerType)d->type, ck); break;
    default: CkAbort("CkDyn: contribute from a kind without reductions");
  }
}

/* ====================================================================== */
/* Sections (step 2)                                                       */
/* ====================================================================== */

extern "C" void CkDynSectionCreateArray(int, int, int, const int *, int, const CkDynCallback *, CkDynSectionID *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}
extern "C" void CkDynSectionCreateGroup(int, int, const int *, int, const CkDynCallback *, CkDynSectionID *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}
extern "C" void CkDynSectionCombine(int, const CkDynSectionID *, const CkDynCallback *, CkDynSectionID *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}
extern "C" void CkDynSectionDelete(const CkDynSectionID *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}
extern "C" int CkDynSectionRedNo(const CkDynTarget *, const CkDynSectionID *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}
extern "C" void CkDynContributeSection(const CkDynTarget *, const CkDynSectionID *, const CkDynRedData *, const CkDynCallback *) {
  CkAbort("CkDyn: sections are not implemented yet (step 2)");
}

/* ====================================================================== */
/* Checkpoint, QD, LB, tracing                                             */
/* ====================================================================== */

extern "C" void CkDynCheckpoint(const char *dirname, const CkDynCallback *done) {
  CkStartCheckpoint(dirname, CkDynMakeCallback(done));
}

extern "C" void CkDynStartQD(const CkDynCallback *cb) { CkStartQD(CkDynMakeCallback(cb)); }

static ArrayElement *localElemOrAbort(const CkDynTarget *t, const char *what) {
  if (t->kind != CKDYN_ARRAY) CkAbort("CkDyn: %s on a target that is not an array element", what);
  ArrayElement *el = localArrayElem(t->u.array.aid, t->u.array.ndims, t->u.array.index);
  if (el == nullptr) CkAbort("CkDyn: %s on an element that is not local", what);
  return el;
}

extern "C" void CkDynAtSync(const CkDynTarget *t) { localElemOrAbort(t, "AtSync")->AtSync(); }
extern "C" void CkDynSetMigratable(const CkDynTarget *t, int migratable) { localElemOrAbort(t, "setMigratable")->setMigratable(migratable); }
extern "C" void CkDynMigrateMe(const CkDynTarget *t, int to_pe) { localElemOrAbort(t, "migrate")->ckMigrate(to_pe); }
extern "C" void CkDynLBInstrument(int on) {
#if CMK_LBDB_ON
  if (on) LBTurnInstrumentOn(); else LBTurnInstrumentOff();
#else
  (void)on;
#endif
}
extern "C" void CkDynSetObjLoad(const CkDynTarget *t, double secs) { localElemOrAbort(t, "setObjLoad")->setObjTime(secs); }
extern "C" void CkDynCallstackPush(const CkDynTarget *t) {
  Chare *c = localChare(t);
  if (c) CkCallstackPush(c);
}
extern "C" void CkDynCallstackPop(const CkDynTarget *t) {
  Chare *c = localChare(t);
  if (c) CkCallstackPop(c);
}

/* The thread listeners of a [threaded] entry do the same for a CthThread:
   CkChareThreadListener_resume/suspend (ck.C) for the running-object stack and
   traceThreadListener_resume/suspend (trace-common.C) for the trace events. */
extern "C" void CkDynCoroResume(const CkDynTarget *t, int ep, uint64_t coro_id) {
  Chare *c = localChare(t);
  if (c == nullptr) CkAbort("CkDyn: CkDynCoroResume on an object that is not local");
  CkCallstackPush(c);
#if CMK_TRACE_ENABLED
  if (CpvAccess(traceOn) && ep >= 0 && (size_t)ep < _entryTable.size() && _entryTable[ep]->traceEnabled) {
    CmiObjId id;
    if (t->kind == CKDYN_ARRAY) {
      ArrayElement *el = (ArrayElement *)c;
      id = el->ckGetArrayIndex().getProjectionID();
      _TRACE_SET_OBJECT(CkArrayID(el->ckGetArrayID()).ckLocalBranch()->getLocMgr()->ckGetGroupID().idx,
                        el->CkMigratable::ckGetID());
    } else {
      /* as CthSetThreadID does for a thread: PE, then the id */
      id.id[0] = CkMyPe();
      id.id[1] = (int)(uint32_t)coro_id;
      id.id[2] = (int)(uint32_t)(coro_id >> 32);
      id.id[3] = 0;
    }
    _TRACE_BEGIN_EXECUTE_DETAILED(-1, ForChareMsg, ep, CkMyPe(), 0, &id, c);
    CkpvAccess(_ckdynCoroTraced).push_back(true);
  } else {
    CkpvAccess(_ckdynCoroTraced).push_back(false);
  }
#else
  (void)ep; (void)coro_id;
#endif
}

extern "C" void CkDynCoroSuspend(const CkDynTarget *t) {
  Chare *c = localChare(t);
  /* A coroutine may destroy its own element and then return; the element is then no
     longer local but its (nulled) entry is still on the running-object stack. */
  if (c == nullptr) c = CkActiveObj();
#if CMK_TRACE_ENABLED
  /* end the event only if Resume began one (tracing may have been switched on in between) */
  auto &tr = CkpvAccess(_ckdynCoroTraced);
  if (tr.empty()) CkAbort("CkDyn: CkDynCoroSuspend without a matching CkDynCoroResume");
  bool began = tr.back();
  tr.pop_back();
  if (began) { _TRACE_END_EXECUTE(); _TRACE_CLEAR_OBJECT(); }
#endif
  CkCallstackPop(c);
}

extern "C" int  CkDynTraceRegisterUserEvent(const char *desc, int id) { return traceRegisterUserEvent(desc, id); }
extern "C" void CkDynTraceBeginUserBracketEvent(int id) { traceBeginUserBracketEvent(id); }
extern "C" void CkDynTraceEndUserBracketEvent(int id) { traceEndUserBracketEvent(id); }
extern "C" void CkDynTraceBeginEntry(int ep) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, ForChareMsg, ep, CkMyPe(), 0, NULL, NULL);
}
extern "C" void CkDynTraceEndEntry(int ep) { (void)ep; _TRACE_END_EXECUTE(); }

/* ====================================================================== */
/* Zero-copy and devices (step 3)                                          */
/* ====================================================================== */

extern "C" void CkDynMsgAddNcpy(CkDynMsgBuilder *, const CkDynBufDesc *, CkDynBufDone, void *) {
  CkAbort("CkDyn: zero-copy slots are not implemented yet (step 3)");
}
extern "C" void CkDynMsgAddDevice(CkDynMsgBuilder *, const CkDynDevDesc *, CkDynBufDone, void *) {
  CkAbort("CkDyn: device slots are not implemented yet (step 3)");
}
extern "C" void CkDynPostBuffer(const CkDynTarget *, int, int, const CkDynBufDesc *) {
  CkAbort("CkDyn: zero-copy posts are not implemented yet (step 3)");
}
extern "C" void CkDynPostDeviceBuffer(const CkDynTarget *, int, int, const CkDynDevDesc *) {
  CkAbort("CkDyn: device posts are not implemented yet (step 3)");
}
extern "C" void CkDynHapiAddCallback(uint64_t, const CkDynCallback *) {
  CkAbort("CkDyn: HAPI callbacks are not implemented yet (step 3)");
}
extern "C" int CkDynHasDeviceSupport(void) {
#if CMK_CUDA
  return 1;
#else
  return 0;
#endif
}

/* ====================================================================== */
/* Runtime wrappers                                                        */
/* ====================================================================== */

extern "C" int    CkDynMyPe(void) { return CkMyPe(); }
extern "C" int    CkDynNumPes(void) { return CkNumPes(); }
extern "C" int    CkDynMyNode(void) { return CkMyNode(); }
extern "C" int    CkDynNumNodes(void) { return CkNumNodes(); }
extern "C" int    CkDynMyRank(void) { return CkMyRank(); }
extern "C" int    CkDynNodeSize(int node) { return CmiNodeSize(node); }
extern "C" int    CkDynNodeFirst(int node) { return CmiNodeFirst(node); }
extern "C" int    CkDynNodeOf(int pe) { return CmiNodeOf(pe); }
extern "C" void   CkDynExit(int code) { realCkExit(code); }
extern "C" void   CkDynAbort(const char *msg) { CkAbort("%s", msg); }
extern "C" void   CkDynPrintf(const char *text) { CmiPrintf("%s", text); }
extern "C" double CkDynWallTimer(void) { return CkWallTimer(); }
extern "C" int    CkDynNumPhysicalNodes(void) { return CmiNumPhysicalNodes(); }
extern "C" int    CkDynPhysicalNodeID(int pe) { return CmiPhysicalNodeID(pe); }
extern "C" int    CkDynPhysicalRank(int pe) { return CmiPhysicalRank(pe); }
extern "C" int    CkDynNumPesOnPhysicalNode(int node) { return CmiNumPesOnPhysicalNode(node); }
extern "C" int    CkDynGetPesOnPhysicalNode(int node, int *pes, int max) {
  int *list = nullptr, n = 0;
  CmiGetPesOnPhysicalNode(node, &list, &n);
  for (int i = 0; i < n && i < max; i++) pes[i] = list[i];
  return n;
}
extern "C" int CkDynTopoTreeEdges(int pe, int root, const int *pes, int npes, int bfactor,
                                  int *parent, int *children, int max) {
  std::vector<int> tmp;
  if (pes) tmp.assign(pes, pes + npes);
  int count = 0, *kids = nullptr;
  getPETopoTreeEdges(pe, root, pes ? tmp.data() : nullptr, pes ? npes : 0, (unsigned)bfactor, parent, &count, &kids);
  for (int i = 0; i < count && i < max; i++) children[i] = kids[i];
  if (count > 0 && kids) free(kids);   /* spanningTree.C mallocs the list only for a PE with children */
  return count;
}
extern "C" void CkDynCallAfter(void (*fn)(void *arg, double now), void *arg, double msecs) {
  CcdCallFnAfter((CcdVoidFn)fn, arg, msecs);
}
extern "C" int CkDynIsRestarting(void) { return _chareRestored ? 1 : 0; }
extern "C" int CkDynHasLB(void) {
#if CMK_LBDB_ON
  return 1;
#else
  return 0;
#endif
}

/* ---- CCS ---------------------------------------------------------------- */

struct CcsRec { CkDynCcsHandler fn; std::string name; };

static void ccsTrampoline(void *userPtr, int reqLen, const void *reqData) {
  CcsRec *r = (CcsRec *)userPtr;
  r->fn(g_hooks.global, r->name.c_str(), reqData, reqLen);
}

extern "C" void CkDynCcsRegister(const char *name, CkDynCcsHandler fn) {
  CcsRec *r = new CcsRec{fn, name};
  CcsRegisterHandlerFn(r->name.c_str(), ccsTrampoline, r);
}
extern "C" int CkDynCcsIsRemoteRequest(void) { return CcsIsRemoteRequest(); }
extern "C" void CkDynCcsSendReply(int len, const void *data) { CcsSendReply(len, data); }
extern "C" void *CkDynCcsDelayReply(void) {
#if CMK_CCS_AVAILABLE
  return new CcsDelayedReply(CcsDelayReply());
#else
  return nullptr;
#endif
}
extern "C" void CkDynCcsSendDelayedReply(void *token, int len, const void *data) {
#if CMK_CCS_AVAILABLE
  CcsDelayedReply *r = (CcsDelayedReply *)token;
  CcsSendDelayedReply(*r, len, data);
  delete r;
#else
  (void)token; (void)len; (void)data;
#endif
}

/* ====================================================================== */
/* Start-up, readonly, hooks                                               */
/* ====================================================================== */

extern "C" int CkDynApiVersion(void) { return CKDYN_API_VERSION; }
extern "C" int CkDynRefnumBytes(void) { return (int)sizeof(CMK_REFNUM_TYPE); }

/* The binding's readonly data, registered as one Charm++ readonly whose pup
 * calls the hooks: the start-up broadcast and the checkpoint file both go
 * through it. */
static void ckdynRoPup(void *pup_er) {
  PUP::er &p = *(PUP::er *)pup_er;
  PUP::mem *mem = dynamic_cast<PUP::mem *>(&p);
  if (!p.isUnpacking()) {
    size_t n = g_hooks.pack_readonly_size ? g_hooks.pack_readonly_size(g_hooks.global) : 0;
    p | n;
    if (p.isSizing()) { p((char *)nullptr, n); return; }
    if (n == 0) return;
    if (mem) { g_hooks.pack_readonly(g_hooks.global, mem->get_current_pointer(), n); mem->advance(n); }
    else { std::vector<char> tmp(n); g_hooks.pack_readonly(g_hooks.global, tmp.data(), n); p(tmp.data(), n); }
  } else {
    size_t n; p | n;
    if (g_hooks.unpack_readonly == nullptr) { if (n) CkAbort("CkDyn: readonly data without an unpack hook"); return; }
    if (mem) { g_hooks.unpack_readonly(g_hooks.global, mem->get_current_pointer(), n); mem->advance(n); }
    else { std::vector<char> tmp(n); p(tmp.data(), n); g_hooks.unpack_readonly(g_hooks.global, tmp.data(), n); }
  }
}

extern "C" void CkDynInit(const CkDynHooks *hooks) {
  if (hooks->version != CKDYN_API_VERSION || hooks->size != sizeof(CkDynHooks)) {
    /* before ConverseInit in a shared-library binding, so not CkAbort */
    fprintf(stderr, "CkDyn: binding built for API version %u with a %u-byte hook table; "
            "this libcharm has version %d and a %zu-byte table\n",
            hooks->version, hooks->size, CKDYN_API_VERSION, sizeof(CkDynHooks));
    abort();
  }
  g_hooks = *hooks;
  g_hooksSet = true;
  if (g_hooks.pack_readonly_size || g_hooks.unpack_readonly)
    CkRegisterReadonly("ckdyn_ro", "char[]", 0, nullptr, ckdynRoPup);
}

bool CkDynCallRegisterHook() {
  if (!g_hooksSet || g_hooks.register_classes == nullptr) return false;
  g_hooks.register_classes(g_hooks.global);
  return true;
}

static std::vector<std::vector<char> > s_args;
static std::vector<char *> s_argv;
void _registerConverseInitHooks();   /* init.C, C++ linkage */

extern "C" void CkDynStart(int argc, char **argv) {
  if (!g_hooksSet) CkAbort("CkDyn: CkDynStart before CkDynInit");
  s_args.resize(argc);
  s_argv.resize(argc + 1, nullptr);
  for (int i = 0; i < argc; i++) {
    s_args[i].assign(argv[i], argv[i] + strlen(argv[i]) + 1);
    s_argv[i] = s_args[i].data();
  }
  _registerConverseInitHooks();
  ConverseInit(argc, s_argv.data(), _initCharm, 0, 0);
}

/* initnode: the foreign reducer, registered once per process in module order. */
void _ckdynInitNode(void) {
  g_foreignType = CkReduction::addReducer(ckdynForeignReduce, false, "CkDyn::foreign");
}

/* initproc: per-PE state. */
void _ckdynInitProc(void) {
  CkpvInitialize(CkDynMsgBuilder *, _ckdynBuilder);
  CkpvAccess(_ckdynBuilder) = nullptr;
#if CMK_TRACE_ENABLED
  CkpvInitialize(std::vector<bool>, _ckdynCoroTraced);
#endif
}

#include "CkDyn.def.h"
