/*
 * CkDyn: the C++ classes behind ckdyn.h. For C++ users (the test binding in
 * tests/charm++/ckdyn) and for ckarray.C, which delivers array broadcasts to
 * dynamic elements through CkDynDeliverBroadcast. Not part of the C ABI.
 */
#ifndef CKDYN_IMPL_H
#define CKDYN_IMPL_H

#include "charm++.h"
#include "pup_stl.h"
#include "ckmulticast.h"
#include "ckdyn.h"
#include <map>
#include <memory>
#include <vector>

PUPbytes(CkDynSectionID)

class CkDynMcastMsg;
#include "CkDyn.decl.h"

/* ---- the two message types ---------------------------------------------- */

/* Layout-identical to CkMarshallMsg: the char* at offset 0 is what the
 * zero-copy receive code expects (ckrdma.C parses descriptors from the start
 * of msgBuf). */
class CkDynMsg : public CMessage_CkDynMsg {
public:
  char *buf;
};

/* A section multicast: CkMcastBaseMsg must come first because
 * CkMulticastMgr writes the section cookie into the first bytes. `sid` names
 * the CkDyn section, so that a member can find its cookie on delivery. */
class CkDynMcastMsg : public CkMcastBaseMsg, public CMessage_CkDynMcastMsg {
public:
  char *buf;
  CkDynSectionID sid;
};

/* ---- per-object state shared by every kind ------------------------------ */

/* Compare section ids so they can key a std::map. */
inline bool operator<(const CkDynSectionID &a, const CkDynSectionID &b) {
  if (a.root_pe != b.root_pe) return a.root_pe < b.root_pe;
  if (a.idx != b.idx) return a.idx < b.idx;
  return a.gen < b.gen;
}

typedef std::map<CkDynSectionID, CkSectionInfo> CkDynCookieMap;

/* A destination posted with CkDynPostBuffer or CkDynPostDeviceBuffer for the
   next message to (this object, ep) with a zero-copy slot at index slot. */
struct CkDynPost {
  int ep;
  int slot;
  bool device;
  int device_ord;      /* DEVICE: the device ordinal the binding gave */
  void *ptr;
  uint64_t len;
};

struct CkDynObj {
  void *handle = nullptr;        /* the binding's object; never dereferenced by CkDyn */
  int   chareIdx = -1;           /* Charm++ chare index of the class */
  size_t sizedLen = 0;           /* pack_size of the sizing pass, checked on the packing pass */
  CkDynCookieMap *sectionCookies = nullptr;   /* this object's CkMulticast cookie per section */
  std::vector<CkDynPost> *posts = nullptr;     /* outstanding posts; not migrated or checkpointed */
  CkDynObj() = default;
  CkDynObj(const CkDynObj &) = delete;
  CkDynObj &operator=(const CkDynObj &) = delete;
  ~CkDynObj() { delete sectionCookies; delete posts; }
};

/* ---- the chare kinds ---------------------------------------------------- */

class MainchareDyn : public Chare {
public:
  CkDynObj d;
  MainchareDyn(CkArgMsg *m);
  MainchareDyn(CkMigrateMessage *m);
  ~MainchareDyn();
  static void callCtor(void *msg, void *obj);
  static void callEntry(void *msg, void *obj);
  static void callMigrate(void *msg, void *obj);
  static void callRelay(void *msg, void *obj);   /* a callback routed through CkDynMgr */
  void pup(PUP::er &p) override;
};

class ChareDyn : public Chare {
public:
  CkDynObj d;
  ChareDyn(CkDynMsg *m);
  ChareDyn(CkMigrateMessage *m);   /* checkpoint restart; singletons do not migrate */
  ~ChareDyn();
  static void callCtor(void *msg, void *obj);
  static void callEntry(void *msg, void *obj);
  static void callMigrate(void *msg, void *obj);
  void pup(PUP::er &p) override;
};

class GroupDyn : public Group {
public:
  CkDynObj d;
  GroupDyn(CkDynMsg *m);
  GroupDyn(CkMigrateMessage *m);
  ~GroupDyn();
  static void callCtor(void *msg, void *obj);
  static void callEntry(void *msg, void *obj);
  static void callMigrate(void *msg, void *obj);
  void pup(PUP::er &p) override;
};

class NodeGroupDyn : public NodeGroup {
public:
  CkDynObj d;
  NodeGroupDyn(CkDynMsg *m);
  NodeGroupDyn(CkMigrateMessage *m);
  ~NodeGroupDyn();
  static void callCtor(void *msg, void *obj);
  static void callEntry(void *msg, void *obj);
  static void callMigrate(void *msg, void *obj);
  void pup(PUP::er &p) override;
  int getRedNo() const { return reductionInfo.redNo; }   // NodeGroup keeps it protected
};

class ArrayElemDyn : public ArrayElement {
public:
  CkDynObj d;
  ArrayElemDyn(CkDynMsg *m);
  ArrayElemDyn(CkMigrateMessage *m);
  ~ArrayElemDyn();
  static void callCtor(void *msg, void *obj);
  static void callEntry(void *msg, void *obj);
  static void callMigrate(void *msg, void *obj);
  void ResumeFromSync() override;
  void pup(PUP::er &p) override;
  void fillTarget(CkDynTarget *t) const;
};

class ArrayMapDyn : public CkArrayMap {
public:
  CkDynObj d;
  ArrayMapDyn(CkDynMsg *m);
  ArrayMapDyn(CkMigrateMessage *m);
  ~ArrayMapDyn();
  static void callCtor(void *msg, void *obj);
  static void callEntry(void *msg, void *obj);
  static void callMigrate(void *msg, void *obj);
  void pup(PUP::er &p) override;
  int procNum(int arrayHdl, const CkArrayIndex &idx) override;
};

/* ---- the per-PE manager group ------------------------------------------- */

/* One section in its root PE's table (doc/internals/ckdyn.md, "Sections"). */
struct CkDynSection {
  bool     live = false;
  bool     ready = false;          /* proxy built and delegated; a combine waits for remote parts */
  uint8_t  kind = 0;               /* CKDYN_ARRAY, CKDYN_GROUP, or 0 for a combined section */
  uint16_t gen = 0;
  int      bfactor = 0;
  int      ndims = 0;
  std::vector<int> aids;                              /* arrays (one per sub-section) */
  std::vector<std::vector<CkArrayIndex> > elems;      /* members per array */
  int      gid = -1;                                  /* group sections */
  std::vector<int> pes;
  CProxySection_ArrayBase aproxy;
  CProxySection_Group     gproxy;
  CkDynCallback done;
  std::vector<CkDynMcastMsg *> held;                  /* sends before the section is ready */
  /* combine in progress: one record per part, filled locally or by sectionParts */
  struct Part { bool have = false; int kind = 0; int ndims = 0; std::vector<int> aids;
                std::vector<std::vector<CkArrayIndex> > elems; };
  std::vector<Part> parts;
  int partsPending = 0;
  struct Query { int combIdx, combGen, partNo, replyPe; };
  std::vector<Query> queries;                         /* member queries that arrived before ready */
};

/* A section callback: the relay slot's target (CkDynMgr::relay). sys marks a
 * slot for a system message (QD, checkpoint), which carries no reference
 * number of its own: the slot holds it in sysRefnum. */
struct CkDynRelaySlot { CkDynSectionID sid; int ep; bool sys; uint32_t sysRefnum; };

class CkDynMgr : public CBase_CkDynMgr {
public:
  CkDynMgr();
  CkDynMgr(CkMigrateMessage *m);
  void pup(PUP::er &p) override;
  /* entries */
  void deposit(CkDynMsg *m);      /* future value: hooks.future */
  void relay(CkDynMsg *m);        /* a section callback: forward the result to the section */
  void hapiDone(CkDynMsg *m);     /* a stream reached a CkDynHapiAddCallback: send to its callback */
  void sectionSend(CkDynMcastMsg *m);   /* a section send from another PE */
  void sectionQuery(int combIdx, int combGen, int partNo, int idx, int gen, int replyPe);
  void sectionParts(int combIdx, int combGen, int partNo, int kind, int ndims,
                    const std::vector<int> &aids, const std::vector<int> &counts, const std::vector<int> &flat);
  void sectionDelete(int idx, int gen);
  /* sections rooted here */
  std::vector<std::unique_ptr<CkDynSection> > sections;
  int  newSection(uint8_t kind, const CkDynCallback *done, int bfactor);
  CkDynSectionID idOf(int idx) const;
  CkDynSection &lookup(const CkDynSectionID &sid, const char *what);
  void finishSection(int idx);
  void sendTo(CkDynSection &s, int ep, CkDynMcastMsg *m);
  void addPart(int combIdx, int partNo, int kind, int ndims, const std::vector<int> &aids,
               const std::vector<std::vector<CkArrayIndex> > &elems);
  void answerQuery(const CkDynSection &s, const CkDynSection::Query &q);
  void deleteSection(const CkDynSectionID &sid);
  /* section callbacks made on this PE */
  std::vector<CkDynRelaySlot> relays;
  std::map<std::vector<uint32_t>, int> relayIndex;
  int relaySlot(const CkDynSectionID &sid, int ep, bool sys = false, uint32_t sysRefnum = 0);
};

/* ---- helpers used by ckarray.C ------------------------------------------ */

/* True when the chare index names a dynamic class. */
bool CkDynIsDynamicChare(int chareIdx);
/* Deliver one array broadcast to the given local dynamic elements with one
 * hooks.deliver_many call. */
void CkDynDeliverBroadcast(void *bcastMsg, ArrayElemDyn **elems, int n);

/* Called from _initCharm in place of CkRegisterMainModule when the program is
 * a shared-library binding: runs hooks.register_classes. Returns false when no
 * hook table was installed. */
bool CkDynCallRegisterHook();

/* Build a CkCallback from a description. sysMsg: the callback will receive a
 * system message (QD, checkpoint), so a SECTION callback's refnum is kept in
 * its relay slot instead of being read from the message. */
CkCallback CkDynMakeCallback(const CkDynCallback *cb, bool sysMsg = false);

#endif /* CKDYN_IMPL_H */
