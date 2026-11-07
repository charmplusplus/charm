/*
 * CkDyn: the C++ classes behind ckdyn.h. For C++ users (the test binding in
 * tests/charm++/ckdyn) and for ckarray.C, which delivers array broadcasts to
 * dynamic elements through CkDynDeliverBroadcast. Not part of the C ABI.
 */
#ifndef CKDYN_IMPL_H
#define CKDYN_IMPL_H

#include "charm++.h"
#include "ckdyn.h"
#include "CkDyn.decl.h"
#include <map>
#include <vector>

/* ---- the two message types ---------------------------------------------- */

/* Layout-identical to CkMarshallMsg: the char* at offset 0 is what the
 * zero-copy receive code expects (ckrdma.C parses descriptors from the start
 * of msgBuf). */
class CkDynMsg : public CMessage_CkDynMsg {
public:
  char *buf;
};

/* For CkMulticast (step 2): CkMcastBaseMsg must come first because
 * CkMulticastMgr writes the section cookie into the first bytes. */
class CkDynMcastMsg : public CkMcastBaseMsg, public CMessage_CkDynMcastMsg {
public:
  char *buf;
};

/* ---- per-object state shared by every kind ------------------------------ */

struct CkDynObj {
  void *handle = nullptr;        /* the binding's object; never dereferenced by CkDyn */
  int   chareIdx = -1;           /* Charm++ chare index of the class */
  size_t sizedLen = 0;           /* pack_size of the sizing pass, checked on the packing pass */
  std::map<CkDynSectionID, CkSectionInfo> *sectionCookies = nullptr;   /* step 2 */
};

/* Compare section ids so they can key a std::map. */
inline bool operator<(const CkDynSectionID &a, const CkDynSectionID &b) {
  if (a.root_pe != b.root_pe) return a.root_pe < b.root_pe;
  if (a.idx != b.idx) return a.idx < b.idx;
  return a.gen < b.gen;
}

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

class CkDynMgr : public CBase_CkDynMgr {
public:
  CkDynMgr();
  CkDynMgr(CkMigrateMessage *m);
  void pup(PUP::er &p) override;
  /* entries */
  void deposit(CkDynMsg *m);      /* future value: hooks.future */
  void relay(CkDynMsg *m);        /* section callback relay (step 2) */
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

/* Build a CkCallback from a description. */
CkCallback CkDynMakeCallback(const CkDynCallback *cb);

#endif /* CKDYN_IMPL_H */
