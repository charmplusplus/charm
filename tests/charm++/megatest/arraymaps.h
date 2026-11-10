#ifndef _ARRAYMAPS_H
#define _ARRAYMAPS_H

#include <vector>
#include "megatest.h"

// ARRAYMAPS_MOVE_INSERTED=0 keeps dynamically inserted elements on their PE
// and alive at the end of a case. CkArray::insertElement runs the element
// constructor without resetting currentChareIdx (CkLocMgr::addElementToRec
// does), so an inserted element takes the chare_objs slot of the last plain
// chare created on its PE. When the element is deleted there, by ckDestroy()
// or by migrating away, ~Chare clears that slot, and the next message to the
// plain chare is delivered to a null object (SIGSEGV). Set to 1 once fixed.
#ifndef ARRAYMAPS_MOVE_INSERTED
#define ARRAYMAPS_MOVE_INSERTED 0
#endif

#include "arraymaps.decl.h"

// Flat (row-major) index of an element, for the 1D, 2D and 3D shapes used here.
int arraymaps_flat(int caseId, int i);
int arraymaps_flat(int caseId, const CkIndex2D& i);
int arraymaps_flat(int caseId, const CkIndex3D& i);
// Number of index slots a case reports on (initial plus inserted elements).
int arraymaps_slots(int caseId);
// Whether an element of this case must sit on its homePe.
bool arraymaps_checkHome(int caseId, bool initial);
const char* arraymaps_mapName(int caseId);
int arraymaps_dims(int caseId);

class arraymaps_driver : public CBase_arraymaps_driver
{
 public:
  arraymaps_driver(void);
  arraymaps_driver(CkMigrateMessage* m) {}
  void phaseDone(int n, int* v);
  void inserted(int flat, int pe);

 private:
  enum Phase { CREATE, INSERT, MIGRATE, PING, DESTROY };
  int caseId;
  Phase phase;
  int nInserted;
  std::vector<int> insertedSeen;
  std::vector<int> expected;  // expected PE per flat index, -1 when not known
  CProxy_arraymaps_inserter inserter;
  CkArrayID arr;

  void startCase(void);
  void nextCase(void);
  void startMigrate(void);
  void checkPhase(int n, const int* v, bool withInserted);
};

class arraymaps_inserter : public CBase_arraymaps_inserter
{
 public:
  arraymaps_inserter(void) {}
  arraymaps_inserter(CkMigrateMessage* m) {}
  void insert(CkArrayID arr, CProxy_arraymaps_driver driver, int caseId,
              int first, int n, std::vector<int> expected);
};

// Element logic shared by the 1D, 2D and 3D arrays.
template <class Base>
class arraymaps_elt : public Base
{
 protected:
  CProxy_arraymaps_driver driver;
  int caseId;
  int destPe;
  bool inserted;  // created by insert() after the initial wave
  std::vector<int> expected;

  void checkPlacement(bool initial);
  void report(void);  // contribute this element's slot to the current phase

 public:
  arraymaps_elt(CProxy_arraymaps_driver driver_, int caseId_, bool initial,
                const std::vector<int>& expected_)
      : driver(driver_), caseId(caseId_), destPe(-1), inserted(!initial), expected(expected_)
  {
    checkPlacement(initial);
    if (initial)
      report();
    else
      driver.inserted(arraymaps_flat(caseId, this->thisIndex), CkMyPe());
  }
  arraymaps_elt(CkMigrateMessage* m) : Base(m) {}

  void pup(PUP::er& p)
  {
    Base::pup(p);
    p | driver;
    p | caseId;
    p | destPe;
    p | inserted;
    p | expected;
  }

  void migrate(void)
  {
    destPe = (inserted && !ARRAYMAPS_MOVE_INSERTED) ? CkMyPe() : (CkMyPe() + 1) % CkNumPes();
    if (destPe == CkMyPe())
      this->thisProxy[this->thisIndex].afterMigrate();
    else
      this->migrateMe(destPe);
  }
  void ckJustMigrated(void)
  {
    Base::ckJustMigrated();
    this->thisProxy[this->thisIndex].afterMigrate();
  }
  void afterMigrate(void) { checkPe("after migration"); report(); }
  void ping(void) { checkPe("on broadcast"); report(); }
  void destroy(void)
  {
    report();
    if (!inserted || ARRAYMAPS_MOVE_INSERTED) this->ckDestroy();
  }

 private:
  void checkPe(const char* when)
  {
    if (CkMyPe() != destPe)
      CkAbort("arraymaps: %s %dD element %d is on PE %d %s, expected PE %d\n",
              arraymaps_mapName(caseId), arraymaps_dims(caseId),
              arraymaps_flat(caseId, this->thisIndex), CkMyPe(), when, destPe);
  }
};

class arraymaps_1d : public arraymaps_elt<CBase_arraymaps_1d>
{
 public:
  arraymaps_1d(CProxy_arraymaps_driver d, int c, bool initial, std::vector<int> e)
      : arraymaps_elt<CBase_arraymaps_1d>(d, c, initial, e) {}
  arraymaps_1d(CkMigrateMessage* m) : arraymaps_elt<CBase_arraymaps_1d>(m) {}
};

class arraymaps_2d : public arraymaps_elt<CBase_arraymaps_2d>
{
 public:
  arraymaps_2d(CProxy_arraymaps_driver d, int c, bool initial, std::vector<int> e)
      : arraymaps_elt<CBase_arraymaps_2d>(d, c, initial, e) {}
  arraymaps_2d(CkMigrateMessage* m) : arraymaps_elt<CBase_arraymaps_2d>(m) {}
};

class arraymaps_3d : public arraymaps_elt<CBase_arraymaps_3d>
{
 public:
  arraymaps_3d(CProxy_arraymaps_driver d, int c, bool initial, std::vector<int> e)
      : arraymaps_elt<CBase_arraymaps_3d>(d, c, initial, e) {}
  arraymaps_3d(CkMigrateMessage* m) : arraymaps_elt<CBase_arraymaps_3d>(m) {}
};

#endif
