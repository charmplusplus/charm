// Insertion without a PE of an index the array's map names no PE for must
// abort with "ckInsert without a PE is not supported with this array map".
// One case per run, chosen with -map:
//   cld        CldMap, 1D 8 elements inserted with PEs, then index 8 without
//   prop       PropMap, 1D 37 elements, index 37 (past its table)
//   propempty  PropMap, 1D array created empty, index 0
//   fast       FastArrayMap, 1D 37 elements, index 37
//   hilbert    HilbertArrayMap, 2D 5x7 elements, index (5,0)
// The array is created, quiescence is awaited, and PE 0 inserts. If the
// insertion does not abort the program prints "no abort" and exits 0, which the
// Makefile counts as a failure.
#include <string.h>
#include "insert_abort.decl.h"

class Elem1 : public CBase_Elem1
{
 public:
  Elem1() {}
  Elem1(CkMigrateMessage* m) : CBase_Elem1(m) {}
};

class Elem2 : public CBase_Elem2
{
 public:
  Elem2() {}
  Elem2(CkMigrateMessage* m) : CBase_Elem2(m) {}
};

class Main : public CBase_Main
{
  std::string map;
  CProxy_Elem1 a1;
  CProxy_Elem2 a2;

 public:
  Main(CkArgMsg* m)
  {
    for (int i = 1; i + 1 < m->argc; i++)
      if (strcmp(m->argv[i], "-map") == 0) map = m->argv[i + 1];
    delete m;
    // Bounds one past the initial extent in the first dimension, as an
    // application that plans to insert there would set them.
    if (map == "cld")
    {
      CkArrayOptions opts(0);
      opts.setBounds(9);
      opts.setMap(CProxy_CldMap::ckNew());
      a1 = CProxy_Elem1::ckNew(opts);
      for (int i = 0; i < 8; i++) a1[i].insert(i % CkNumPes());
      a1.doneInserting();
    }
    else if (map == "prop" || map == "fast")
    {
      CkArrayOptions opts(37);
      opts.setBounds(38);
      opts.setMap(map == "prop" ? CkGroupID(CProxy_PropMap::ckNew())
                                : CkGroupID(CProxy_FastArrayMap::ckNew()));
      a1 = CProxy_Elem1::ckNew(opts);
    }
    else if (map == "propempty")
    {
      CkArrayOptions opts(0);
      opts.setBounds(1);
      opts.setMap(CProxy_PropMap::ckNew());
      a1 = CProxy_Elem1::ckNew(opts);
    }
    else if (map == "hilbert")
    {
      CkArrayOptions opts(5, 7);
      opts.setBounds(6, 7);
      opts.setMap(CProxy_HilbertArrayMap::ckNew());
      a2 = CProxy_Elem2::ckNew(opts);
    }
    else
      CkAbort("insert_abort: -map cld|prop|propempty|fast|hilbert, got \"%s\"\n", map.c_str());
    CkPrintf("insert_abort: %s, %d PEs\n", map.c_str(), CkNumPes());
    CkStartQD(CkCallback(CkIndex_Main::created(), thisProxy));
  }

  void created(void)
  {
    if (map == "cld") a1[8].insert();
    else if (map == "prop" || map == "fast") a1[37].insert();
    else if (map == "propempty") a1[0].insert();
    else a2(5, 0).insert();
    CkStartQD(CkCallback(CkIndex_Main::done(), thisProxy));
  }

  void done(void)
  {
    CkPrintf("insert_abort: %s: no abort\n", map.c_str());
    CkExit();
  }
};

#include "insert_abort.def.h"
