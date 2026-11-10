/*
 * arraymaps: the built-in array maps of cklocation.C.
 *
 * For each map, arrays of 37 (1D), 5x7 (2D) and 3x4x5 (3D) elements are
 * created with that map, one case at a time. Each case:
 *   1. creation: every element checks it was built on the PE the map names
 *      (CkLocMgr::homePe, and a table the test computed when it knows the
 *      placement independently); a sum reduction of [1, f(index), one slot
 *      per index] checks that every element exists exactly once;
 *   2. insertion (1D, maps that support it): elements 37..41 are inserted
 *      from PE 1 (PE 0 on one PE), checked the same way;
 *   3. every element migrates to (CkMyPe()+1) % CkNumPes() and checks it arrived;
 *   4. a broadcast reaches every element (reduction checks the full set);
 *   5. every element contributes once more and calls ckDestroy();
 *      the next case starts when that reduction arrives.
 *
 * Maps and what is left out:
 *   RRMap, DefaultArrayMap, BlockMap: 1D/2D/3D, with insertion.
 *   FastArrayMap: 1D/2D/3D. No insertion: procNum is flat/ceil(n/P), so an
 *     index past the initial size names a PE beyond CkNumPes() (the map's
 *     comment says it is for arrays that promise no late insertion).
 *   HilbertArrayMap: 1D 37, 2D 4x4, 3D 4x4x4 (see the case table for why not
 *     5x7 and 3x4x5). No insertion, as FastArrayMap (it indexes its PE list
 *     with flat/ceil(n/P)).
 *   PropMap: 1D/2D/3D. No insertion: 1D procNum indexes a table of the initial
 *     size with the element index.
 *   CldMap: 1D/2D/3D. procNum is CLD_ANYWHERE, so placement is not checked;
 *     only that each element exists exactly once. Elements are inserted with
 *     a PE; ARRAYMAPS_CLDMAP_INSERT=1 inserts without one, which must abort
 *     with "ckInsert without a PE is not supported".
 *   ConfigurableRRMap: only with the run line flag +ConfigurableRRMap
 *     "<config>", e.g. "4 2 0 1 1 0"; 1D only (populateInitial aborts on other
 *     dimensions). Initial placement is checked against the configuration and
 *     inserted elements against homePe (procNum is RRMap's).
 *   Simple1DFileMap: 1D only (it aborts on other dimensions). ReadFileMap:
 *     1D/2D/3D. PE 0 writes ./mapfile before creating the map group; every PE
 *     reads it in registerArray, so all processes must share the directory.
 *     No insertion: procNum indexes the table read from the file.
 */
#include <stdio.h>
#include <string.h>
#include <sstream>
#include "TopoManager.h"
#include "arraymaps.h"

#ifndef ARRAYMAPS_CLDMAP_INSERT
#define ARRAYMAPS_CLDMAP_INSERT 0
#endif
#ifndef ARRAYMAPS_HILBERT_ODD
#define ARRAYMAPS_HILBERT_ODD 0
#endif

enum MapKind { RR, DEFAULT, FAST, BLOCK, HILBERT, PROP, CLD, CONFIGRR, FILE1D, READFILE };

static const char* mapNames[] = {"RRMap",   "DefaultArrayMap", "FastArrayMap",
                                 "BlockMap", "HilbertArrayMap", "PropMap",
                                 "CldMap",   "ConfigurableRRMap", "Simple1DFileMap",
                                 "ReadFileMap"};

struct arraymaps_case
{
  MapKind map;
  int dims;
  bool insert;  // insert elements 37..41 after creation (1D only)
  int nx, ny, nz;  // extents; ny = nz = 1 below 2D
};

// 1D 37 (+5 inserted), 2D 5x7, 3D 3x4x5, except where a comment says otherwise.
#define D1 1, true, 37, 1, 1
#define D2 2, false, 5, 7, 1
#define D3 3, false, 3, 4, 5
static const arraymaps_case cases[] = {
    {RR, D1}, {RR, D2}, {RR, D3},
    {DEFAULT, D1}, {DEFAULT, D2}, {DEFAULT, D3},
    {BLOCK, D1}, {BLOCK, D2}, {BLOCK, D3},
    {FAST, 1, false, 37, 1, 1}, {FAST, D2}, {FAST, D3},
    // HilbertArrayMap 2D/3D use power-of-two cubes: with 5x7 and 3x4x5 the
    // curve position of some indices exceeds the element count, procNum reads
    // past its PE list, and those elements are never created (ARRAYMAPS_HILBERT_ODD).
    {HILBERT, 1, false, 37, 1, 1},
#if ARRAYMAPS_HILBERT_ODD
    {HILBERT, D2}, {HILBERT, D3},
#else
    {HILBERT, 2, false, 4, 4, 1}, {HILBERT, 3, false, 4, 4, 4},
#endif
    {PROP, 1, false, 37, 1, 1}, {PROP, D2}, {PROP, D3},
    {CLD, D1}, {CLD, D2}, {CLD, D3},
    {CONFIGRR, D1},
    {FILE1D, 1, false, 37, 1, 1},
    {READFILE, 1, false, 37, 1, 1}, {READFILE, D2}, {READFILE, D3},
};
static const int nCases = sizeof(cases) / sizeof(cases[0]);

static const int N1 = 37, NINSERT = 5;  // inserted elements get indices N1..N1+NINSERT-1

// +ConfigurableRRMap "<objs per block> <PEs per block> <location of each obj>",
// read on PE 0 by arraymaps_moduleinit without removing it from argv.
static bool crrPresent = false;
static int crrObjs = 0, crrPEs = 0;
static std::vector<int> crrLoc;

int arraymaps_flat(int c, int i) { return i; }
int arraymaps_flat(int c, const CkIndex2D& i) { return i.x * cases[c].ny + i.y; }
int arraymaps_flat(int c, const CkIndex3D& i)
{
  return (i.x * cases[c].ny + i.y) * cases[c].nz + i.z;
}

static int initialCount(int c)
{
  return cases[c].nx * cases[c].ny * cases[c].nz;
}
int arraymaps_slots(int c) { return initialCount(c) + (cases[c].insert ? NINSERT : 0); }
const char* arraymaps_mapName(int c) { return mapNames[cases[c].map]; }
int arraymaps_dims(int c) { return cases[c].dims; }
bool arraymaps_checkHome(int c, bool initial)
{
  if (cases[c].map == CLD) return false;
  if (cases[c].map == CONFIGRR && initial) return false;
  return true;
}

static int f(int flat) { return 3 * flat + 1; }

/******************** elements ********************/

template <class CBase>
void arraymaps_elt<CBase>::checkPlacement(bool initial)
{
  const int flat = arraymaps_flat(caseId, this->thisIndex);
  if (arraymaps_checkHome(caseId, initial))
  {
    const int home = this->thisProxy.ckLocMgr()->homePe(this->thisIndexMax);
    if (home != CkMyPe())
      CkAbort("arraymaps: %s %dD element %d created on PE %d, homePe is %d\n",
              arraymaps_mapName(caseId), arraymaps_dims(caseId), flat, CkMyPe(), home);
  }
  if (flat < (int)expected.size() && expected[flat] >= 0 && expected[flat] != CkMyPe())
    CkAbort("arraymaps: %s %dD element %d created on PE %d, expected PE %d\n",
            arraymaps_mapName(caseId), arraymaps_dims(caseId), flat, CkMyPe(),
            expected[flat]);
}

template <class CBase>
void arraymaps_elt<CBase>::report(void)
{
  const int flat = arraymaps_flat(caseId, this->thisIndex);
  std::vector<int> v(2 + arraymaps_slots(caseId), 0);
  v[0] = 1;
  v[1] = f(flat);
  v[2 + flat] = 1;
  this->contribute(v, CkReduction::sum_int,
                   CkCallback(CkReductionTarget(arraymaps_driver, phaseDone), driver));
}

/******************** inserter ********************/

void arraymaps_inserter::insert(CkArrayID aid, CProxy_arraymaps_driver driver,
                                int caseId, int first, int n, std::vector<int> expected)
{
  CProxy_arraymaps_1d arr(aid);
  // CldMap names no PE, so its elements are inserted with one (unless
  // ARRAYMAPS_CLDMAP_INSERT, which checks the abort without one).
  const bool withPe = cases[caseId].map == CLD && !ARRAYMAPS_CLDMAP_INSERT;
  for (int i = first; i < first + n; i++)
    if (withPe) arr[i].insert(driver, caseId, false, expected, i % CkNumPes());
    else arr[i].insert(driver, caseId, false, expected);
  arr.doneInserting();
}

/******************** driver ********************/

// Expected PE per flat index when the test knows it without asking the map;
// -1 elsewhere. File maps: PE 0 also writes ./mapfile here.
static std::vector<int> expectedTable(int c)
{
  std::vector<int> e(arraymaps_slots(c), -1);
  const int P = CkNumPes(), n = initialCount(c);
  if (cases[c].map == CONFIGRR)
  {
    for (int i = 0; i < n; i++)
      e[i] = ((i / crrObjs) * crrPEs + crrLoc[i % crrObjs]) % P;
  }
  else if (cases[c].map == FILE1D)
  {
    FILE* fp = fopen("mapfile", "w");
    if (!fp) CkAbort("arraymaps: cannot write ./mapfile\n");
    for (int i = 0; i < n; i++)
    {
      e[i] = (7 * i + 3) % P;
      fprintf(fp, "%d\n", e[i]);
    }
    fclose(fp);
  }
  else if (cases[c].map == READFILE)
  {
    // ReadFileMap maps "x y z t" through TopoManager::coordinatesToRank, so
    // write the coordinates TopoManager gives for the intended PE.
    TopoManager tmgr;
    FILE* fp = fopen("mapfile", "w");
    if (!fp) CkAbort("arraymaps: cannot write ./mapfile\n");
    for (int i = 0; i < n; i++)
    {
      int x, y, z, t;
      e[i] = (5 * i + 1) % P;
      tmgr.rankToCoordinates(e[i], x, y, z, t);
      fprintf(fp, "%d %d %d %d\n", x, y, z, t);
    }
    fclose(fp);
  }
  return e;
}

static CkGroupID createMap(MapKind m)
{
  switch (m)
  {
    case RR: return CProxy_RRMap::ckNew();
    case DEFAULT: return CProxy_DefaultArrayMap::ckNew();
    case FAST: return CProxy_FastArrayMap::ckNew();
    case BLOCK: return CProxy_BlockMap::ckNew();
    case HILBERT: return CProxy_HilbertArrayMap::ckNew();
    case PROP: return CProxy_PropMap::ckNew();
    case CLD: return CProxy_CldMap::ckNew();
    case CONFIGRR: return CProxy_ConfigurableRRMap::ckNew();
    case FILE1D: return CProxy_Simple1DFileMap::ckNew();
    case READFILE: return CProxy_ReadFileMap::ckNew();
  }
  return CkGroupID();
}

arraymaps_driver::arraymaps_driver(void)
{
  inserter = CProxy_arraymaps_inserter::ckNew();
  caseId = -1;
  nextCase();
}

void arraymaps_driver::nextCase(void)
{
  caseId++;
  // ConfigurableRRMap needs its command-line flag; without it the map has no
  // configuration to place elements by.
  while (caseId < nCases && cases[caseId].map == CONFIGRR && !crrPresent) caseId++;
  if (caseId >= nCases)
  {
    remove("mapfile");
    megatest_finish();
    return;
  }
  startCase();
}

void arraymaps_driver::startCase(void)
{
  const arraymaps_case& c = cases[caseId];
  expected = expectedTable(caseId);
  phase = CREATE;
  CkArrayOptions opts = c.dims == 1 ? CkArrayOptions(c.nx)
                        : c.dims == 2 ? CkArrayOptions(c.nx, c.ny)
                                      : CkArrayOptions(c.nx, c.ny, c.nz);
  if (c.insert) opts.setBounds(N1 + NINSERT);
  opts.setMap(createMap(c.map));
  CProxy_arraymaps_driver self(thisProxy);
  if (c.dims == 1)
    arr = CProxy_arraymaps_1d::ckNew(self, caseId, true, expected, opts);
  else if (c.dims == 2)
    arr = CProxy_arraymaps_2d::ckNew(self, caseId, true, expected, opts);
  else
    arr = CProxy_arraymaps_3d::ckNew(self, caseId, true, expected, opts);
}

// v = [count, sum f(flat), one slot per flat index]
void arraymaps_driver::checkPhase(int n, const int* v, bool withInserted)
{
  const int slots = arraymaps_slots(caseId);
  const int count = initialCount(caseId) + (withInserted ? NINSERT : 0);
  const char* name = arraymaps_mapName(caseId);
  const int dims = cases[caseId].dims;
  if (n != 2 + slots)
    CkAbort("arraymaps: %s %dD phase %d: reduction length %d, expected %d\n", name, dims,
            (int)phase, n, 2 + slots);
  long sumf = 0;
  for (int i = 0; i < count; i++) sumf += f(i);
  if (v[0] != count || v[1] != sumf)
    CkAbort("arraymaps: %s %dD phase %d: count %d sum %d, expected %d and %ld\n", name,
            dims, (int)phase, v[0], v[1], count, sumf);
  for (int i = 0; i < slots; i++)
    if (v[2 + i] != (i < count ? 1 : 0))
      CkAbort("arraymaps: %s %dD phase %d: element %d reported %d times, expected %d\n",
              name, dims, (int)phase, i, v[2 + i], i < count ? 1 : 0);
}

void arraymaps_driver::phaseDone(int n, int* v)
{
  const arraymaps_case& c = cases[caseId];
  const bool inserted = c.insert && phase != CREATE;
  checkPhase(n, v, inserted);
  switch (phase)
  {
    case CREATE:
      if (c.insert)
      {
        phase = INSERT;
        nInserted = 0;
        insertedSeen.assign(NINSERT, 0);
        // Inserted elements are placed by procNum; for ConfigurableRRMap that is
        // RRMap's, so homePe is the check for them (expected stays -1).
        inserter[CkNumPes() > 1 ? 1 : 0].insert(arr, thisProxy, caseId,
                                                N1, NINSERT, expected);
        return;
      }
      startMigrate();
      break;
    case INSERT:  // ends in inserted(), not in a reduction
      CkAbort("arraymaps: reduction during insertion\n");
      break;
    case MIGRATE:
      phase = PING;
      if (c.dims == 1) CProxy_arraymaps_1d(arr).ping();
      else if (c.dims == 2) CProxy_arraymaps_2d(arr).ping();
      else CProxy_arraymaps_3d(arr).ping();
      break;
    case PING:
      phase = DESTROY;
      if (c.dims == 1) CProxy_arraymaps_1d(arr).destroy();
      else if (c.dims == 2) CProxy_arraymaps_2d(arr).destroy();
      else CProxy_arraymaps_3d(arr).destroy();
      break;
    case DESTROY:
      nextCase();
      break;
  }
}

void arraymaps_driver::inserted(int flat, int pe)
{
  if (flat < N1 || flat >= N1 + NINSERT || insertedSeen[flat - N1]++ != 0)
    CkAbort("arraymaps: %s inserted element %d reported twice or out of range (PE %d)\n",
            arraymaps_mapName(caseId), flat, pe);
  if (++nInserted == NINSERT) startMigrate();
}

void arraymaps_driver::startMigrate(void)
{
  const int dims = cases[caseId].dims;
  phase = MIGRATE;
  if (dims == 1) CProxy_arraymaps_1d(arr).migrate();
  else if (dims == 2) CProxy_arraymaps_2d(arr).migrate();
  else CProxy_arraymaps_3d(arr).migrate();
}

/******************** megatest hooks ********************/

void arraymaps_init(void) { CProxy_arraymaps_driver::ckNew(0); }

void arraymaps_moduleinit(void)
{
  char** argv = CkGetArgv();
  for (int i = 0; argv[i] != NULL; i++)
  {
    if (strcmp(argv[i], "+ConfigurableRRMap") != 0 || argv[i + 1] == NULL) continue;
    std::istringstream in(argv[i + 1]);
    in >> crrObjs >> crrPEs;
    crrLoc.assign(crrObjs > 0 ? crrObjs : 0, 0);
    for (int k = 0; k < crrObjs; k++) in >> crrLoc[k];
    crrPresent = crrObjs > 0 && crrPEs > 0 && !in.fail();
    if (!crrPresent) CkAbort("arraymaps: cannot parse +ConfigurableRRMap \"%s\"\n", argv[i + 1]);
  }
}

MEGATEST_REGISTER_TEST(arraymaps, "kale", 0)
#include "arraymaps.def.h"
