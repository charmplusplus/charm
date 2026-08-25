/**
 * \addtogroup CkLdb
 */
/*@{*/

#include "speedupsim.h"

#include "charm++.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

#if CMK_LBDB_ON

namespace
{

typedef std::pair<double, int> LoadBin;  // (load, index)
typedef std::priority_queue<LoadBin, std::vector<LoadBin>, std::greater<LoadBin> > MinBins;

/** Pack objects onto `width` bins, longest first onto the lightest bin.
 *
 * Longest-processing-time-first is the standard greedy makespan heuristic, and
 * it is also close to what will actually happen: GreedyRefine, the balancer
 * these jobs run, is greedy on load in this same order. Using the same rule
 * the balancer uses matters more here than using the optimal one -- the
 * question is what this job will do at that width, not what it could do.
 *
 * Each bin starts at `seedLoad`, the per-processor overhead that does not
 * divide when the job spreads out.
 */
void packLPT(const std::vector<LoadBin>& descending, int width, double seedLoad,
             std::vector<double>& binLoad, std::vector<int>& objBin)
{
  binLoad.assign(width, seedLoad);

  MinBins bins;
  for (int b = 0; b < width; b++) bins.push(LoadBin(seedLoad, b));

  for (size_t i = 0; i < descending.size(); i++)
  {
    const int b = bins.top().second;
    bins.pop();
    binLoad[b] += descending[i].first;
    objBin[descending[i].second] = b;
    bins.push(LoadBin(binLoad[b], b));
  }
}

/** Resolve one end of a communication edge to a bin under the hypothetical
 *  placement. Returns -1 if the object is not in this window's data.
 *
 * Edges whose end is a bare processor rather than an object -- runtime
 * traffic, broadcasts -- have nothing to follow to a new width. Folding them
 * onto `pe % width` keeps their count in the model and spreads them the way
 * the real thing would be spread, without pretending to know where they land.
 */
inline int binOfObj(BaseLB::LDStats* stats, const LDObjKey& key,
                    const std::vector<int>& objBin)
{
  const int idx = stats->getHash(key);
  if (idx < 0 || (size_t)idx >= objBin.size()) return -1;
  return objBin[idx];
}

/** Charge each bin for the messages it would send and receive at this width.
 *
 * Only edges that cross bins are charged: traffic between two objects that
 * land together stays free, which is the whole reason placement has to be
 * modelled rather than the window's total message count scaled by width.
 */
void addCommOverhead(BaseLB::LDStats* stats, const std::vector<int>& objBin, int width,
                     std::vector<double>& binLoad)
{
  const double alpha = _lb_args.alpha();
  const double beta = _lb_args.beta();

  std::vector<double> sendCost(width, 0.0);
  std::vector<double> recvCost(width, 0.0);
  std::vector<int> hitBins;  // scratch, for multicast destination dedup

  for (size_t e = 0; e < stats->commData.size(); e++)
  {
    LDCommData& cdata = stats->commData[e];

    int senderBin;
    if (cdata.from_proc())
      senderBin = cdata.src_proc % width;
    else
    {
      senderBin = binOfObj(stats, cdata.sender, objBin);
      if (senderBin < 0) continue;  // sender is not in this window's data
    }

    const double sent = cdata.messages * alpha + cdata.bytes * beta;
    const double recvd =
        cdata.messages * PER_MESSAGE_RECV_OVERHEAD + cdata.bytes * PER_BYTE_RECV_OVERHEAD;

    const int receiverType = cdata.receiver.get_type();
    if (receiverType == LD_PROC_MSG || receiverType == LD_OBJ_MSG)
    {
      const int receiverBin = (receiverType == LD_PROC_MSG)
                                  ? cdata.receiver.proc() % width
                                  : binOfObj(stats, cdata.receiver.get_destObj(), objBin);
      if (receiverBin < 0) continue;
      if (receiverBin != senderBin)
      {
        sendCost[senderBin] += sent;
        recvCost[receiverBin] += recvd;
      }
    }
    else if (receiverType == LD_OBJLIST_MSG)
    {
      // One send per distinct destination bin: a multicast to several objects
      // that land together is one message on the wire, so the cost of a
      // multicast falls as its destinations are packed closer, which is
      // exactly the effect worth modelling.
      int nobjs = 0;
      const LDObjKey* objs = cdata.receiver.get_destObjs(nobjs);
      hitBins.clear();
      for (int i = 0; i < nobjs; i++)
      {
        const int receiverBin = binOfObj(stats, objs[i], objBin);
        if (receiverBin < 0 || receiverBin == senderBin) continue;
        if (std::find(hitBins.begin(), hitBins.end(), receiverBin) != hitBins.end()) continue;
        hitBins.push_back(receiverBin);
        sendCost[senderBin] += sent;
        recvCost[receiverBin] += recvd;
      }
    }
  }

  for (int b = 0; b < width; b++) binLoad[b] += sendCost[b] + recvCost[b];
}

}  // namespace

unsigned int CkPredictWidths(BaseLB::LDStats* stats, const int* widths, int nwidths,
                             CkWidthPrediction* out)
{
  if (stats == NULL || widths == NULL || out == NULL || nwidths <= 0) return 0;

  const size_t nobjs = stats->objData.size();
  if (nobjs == 0) return 0;

  // Sorted once and reused: the packing order is the same at every width.
  std::vector<LoadBin> descending;
  descending.reserve(nobjs);
  for (size_t i = 0; i < nobjs; i++)
    descending.push_back(LoadBin(stats->objData[i].wallTime, (int)i));
  std::sort(descending.begin(), descending.end(), std::greater<LoadBin>());

  // Per-processor overhead -- scheduler, runtime, whatever the balancer calls
  // background -- is charged to every bin at every width. It is the term that
  // stops speedup at some point even when the objects divide perfectly.
  double totalBg = 0.0;
  for (size_t i = 0; i < stats->nprocs(); i++) totalBg += stats->procs[i].bg_walltime;
  const double bgPerPe = (stats->nprocs() > 0) ? (totalBg / (double)stats->nprocs()) : 0.0;

  const bool haveComm = !stats->commData.empty();
  if (haveComm) stats->makeCommHash();

  std::vector<double> binLoad;
  std::vector<int> objBin(nobjs, 0);

  for (int w = 0; w < nwidths; w++)
  {
    const int width = (widths[w] < 1) ? 1 : widths[w];
    packLPT(descending, width, bgPerPe, binLoad, objBin);
    if (haveComm) addCommOverhead(stats, objBin, width, binLoad);

    double busiest = 0.0;
    for (int b = 0; b < width; b++)
      if (binLoad[b] > busiest) busiest = binLoad[b];

    out[w].numPes = width;
    out[w].predWall = busiest;
  }

  return CK_WIDTHMODEL_PACKING | (haveComm ? CK_WIDTHMODEL_COMM : 0u);
}

int CkChooseWidths(int lo, int hi, int current, int numObjs, int* out, int maxOut)
{
  if (out == NULL || maxOut <= 0) return 0;

  if (lo < 1) lo = 1;
  if (numObjs > 0 && hi > numObjs) hi = numObjs;
  if (hi < lo) hi = lo;

  std::vector<int> widths;
  widths.push_back(lo);
  widths.push_back(hi);
  // The current width is always predicted, even when it falls outside the
  // range the scheduler asked about. Comparing its prediction against the
  // window actually measured is the only check available on whether any of
  // the other points mean anything.
  if (current >= 1) widths.push_back(current);

  const int interior = maxOut - (int)widths.size();
  if (interior > 0 && hi > lo)
  {
    const double step = std::log((double)hi / (double)lo) / (double)(interior + 1);
    for (int i = 1; i <= interior; i++)
    {
      int p = (int)(lo * std::exp(step * i) + 0.5);
      if (p < lo) p = lo;
      if (p > hi) p = hi;
      widths.push_back(p);
    }
  }

  std::sort(widths.begin(), widths.end());
  widths.erase(std::unique(widths.begin(), widths.end()), widths.end());

  int n = 0;
  for (size_t i = 0; i < widths.size() && n < maxOut; i++) out[n++] = widths[i];
  return n;
}

#else

unsigned int CkPredictWidths(BaseLB::LDStats*, const int*, int, CkWidthPrediction*)
{
  return 0;
}
int CkChooseWidths(int, int, int, int, int*, int) { return 0; }

#endif

/*@}*/
