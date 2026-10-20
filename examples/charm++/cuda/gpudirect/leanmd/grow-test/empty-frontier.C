// Standalone decision-code reproduction: three movable objects, no local edges.
// This links the runtime stubs, never starts Charm/LCI/CUDA.
#include "DiffusionSelect.h"
#include <cassert>
#include <cstdlib>
void lbsimParseArgs(int&, char**);
int CmiNumPes() { return 2; }
int main(int argc, char** argv) {
  lbsimParseArgs(argc, argv);
  diffusionLoadDimDevice = LB_MODE_DEVICE;
  for (int grow = 0; grow < 2; ++grow) {
    if (grow) setenv("CHARM_DIFFUSION_GROW_ANY", "1", 1);
    else unsetenv("CHARM_DIFFUSION_GROW_ANY");
    BaseLB::LDStats st;
    st.objData.resize(3);
    for (auto& o : st.objData) {
      o.migratable = true; o.wallTime = 0.01; o.gpuTime = 1.0;
      o.gpuPupSize = 0; o.pupSize = 0;
    }
    DiffusionCostConfig cfg;
    cfg.calibrated = true; cfg.migrateAlpha = 0.4;
    cfg.placementLifetimeIntervals = 4.0;
    double internal = 0, external = 0;
    std::vector<double> quota{3.0};
    MetricComm metric(&st, 0, 1, 1, quota, {1}, internal, external, &cfg);
    double remaining = 3.0, shed = 0.0; int count = 0;
    std::vector<DiffusionMove> moves;
    diffusionSelectMoves(metric, st, quota, remaining, 3.0, {}, 0, moves, shed, count);
    CkPrintf("grow=%d accepted=%d priced_out=%d unshed=%.1f\n", grow, count, metric.rejectedCount(), remaining);
    assert(count == (grow ? 3 : 1));
    assert(metric.rejectedCount() == 0);
  }
}
