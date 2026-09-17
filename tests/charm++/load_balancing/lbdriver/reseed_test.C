// Runtime-free tests of the production selector, safe on login nodes.
#include "DiffusionMetric.h"
#include <cassert>
#include <cstdlib>
#include <memory>
void lbsimParseArgs(int&, char**);
int CmiNumPes() { return 4; }
struct Fixture {
  BaseLB::LDStats st;
  DiffusionCostConfig cfg;
  std::unique_ptr<MetricComm> metric;
  double internal=0, external=0;
  Fixture(std::initializer_list<double> loads) {
    st.objData.resize(loads.size());
    int i=0;
    for (double load:loads) {
      auto& o=st.objData[i];
      o.handle.omhandle.id.id.idx=1; o.handle.id=i;
      o.migratable=true; o.wallTime=.01; o.gpuTime=load;
      o.gpuPupSize=0; o.pupSize=0; ++i;
    }
    st.makeCommHash();
    cfg.calibrated=true; cfg.migrateAlpha=.4; cfg.placementLifetimeIntervals=4;
  }
  void edge(int from,int to,int pe=1) {
    LDCommData c;
    c.src_proc=-1; c.sender.omId=st.objData[from].omID(); c.sender.objId=from;
    CmiUInt8 dest=to;
    c.receiver.init_objmsg(c.sender.omId,dest,pe);
    c.messages=1; c.bytes=100; c.sendHash=c.recvHash=-1;
    st.commData.push_back(c);
  }
  void start(double quota=20) {
    metric.reset(new MetricComm(&st,0,1,1,{quota},{1},internal,external,&cfg));
    metric->setRemainingShed(20);
  }
  int take() {int i=metric->popBestObject(0);if(i>=0)metric->updateState(i,0);return i;}
};
int main(int argc,char** argv) {
  lbsimParseArgs(argc,argv); diffusionLoadDimDevice=LB_MODE_DEVICE;
  unsetenv("CHARM_DIFFUSION_GROW_ANY");
  { // Independent remote-partner computes reseed by score, not by shared endpoint.
    Fixture f{3,2,1}; f.edge(0,100);f.edge(1,101);f.edge(2,100);f.start();
    assert(f.take()==0);assert(f.take()==1);assert(f.take()==2);assert(f.take()==-1);
  }
  { // No comm graph at all: independent objects still reseed by score.
    Fixture f{3,2};f.start();assert(f.take()==0);assert(f.take()==1);assert(f.take()==-1);
  }
  { // A reseed need not already talk to the receiver; the score prices that.
    Fixture f{3,2};f.edge(0,100);f.edge(1,101,2);f.start();
    assert(f.take()==0);assert(f.take()==1);assert(f.take()==-1);
  }
  { // A reseed that does not pay is refused like any other move.
    Fixture f{3,.05};f.edge(0,100);f.edge(1,100);f.start();
    assert(f.take()==0);assert(f.take()==-1);assert(f.metric->rejectedCount()==1);
  }
  { // Fixed local frontier permits an attached new seed.
    Fixture f{3,2,0};f.st.objData[2].migratable=false;
    f.edge(0,2,0);f.edge(1,2,0);f.edge(0,100);f.edge(1,100);f.start();
    assert(f.take()==0);assert(f.take()==1);
  }
  { // Exhausting a connected component cannot seed another stencil component.
    Fixture f{3,2,1};f.edge(0,1,0);f.edge(0,100);f.edge(2,100);f.start();
    assert(f.take()==0);assert(f.take()==1);assert(f.take()==-1);
  }
  { // A quota-rejected live frontier cannot be bypassed.
    Fixture f{3,2,.5};f.edge(0,1,0);f.edge(0,100);f.edge(2,100);f.start(4);
    assert(f.take()==0);assert(f.take()==-1);
  }
  { // Neither can a host-capacity-rejected frontier.
    Fixture f{3,2,1};f.st.objData[1].wallTime=100;
    f.edge(0,1,0);f.edge(0,100);f.edge(2,100);f.start();
    f.metric->setReceiverCapacity({1},{100});
    assert(f.take()==0);assert(f.take()==-1);
  }
  { // Nor an allowed-mask rejection.
    Fixture f{3,2,1};f.edge(0,1,0);f.edge(0,100);f.edge(2,100);f.start();
    std::vector<char> allowed{1,0,1};f.metric->setAllowed(&allowed);
    assert(f.take()==0);assert(f.take()==-1);
  }
  { // Nor a cost-rejected live frontier.
    Fixture f{3,.05,1};f.edge(0,1,0);f.edge(0,100);f.edge(2,100);f.start();
    assert(f.take()==0);assert(f.take()==-1);assert(f.metric->rejectedCount()==1);
  }
  { // Reseeding itself still enforces quota and capacity.
    Fixture f{3,2};f.edge(0,100);f.edge(1,100);f.start(4);
    assert(f.take()==0);assert(f.take()==-1);
  }
  { // A capacity-refused candidate must not hide a valid reseed.
    Fixture f{3,1,2};f.st.objData[1].wallTime=100;
    f.edge(0,100);f.edge(1,100);f.edge(2,101);f.start();
    f.metric->setReceiverCapacity({1},{100});
    assert(f.take()==0);assert(f.take()==2);
  }
  { // Memory rejection on a live frontier cannot be bypassed either.
    Fixture f{3,2,1};f.edge(0,1,0);f.edge(0,100);f.edge(2,100);f.start();
    f.metric->setMemoryCapacity({2},100,0,{1,3,1},{1,1,1},f.st.objData);
    assert(f.take()==0);assert(f.take()==-1);assert(f.metric->memRefusals>0);
  }
  { // The explicit broad ablation remains available.
    setenv("CHARM_DIFFUSION_GROW_ANY","1",1);
    Fixture f{3,2};f.start();assert(f.take()==0);assert(f.take()==1);
    unsetenv("CHARM_DIFFUSION_GROW_ANY");
  }
  CkPrintf("reseed_test: 14 scenarios passed\n");
}
