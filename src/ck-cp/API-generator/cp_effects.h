#include <string>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include "charm++.h"
#include "ck.h"
#include "ckarray.h"

namespace ControlPoint {
  class ControlPointAssociation {
  public:
    std::set<int> EntryID;
    std::set<int> ArrayGroupIdx;
      ControlPointAssociation() {
	// nothing here yet
      }
  };
  
  class ControlPointAssociatedEntry : public ControlPointAssociation {
    public :
	ControlPointAssociatedEntry() : ControlPointAssociation() {}

	ControlPointAssociatedEntry(int epid) : ControlPointAssociation() {
	  EntryID.insert(epid);
	}    
  };
  
  class ControlPointAssociatedArray : public ControlPointAssociation {
  public:
    ControlPointAssociatedArray() : ControlPointAssociation() {}

    ControlPointAssociatedArray(CProxy_ArrayBase &a) : ControlPointAssociation() {
      CkGroupID aid = a.ckGetArrayID();
      int groupIdx = aid.idx;
      ArrayGroupIdx.insert(groupIdx);
    }    
  };
  
  ControlPointAssociation NoControlPointAssociation;
  int epid = 2;
  ControlPointAssociatedEntry EntryAssociation(epid);
  ControlPointAssociatedArray ArrayAssociation;

namespace EffectIncrease {
	void Priority(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void MemoryConsumption(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void Granularity(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void ComputeDurations(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void FlopRate(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void NumComputeObjects(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void NumMessages(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void MessageSizes(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void MessageOverhead(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void UnnecessarySyncronization(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void Concurrency(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void PotentialOverlap(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void LoadBalancingPeriod(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void GPUOffloadedWork(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
}

namespace EffectDecrease {
	void Priority(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void MemoryConsumption(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void Granularity(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void ComputeDurations(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void FlopRate(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void NumComputeObjects(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void NumMessages(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void MessageSizes(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void MessageOverhead(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void UnnecessarySyncronization(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void Concurrency(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void PotentialOverlap(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void LoadBalancingPeriod(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
	void GPUOffloadedWork(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);
}
} //namespace ControlPoint 
