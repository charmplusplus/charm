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

    ControlPointAssociatedArray(const CProxy_ArrayBase &a) : ControlPointAssociation() {
      CkGroupID aid = a.ckGetArrayID();
      int groupIdx = aid.idx;
      ArrayGroupIdx.insert(groupIdx);
    }
  };
  
  class NoControlPointAssociation : public ControlPointAssociation { };
	void initControlPointEffects();
	ControlPointAssociatedEntry assocWithEntry(const int entry);
	ControlPointAssociatedArray assocWithArray(const CProxy_ArrayBase &array);
namespace EffectIncrease {
	void Priority(std::string name, const ControlPoint::ControlPointAssociation &a);
	void Priority(std::string name);
	void MemoryConsumption(std::string name, const ControlPoint::ControlPointAssociation &a);
	void MemoryConsumption(std::string name);
	void Granularity(std::string name, const ControlPoint::ControlPointAssociation &a);
	void Granularity(std::string name);
	void ComputeDurations(std::string name, const ControlPoint::ControlPointAssociation &a);
	void ComputeDurations(std::string name);
	void FlopRate(std::string name, const ControlPoint::ControlPointAssociation &a);
	void FlopRate(std::string name);
	void NumComputeObjects(std::string name, const ControlPoint::ControlPointAssociation &a);
	void NumComputeObjects(std::string name);
	void NumMessages(std::string name, const ControlPoint::ControlPointAssociation &a);
	void NumMessages(std::string name);
	void MessageSizes(std::string name, const ControlPoint::ControlPointAssociation &a);
	void MessageSizes(std::string name);
	void MessageOverhead(std::string name, const ControlPoint::ControlPointAssociation &a);
	void MessageOverhead(std::string name);
	void UnnecessarySyncronization(std::string name, const ControlPoint::ControlPointAssociation &a);
	void UnnecessarySyncronization(std::string name);
	void Concurrency(std::string name, const ControlPoint::ControlPointAssociation &a);
	void Concurrency(std::string name);
	void PotentialOverlap(std::string name, const ControlPoint::ControlPointAssociation &a);
	void PotentialOverlap(std::string name);
	void LoadBalancingPeriod(std::string name, const ControlPoint::ControlPointAssociation &a);
	void LoadBalancingPeriod(std::string name);
	void GPUOffloadedWork(std::string name, const ControlPoint::ControlPointAssociation &a);
	void GPUOffloadedWork(std::string name);
}

namespace EffectDecrease {
	void Priority(std::string name, const ControlPoint::ControlPointAssociation &a);
	void Priority(std::string name);
	void MemoryConsumption(std::string name, const ControlPoint::ControlPointAssociation &a);
	void MemoryConsumption(std::string name);
	void Granularity(std::string name, const ControlPoint::ControlPointAssociation &a);
	void Granularity(std::string name);
	void ComputeDurations(std::string name, const ControlPoint::ControlPointAssociation &a);
	void ComputeDurations(std::string name);
	void FlopRate(std::string name, const ControlPoint::ControlPointAssociation &a);
	void FlopRate(std::string name);
	void NumComputeObjects(std::string name, const ControlPoint::ControlPointAssociation &a);
	void NumComputeObjects(std::string name);
	void NumMessages(std::string name, const ControlPoint::ControlPointAssociation &a);
	void NumMessages(std::string name);
	void MessageSizes(std::string name, const ControlPoint::ControlPointAssociation &a);
	void MessageSizes(std::string name);
	void MessageOverhead(std::string name, const ControlPoint::ControlPointAssociation &a);
	void MessageOverhead(std::string name);
	void UnnecessarySyncronization(std::string name, const ControlPoint::ControlPointAssociation &a);
	void UnnecessarySyncronization(std::string name);
	void Concurrency(std::string name, const ControlPoint::ControlPointAssociation &a);
	void Concurrency(std::string name);
	void PotentialOverlap(std::string name, const ControlPoint::ControlPointAssociation &a);
	void PotentialOverlap(std::string name);
	void LoadBalancingPeriod(std::string name, const ControlPoint::ControlPointAssociation &a);
	void LoadBalancingPeriod(std::string name);
	void GPUOffloadedWork(std::string name, const ControlPoint::ControlPointAssociation &a);
	void GPUOffloadedWork(std::string name);
}
} //namespace ControlPoint 
