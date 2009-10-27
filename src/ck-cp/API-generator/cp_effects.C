#include "cp_effects.h"

using namespace ControlPoint;
using namespace std;

enum EFFECT {EFF_DEC, EFF_INC};

typedef map<std::string, map<std::string, vector<pair<int, ControlPoint::ControlPointAssociation> > > > cp_effect_map;
typedef map<std::string, vector<pair<int, ControlPoint::ControlPointAssociation> > > cp_name_map;

CkpvDeclare(cp_effect_map, cp_effects);
CkpvDeclare(cp_name_map, cp_names);

void initControlPointEffects() {
	CkpvInitialize(cp_effect_map, cp_effects);
	CkpvInitialize(cp_name_map, cp_names);
}

void testControlPointEffects() {

	ControlPoint::EffectIncrease::Priority("name");
	ControlPoint::EffectDecrease::Priority("name");
	ControlPoint::EffectIncrease::Priority("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::Priority("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::Priority("name", EntryAssociation);
	ControlPoint::EffectDecrease::Priority("name", EntryAssociation);
	ControlPoint::EffectIncrease::Priority("name", ArrayAssociation);
	ControlPoint::EffectDecrease::Priority("name", ArrayAssociation);
	ControlPoint::EffectIncrease::MemoryConsumption("name");
	ControlPoint::EffectDecrease::MemoryConsumption("name");
	ControlPoint::EffectIncrease::MemoryConsumption("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::MemoryConsumption("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::MemoryConsumption("name", EntryAssociation);
	ControlPoint::EffectDecrease::MemoryConsumption("name", EntryAssociation);
	ControlPoint::EffectIncrease::MemoryConsumption("name", ArrayAssociation);
	ControlPoint::EffectDecrease::MemoryConsumption("name", ArrayAssociation);
	ControlPoint::EffectIncrease::Granularity("name");
	ControlPoint::EffectDecrease::Granularity("name");
	ControlPoint::EffectIncrease::Granularity("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::Granularity("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::Granularity("name", EntryAssociation);
	ControlPoint::EffectDecrease::Granularity("name", EntryAssociation);
	ControlPoint::EffectIncrease::Granularity("name", ArrayAssociation);
	ControlPoint::EffectDecrease::Granularity("name", ArrayAssociation);
	ControlPoint::EffectIncrease::ComputeDurations("name");
	ControlPoint::EffectDecrease::ComputeDurations("name");
	ControlPoint::EffectIncrease::ComputeDurations("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::ComputeDurations("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::ComputeDurations("name", EntryAssociation);
	ControlPoint::EffectDecrease::ComputeDurations("name", EntryAssociation);
	ControlPoint::EffectIncrease::ComputeDurations("name", ArrayAssociation);
	ControlPoint::EffectDecrease::ComputeDurations("name", ArrayAssociation);
	ControlPoint::EffectIncrease::FlopRate("name");
	ControlPoint::EffectDecrease::FlopRate("name");
	ControlPoint::EffectIncrease::FlopRate("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::FlopRate("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::FlopRate("name", EntryAssociation);
	ControlPoint::EffectDecrease::FlopRate("name", EntryAssociation);
	ControlPoint::EffectIncrease::FlopRate("name", ArrayAssociation);
	ControlPoint::EffectDecrease::FlopRate("name", ArrayAssociation);
	ControlPoint::EffectIncrease::NumComputeObjects("name");
	ControlPoint::EffectDecrease::NumComputeObjects("name");
	ControlPoint::EffectIncrease::NumComputeObjects("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::NumComputeObjects("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::NumComputeObjects("name", EntryAssociation);
	ControlPoint::EffectDecrease::NumComputeObjects("name", EntryAssociation);
	ControlPoint::EffectIncrease::NumComputeObjects("name", ArrayAssociation);
	ControlPoint::EffectDecrease::NumComputeObjects("name", ArrayAssociation);
	ControlPoint::EffectIncrease::NumMessages("name");
	ControlPoint::EffectDecrease::NumMessages("name");
	ControlPoint::EffectIncrease::NumMessages("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::NumMessages("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::NumMessages("name", EntryAssociation);
	ControlPoint::EffectDecrease::NumMessages("name", EntryAssociation);
	ControlPoint::EffectIncrease::NumMessages("name", ArrayAssociation);
	ControlPoint::EffectDecrease::NumMessages("name", ArrayAssociation);
	ControlPoint::EffectIncrease::MessageSizes("name");
	ControlPoint::EffectDecrease::MessageSizes("name");
	ControlPoint::EffectIncrease::MessageSizes("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::MessageSizes("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::MessageSizes("name", EntryAssociation);
	ControlPoint::EffectDecrease::MessageSizes("name", EntryAssociation);
	ControlPoint::EffectIncrease::MessageSizes("name", ArrayAssociation);
	ControlPoint::EffectDecrease::MessageSizes("name", ArrayAssociation);
	ControlPoint::EffectIncrease::MessageOverhead("name");
	ControlPoint::EffectDecrease::MessageOverhead("name");
	ControlPoint::EffectIncrease::MessageOverhead("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::MessageOverhead("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::MessageOverhead("name", EntryAssociation);
	ControlPoint::EffectDecrease::MessageOverhead("name", EntryAssociation);
	ControlPoint::EffectIncrease::MessageOverhead("name", ArrayAssociation);
	ControlPoint::EffectDecrease::MessageOverhead("name", ArrayAssociation);
	ControlPoint::EffectIncrease::UnnecessarySyncronization("name");
	ControlPoint::EffectDecrease::UnnecessarySyncronization("name");
	ControlPoint::EffectIncrease::UnnecessarySyncronization("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::UnnecessarySyncronization("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::UnnecessarySyncronization("name", EntryAssociation);
	ControlPoint::EffectDecrease::UnnecessarySyncronization("name", EntryAssociation);
	ControlPoint::EffectIncrease::UnnecessarySyncronization("name", ArrayAssociation);
	ControlPoint::EffectDecrease::UnnecessarySyncronization("name", ArrayAssociation);
	ControlPoint::EffectIncrease::Concurrency("name");
	ControlPoint::EffectDecrease::Concurrency("name");
	ControlPoint::EffectIncrease::Concurrency("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::Concurrency("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::Concurrency("name", EntryAssociation);
	ControlPoint::EffectDecrease::Concurrency("name", EntryAssociation);
	ControlPoint::EffectIncrease::Concurrency("name", ArrayAssociation);
	ControlPoint::EffectDecrease::Concurrency("name", ArrayAssociation);
	ControlPoint::EffectIncrease::PotentialOverlap("name");
	ControlPoint::EffectDecrease::PotentialOverlap("name");
	ControlPoint::EffectIncrease::PotentialOverlap("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::PotentialOverlap("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::PotentialOverlap("name", EntryAssociation);
	ControlPoint::EffectDecrease::PotentialOverlap("name", EntryAssociation);
	ControlPoint::EffectIncrease::PotentialOverlap("name", ArrayAssociation);
	ControlPoint::EffectDecrease::PotentialOverlap("name", ArrayAssociation);
	ControlPoint::EffectIncrease::LoadBalancingPeriod("name");
	ControlPoint::EffectDecrease::LoadBalancingPeriod("name");
	ControlPoint::EffectIncrease::LoadBalancingPeriod("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::LoadBalancingPeriod("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::LoadBalancingPeriod("name", EntryAssociation);
	ControlPoint::EffectDecrease::LoadBalancingPeriod("name", EntryAssociation);
	ControlPoint::EffectIncrease::LoadBalancingPeriod("name", ArrayAssociation);
	ControlPoint::EffectDecrease::LoadBalancingPeriod("name", ArrayAssociation);
	ControlPoint::EffectIncrease::GPUOffloadedWork("name");
	ControlPoint::EffectDecrease::GPUOffloadedWork("name");
	ControlPoint::EffectIncrease::GPUOffloadedWork("name", NoControlPointAssociation);
	ControlPoint::EffectDecrease::GPUOffloadedWork("name", NoControlPointAssociation);
	ControlPoint::EffectIncrease::GPUOffloadedWork("name", EntryAssociation);
	ControlPoint::EffectDecrease::GPUOffloadedWork("name", EntryAssociation);
	ControlPoint::EffectIncrease::GPUOffloadedWork("name", ArrayAssociation);
	ControlPoint::EffectDecrease::GPUOffloadedWork("name", ArrayAssociation);
}

void insert(const std::string control_type, const std::string name, const ControlPoint::ControlPointAssociation &a, const int effect) {
	CkpvAccess(cp_effects)[control_type][name].push_back(std::make_pair(effect, a));
	CkpvAccess(cp_names)[name].push_back(std::make_pair(effect, a));
}
void ControlPoint::EffectDecrease::Priority(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("Priority", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::Priority(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("Priority", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::MemoryConsumption(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("MemoryConsumption", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::MemoryConsumption(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("MemoryConsumption", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::Granularity(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("Granularity", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::Granularity(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("Granularity", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::ComputeDurations(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("ComputeDurations", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::ComputeDurations(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("ComputeDurations", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::FlopRate(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("FlopRate", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::FlopRate(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("FlopRate", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::NumComputeObjects(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("NumComputeObjects", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::NumComputeObjects(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("NumComputeObjects", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::NumMessages(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("NumMessages", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::NumMessages(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("NumMessages", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::MessageSizes(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("MessageSizes", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::MessageSizes(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("MessageSizes", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::MessageOverhead(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("MessageOverhead", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::MessageOverhead(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("MessageOverhead", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::UnnecessarySyncronization(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("UnnecessarySyncronization", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::UnnecessarySyncronization(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("UnnecessarySyncronization", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::Concurrency(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("Concurrency", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::Concurrency(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("Concurrency", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::PotentialOverlap(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("PotentialOverlap", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::PotentialOverlap(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("PotentialOverlap", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::LoadBalancingPeriod(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("LoadBalancingPeriod", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::LoadBalancingPeriod(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("LoadBalancingPeriod", s, a, EFF_INC);
}
void ControlPoint::EffectDecrease::GPUOffloadedWork(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("GPUOffloadedWork", s, a, EFF_DEC);
}
void ControlPoint::EffectIncrease::GPUOffloadedWork(std::string s, const ControlPoint::ControlPointAssociation &a) {
	insert("GPUOffloadedWork", s, a, EFF_INC);
}
