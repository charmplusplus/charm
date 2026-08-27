#ifndef CK_LDB_GPU_SCALING_MODEL_H
#define CK_LDB_GPU_SCALING_MODEL_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "pup.h"
#include "pup_stl.h"

// CUDA-independent key and estimator types for learning relative GPU service
// demand. Keeping this component independent of CUPTI lets its numerical and
// PUP behavior be tested in a CPU-only Charm++ build.
struct GpuKernelKey
{
  uint64_t kernelClass = 0;
  uint64_t workBucket = 0;

  GpuKernelKey() = default;
  GpuKernelKey(uint64_t kernelClass_, uint64_t workBucket_)
      : kernelClass(kernelClass_), workBucket(workBucket_)
  {
  }

  bool operator==(const GpuKernelKey& other) const
  {
    return kernelClass == other.kernelClass && workBucket == other.workBucket;
  }

  bool operator<(const GpuKernelKey& other) const
  {
    if (kernelClass != other.kernelClass)
      return kernelClass < other.kernelClass;
    return workBucket < other.workBucket;
  }

  void pup(PUP::er& p)
  {
    p | kernelClass;
    p | workBucket;
  }
};

// CUDA-independent launch metadata. CUPTI fills this structure when scaling
// is enabled, but keeping it here makes classification deterministic and
// directly unit-testable without a GPU.
struct GpuLaunchSignature
{
  uint32_t gridX = 0;
  uint32_t gridY = 0;
  uint32_t gridZ = 0;
  uint32_t blockX = 0;
  uint32_t blockY = 0;
  uint32_t blockZ = 0;
  uint64_t staticSharedMemory = 0;
  uint64_t dynamicSharedMemory = 0;

  void pup(PUP::er& p)
  {
    p | gridX;
    p | gridY;
    p | gridZ;
    p | blockX;
    p | blockY;
    p | blockZ;
    p | staticSharedMemory;
    p | dynamicSharedMemory;
  }
};

namespace gpu_scaling_detail
{

inline void fnvAppendByte(uint64_t& hash, uint8_t value)
{
  hash ^= value;
  hash *= UINT64_C(1099511628211);
}

inline void fnvAppendUint64(uint64_t& hash, uint64_t value)
{
  // Specify byte order explicitly so an identity is stable across host
  // architectures.
  for (unsigned int byte = 0; byte < 8; ++byte)
    fnvAppendByte(hash, static_cast<uint8_t>(value >> (byte * 8)));
}

inline bool validPositive(double value) { return std::isfinite(value) && value > 0.0; }

inline uint64_t mixHash64(uint64_t value)
{
  // SplitMix64's finalizer gives stable, well-distributed hashes for the
  // already-interned integer identities used by the model.
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  return value;
}

inline uint64_t combineHash64(uint64_t seed, uint64_t value)
{
  return seed ^
         (mixHash64(value) + UINT64_C(0x9e3779b97f4a7c15) + (seed << 6) + (seed >> 2));
}

}  // namespace gpu_scaling_detail

// Stable FNV-1a based identifiers used at the CUPTI boundary. These are
// inline because HAPI and the LB model live in separate static libraries.
inline uint64_t gpuStableKernelClass(const char* kernelName)
{
  uint64_t hash = UINT64_C(14695981039346656037);
  if (kernelName == nullptr)
    return hash;

  for (const unsigned char* character =
           reinterpret_cast<const unsigned char*>(kernelName);
       *character != '\0'; ++character)
    gpu_scaling_detail::fnvAppendByte(hash, *character);
  return hash;
}

inline uint64_t gpuStableWorkBucket(const GpuLaunchSignature& launch,
                                    bool hasExplicitTag, uint64_t explicitTag)
{
  uint64_t hash = UINT64_C(14695981039346656037);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.gridX);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.gridY);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.gridZ);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.blockX);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.blockY);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.blockZ);
  gpu_scaling_detail::fnvAppendUint64(hash, launch.dynamicSharedMemory);

  // The discriminator prevents explicitly tagged and untagged launches from
  // sharing an identity even when the caller supplies tag zero.
  gpu_scaling_detail::fnvAppendByte(hash, hasExplicitTag ? UINT8_C(1) : UINT8_C(0));
  if (hasExplicitTag)
    gpu_scaling_detail::fnvAppendUint64(hash, explicitTag);
  return hash;
}

struct GpuKernelTypeKey
{
  GpuKernelKey kernel;
  uint64_t gpuType = 0;

  GpuKernelTypeKey() = default;
  GpuKernelTypeKey(const GpuKernelKey& kernel_, uint64_t gpuType_)
      : kernel(kernel_), gpuType(gpuType_)
  {
  }

  bool operator==(const GpuKernelTypeKey& other) const
  {
    return kernel == other.kernel && gpuType == other.gpuType;
  }

  bool operator<(const GpuKernelTypeKey& other) const
  {
    if (kernel < other.kernel)
      return true;
    if (other.kernel < kernel)
      return false;
    return gpuType < other.gpuType;
  }

  void pup(PUP::er& p)
  {
    p | kernel;
    p | gpuType;
  }
};

// Named hash functors keep the key usable in either ordered or hash-based
// containers without relying on implementation-defined std::hash combining.
// Inline for the same reason as the identity helpers above: HAPI and the LB
// model live in separate static libraries, and libhybridapi is linked after
// libck, so anything HAPI touches cannot be resolved out of libck's archive.
struct GpuKernelKeyHash
{
  std::size_t operator()(const GpuKernelKey& key) const
  {
    uint64_t hash = gpu_scaling_detail::mixHash64(key.kernelClass);
    hash = gpu_scaling_detail::combineHash64(hash, key.workBucket);
    return static_cast<std::size_t>(hash);
  }
};

struct GpuKernelTypeKeyHash
{
  std::size_t operator()(const GpuKernelTypeKey& key) const
  {
    uint64_t hash = gpu_scaling_detail::mixHash64(key.kernel.kernelClass);
    hash = gpu_scaling_detail::combineHash64(hash, key.kernel.workBucket);
    hash = gpu_scaling_detail::combineHash64(hash, key.gpuType);
    return static_cast<std::size_t>(hash);
  }
};

struct GpuDeviceTypeInfo
{
  uint64_t typeId = 0;
  double peakRateScore = 1.0;

  GpuDeviceTypeInfo() = default;
  GpuDeviceTypeInfo(uint64_t typeId_, double peakRateScore_)
      : typeId(typeId_), peakRateScore(peakRateScore_)
  {
  }

  void pup(PUP::er& p)
  {
    p | typeId;
    p | peakRateScore;
  }
};

enum class GpuPeakRateSource : uint8_t
{
  Unknown = 0,
  UserOverride = 1,
  ArchitectureTable = 2,
  SmClockProxy = 3,
};

// Hardware identity discovered once per HAPI DeviceManager. instanceId names
// one physical device in the current job, while typeId is the same for
// equivalent devices on different hosts.
struct GpuDeviceDescriptor
{
  uint64_t instanceId = 0;
  uint64_t typeId = 0;
  double peakRateScore = 0.0;
  uint32_t smCount = 0;
  uint32_t computeMajor = 0;
  uint32_t computeMinor = 0;
  uint32_t maxClockKHz = 0;
  uint64_t totalMemory = 0;
  GpuPeakRateSource peakRateSource = GpuPeakRateSource::Unknown;

  void pup(PUP::er& p)
  {
    p | instanceId;
    p | typeId;
    p | peakRateScore;
    p | smCount;
    p | computeMajor;
    p | computeMinor;
    p | maxClockKHz;
    p | totalMemory;
    uint8_t encodedSource = static_cast<uint8_t>(peakRateSource);
    p | encodedSource;
    if (p.isUnpacking())
      peakRateSource = static_cast<GpuPeakRateSource>(encodedSource);
  }
};

inline uint64_t gpuStableDeviceType(const char* productName, uint32_t smCount,
                                    uint32_t computeMajor, uint32_t computeMinor,
                                    uint32_t maxClockKHz, uint64_t totalMemory)
{
  uint64_t hash = gpuStableKernelClass(productName);
  gpu_scaling_detail::fnvAppendUint64(hash, smCount);
  gpu_scaling_detail::fnvAppendUint64(hash, computeMajor);
  gpu_scaling_detail::fnvAppendUint64(hash, computeMinor);
  gpu_scaling_detail::fnvAppendUint64(hash, maxClockKHz);
  gpu_scaling_detail::fnvAppendUint64(hash, totalMemory);
  return hash;
}

// Single-precision lanes per SM, keyed by compute capability. SM count times
// clock alone is not a usable cross-architecture prior: lanes per SM changes
// between architectures, so an Ampere-to-Hopper comparison built from SMs and
// clock is wrong by roughly the ratio of their lane counts. Returns 0 for a
// capability this table does not know.
inline uint32_t gpuCoresPerSm(uint32_t computeMajor, uint32_t computeMinor)
{
  switch ((computeMajor << 4) | (computeMinor & 0xf))
  {
    case 0x30:
    case 0x32:
    case 0x35:
    case 0x37:
      return 192;  // Kepler
    case 0x50:
    case 0x52:
    case 0x53:
      return 128;  // Maxwell
    case 0x60:
      return 64;  // Pascal GP100
    case 0x61:
    case 0x62:
      return 128;  // Pascal GP10x
    case 0x70:
    case 0x72:
      return 64;  // Volta
    case 0x75:
      return 64;  // Turing
    case 0x80:
      return 64;  // Ampere GA100
    case 0x86:
    case 0x87:
    case 0x89:
      return 128;  // Ampere GA10x and Ada
    case 0x90:
      return 128;  // Hopper
    case 0xa0:
    case 0xa1:
      return 128;  // Blackwell datacenter
    case 0xc0:
    case 0xc1:
      return 128;  // Blackwell consumer
    default:
      return 0;
  }
}

// Substituted when a property needed by the rate prior is missing. Both keep
// the score in the same units as a fully known device: a fallback that changed
// the units (dropping the clock factor, or collapsing to 1.0) would make a
// degraded device incomparable with its neighbours by many orders of
// magnitude, which is far worse than being wrong by the spread of real clock
// rates or lane counts.
inline uint32_t gpuNominalClockKHz() { return 1500000; }
inline uint32_t gpuNominalCoresPerSm() { return 128; }

// Derives the cross-GPU rate prior from discovered hardware properties.
// Returns false when the device is too poorly described to rank at all, in
// which case score is a neutral 1.0 and source is Unknown.
inline bool gpuDerivePeakRateScore(uint32_t smCount, uint32_t computeMajor,
                                   uint32_t computeMinor, uint32_t maxClockKHz,
                                   double& peakRateScore,
                                   GpuPeakRateSource& peakRateSource)
{
  if (smCount == 0)
  {
    peakRateScore = 1.0;
    peakRateSource = GpuPeakRateSource::Unknown;
    return false;
  }

  const uint32_t tabulatedCores = gpuCoresPerSm(computeMajor, computeMinor);
  const bool coresKnown = tabulatedCores != 0;
  const uint32_t cores = coresKnown ? tabulatedCores : gpuNominalCoresPerSm();

  const bool clockKnown = maxClockKHz != 0;
  const uint32_t clockKHz = clockKnown ? maxClockKHz : gpuNominalClockKHz();

  // Fixed evaluation order: two reporters of the same device type must produce
  // bit-identical scores or GpuScalingModel::registerGpuType rejects the second
  // one as a conflicting definition.
  peakRateScore = static_cast<double>(smCount) * static_cast<double>(cores) *
                  static_cast<double>(clockKHz);

  if (coresKnown && clockKnown)
    peakRateSource = GpuPeakRateSource::ArchitectureTable;
  else if (clockKnown)
    peakRateSource = GpuPeakRateSource::SmClockProxy;
  else
    peakRateSource = GpuPeakRateSource::Unknown;
  return true;
}

inline const char* gpuPeakRateSourceName(GpuPeakRateSource source)
{
  switch (source)
  {
    case GpuPeakRateSource::UserOverride:
      return "user-override";
    case GpuPeakRateSource::ArchitectureTable:
      return "architecture-table";
    case GpuPeakRateSource::SmClockProxy:
      return "sm-clock-proxy";
    case GpuPeakRateSource::Unknown:
      break;
  }
  return "unknown";
}

// Mergeable ordinary moments. The values stored by the scaling model are log
// costs, but this utility intentionally has no logarithm-specific behavior.
struct GpuRunningMoments
{
  uint64_t samples = 0;
  double mean = 0.0;
  double M2 = 0.0;

  // Inline because the CUPTI aggregation path in libhybridapi builds these
  // directly; see the note on GpuKernelKeyHash.
  bool observe(double value)
  {
    if (!std::isfinite(value) || samples == std::numeric_limits<uint64_t>::max())
      return false;

    ++samples;
    const double delta = value - mean;
    mean += delta / static_cast<double>(samples);
    const double delta2 = value - mean;
    M2 += delta * delta2;
    if (M2 < 0.0 && M2 > -std::numeric_limits<double>::epsilon())
      M2 = 0.0;
    return true;
  }

  bool merge(const GpuRunningMoments& other)
  {
    if (other.samples == 0)
      return true;
    if (!std::isfinite(other.mean) || !std::isfinite(other.M2) || other.M2 < 0.0)
      return false;

    if (samples == 0)
    {
      *this = other;
      return true;
    }

    if (!std::isfinite(mean) || !std::isfinite(M2) || M2 < 0.0)
      return false;

    const uint64_t combined = samples + other.samples;
    if (combined < samples)
      return false;

    const double delta = other.mean - mean;
    const double leftWeight = static_cast<double>(samples);
    const double rightWeight = static_cast<double>(other.samples);
    const double totalWeight = static_cast<double>(combined);

    mean += delta * rightWeight / totalWeight;
    M2 += other.M2 + delta * delta * leftWeight * rightWeight / totalWeight;
    samples = combined;
    if (M2 < 0.0 && M2 > -std::numeric_limits<double>::epsilon())
      M2 = 0.0;
    return std::isfinite(mean) && std::isfinite(M2);
  }

  double populationVariance() const
  {
    if (samples == 0)
      return 0.0;
    return M2 / static_cast<double>(samples);
  }

  double sampleVariance() const
  {
    if (samples < 2)
      return 0.0;
    return M2 / static_cast<double>(samples - 1);
  }

  double standardError() const
  {
    if (samples < 2)
      return 0.0;
    return std::sqrt(sampleVariance() / static_cast<double>(samples));
  }

  void pup(PUP::er& p)
  {
    p | samples;
    p | mean;
    p | M2;
  }
};

// Lifetime moments are retained for confidence. estimateMean is the value used
// for prediction: with alphaMin == 0 it is the ordinary running mean; with a
// positive floor it becomes an EWMA after 1/n falls below that floor.
struct GpuAdaptiveLogStats
{
  GpuRunningMoments lifetime;
  double estimateMean = 0.0;
  uint64_t lastEpoch = 0;

  bool observeLogCost(double logCost, uint64_t epoch, double alphaMin);

  void pup(PUP::er& p)
  {
    p | lifetime;
    p | estimateMean;
    p | lastEpoch;
  }
};

struct GpuScalingEntry
{
  GpuAdaptiveLogStats normalizedDemand;
  GpuAdaptiveLogStats rawDuration;

  void pup(PUP::er& p)
  {
    p | normalizedDemand;
    p | rawDuration;
  }
};

// One epoch's worth of one object's invocations of one kernel identity. This is
// a wire type: it is what actually crosses to the central LB, so it holds
// aggregates rather than per-invocation records. Sizes are proportional to
// distinct kernel identities, not to launch count.
struct GpuKernelEpochCost
{
  GpuKernelKey key;
  uint64_t calls = 0;
  // Seconds of whole-device occupancy summed over `calls` invocations. This is
  // the quantity the load balancer adds up; the moments below describe the
  // per-invocation distribution behind it.
  double normalizedDemand = 0.0;
  GpuRunningMoments logNormalizedDemand;
  GpuRunningMoments logDuration;

  // Inline: built by the CUPTI aggregation path in libhybridapi.
  bool observe(double invocationDemand, double invocationDuration)
  {
    if (!gpu_scaling_detail::validPositive(invocationDemand) ||
        !gpu_scaling_detail::validPositive(invocationDuration))
      return false;
    if (calls == std::numeric_limits<uint64_t>::max())
      return false;

    // Advance copies first: if either series rejects the sample, the total and
    // the call count must not move either, or the summary stops reconciling
    // with the scalar load it is supposed to decompose.
    GpuRunningMoments demandMoments = logNormalizedDemand;
    GpuRunningMoments durationMoments = logDuration;
    if (!demandMoments.observe(std::log(invocationDemand)) ||
        !durationMoments.observe(std::log(invocationDuration)))
      return false;

    const double total = normalizedDemand + invocationDemand;
    if (!std::isfinite(total))
      return false;

    logNormalizedDemand = demandMoments;
    logDuration = durationMoments;
    normalizedDemand = total;
    ++calls;
    return true;
  }

  bool merge(const GpuKernelEpochCost& other)
  {
    if (!(key == other.key))
      return false;
    if (other.calls == 0)
      return true;

    const uint64_t combined = calls + other.calls;
    if (combined < calls)
      return false;
    const double total = normalizedDemand + other.normalizedDemand;
    if (!std::isfinite(total))
      return false;

    GpuRunningMoments demandMoments = logNormalizedDemand;
    GpuRunningMoments durationMoments = logDuration;
    if (!demandMoments.merge(other.logNormalizedDemand) ||
        !durationMoments.merge(other.logDuration))
      return false;

    logNormalizedDemand = demandMoments;
    logDuration = durationMoments;
    normalizedDemand = total;
    calls = combined;
    return true;
  }

  void pup(PUP::er& p)
  {
    p | key;
    p | calls;
    p | normalizedDemand;
    p | logNormalizedDemand;
    p | logDuration;
  }
};

// Everything the estimator needs about one object's GPU work in one epoch.
// `sourceTypeId` names the GPU the measurements were taken on, without which a
// destination prediction has no baseline to scale from.
struct GpuObjectEpochCosts
{
  uint64_t sourceInstanceId = 0;
  uint64_t sourceTypeId = 0;
  std::vector<GpuKernelEpochCost> components;
  // Demand that is real but not attributable to a modeled component: kernels
  // beyond the component cap, and kernels whose work tag was lost so their
  // identity cannot be trusted. Predicted with the hardware prior rather than
  // dropped, so the per-object total always reconciles with scalar gpuTime.
  double unmodeledGpuTime = 0.0;

  double modeledDemand() const
  {
    double total = 0.0;
    for (const GpuKernelEpochCost& component : components)
      total += component.normalizedDemand;
    return total;
  }

  double totalDemand() const { return modeledDemand() + unmodeledGpuTime; }
  bool empty() const { return components.empty() && unmodeledGpuTime == 0.0; }

  void clear()
  {
    sourceInstanceId = 0;
    sourceTypeId = 0;
    components.clear();
    unmodeledGpuTime = 0.0;
  }

  // Keeps the largest `maxComponents` by demand and folds the rest into
  // unmodeledGpuTime, so the object total is unchanged by capping. Ordering is
  // by demand then by key, so every replica that sees the same components in
  // any order keeps the same ones.
  void enforceComponentCap(std::size_t maxComponents)
  {
    if (maxComponents == 0)
    {
      unmodeledGpuTime += modeledDemand();
      components.clear();
      return;
    }
    if (components.size() <= maxComponents)
      return;

    std::sort(components.begin(), components.end(),
              [](const GpuKernelEpochCost& left, const GpuKernelEpochCost& right) {
                if (left.normalizedDemand != right.normalizedDemand)
                  return left.normalizedDemand > right.normalizedDemand;
                return left.key < right.key;
              });

    for (std::size_t i = maxComponents; i < components.size(); ++i)
      unmodeledGpuTime += components[i].normalizedDemand;
    components.resize(maxComponents);
  }

  void pup(PUP::er& p)
  {
    p | sourceInstanceId;
    p | sourceTypeId;
    p | components;
    p | unmodeledGpuTime;
  }
};

enum class GpuCostMetric : uint8_t
{
  NormalizedDemand = 0,
  RawDuration = 1,
};

enum class GpuPredictionSource : uint8_t
{
  Invalid = 0,
  PriorOnly = 1,
  Mixed = 2,
  Calibrated = 3,
};

class GpuScalingModel
{
public:
  explicit GpuScalingModel(double alphaMin = 0.0, uint64_t minSamples = 1);

  bool configure(double alphaMin, uint64_t minSamples);
  double alphaMin() const { return alphaMin_; }
  uint64_t minSamples() const { return minSamples_; }

  bool registerGpuType(uint64_t typeId, double peakRateScore);
  bool hasGpuType(uint64_t typeId) const;
  const GpuDeviceTypeInfo* findGpuType(uint64_t typeId) const;

  // Selects the most common registered type in availableTypes. An existing
  // reference is retained while present; ties otherwise use the smallest type
  // id so all replicas make the same choice.
  bool selectReference(const std::vector<uint64_t>& availableTypes);
  bool setReference(uint64_t typeId);
  bool hasReference() const { return hasReference_; }
  uint64_t referenceType() const { return referenceType_; }

  bool observe(const GpuKernelKey& kernel, uint64_t gpuType, GpuCostMetric metric,
               double cost, uint64_t epoch);
  bool observe(const GpuKernelKey& kernel, uint64_t gpuType, double normalizedDemand,
               double rawDuration, uint64_t epoch);
  // Log-domain entry point. The epoch summaries already carry mean log costs,
  // so going through exp() and back would only lose precision.
  bool observeLog(const GpuKernelKey& kernel, uint64_t gpuType,
                  double logNormalizedDemand, double logDuration, uint64_t epoch);

  // Feeds one object's epoch summary to the model, one observation per
  // component. Weighting is deliberately per component rather than per call: a
  // component's mean log cost is already an estimate of the per-invocation cost,
  // and the running mean counts observations. A component backed by a thousand
  // calls is a lower-variance estimate than one backed by two but currently
  // carries the same weight; inverse-variance weighting is a later refinement.
  // Returns the number of components accepted, and reports how many were
  // rejected as malformed.
  std::size_t observeObjectCosts(const GpuObjectEpochCosts& costs, uint64_t epoch,
                                 std::size_t* rejected = nullptr);

  // Ratio of hardware rate priors alone, used to price demand that has no
  // modeled identity.
  bool priorScale(uint64_t sourceType, uint64_t destinationType, double& scale) const;

  // Predicted whole-device GPU demand, in seconds, if this object's epoch of
  // work were replayed on destinationType. The residual is priced with the
  // hardware prior rather than dropped. `weakest` reports the least-calibrated
  // component that contributed, which is what a strategy should throttle on.
  bool predictObjectCost(const GpuObjectEpochCosts& costs, uint64_t destinationType,
                         GpuCostMetric metric, double& predictedCost,
                         GpuPredictionSource* weakest = nullptr) const;

  const GpuScalingEntry* findEntry(const GpuKernelKey& kernel, uint64_t gpuType) const;

  bool derivedLogE(const GpuKernelKey& kernel, uint64_t gpuType, GpuCostMetric metric,
                   double& logE) const;

  bool predictCost(const GpuKernelKey& kernel, uint64_t sourceType,
                   uint64_t destinationType, GpuCostMetric metric,
                   double observedSourceCost, double& predictedCost,
                   GpuPredictionSource* predictionSource = nullptr) const;

  void clear();
  std::size_t entryCount() const { return entries_.size(); }

  void pup(PUP::er& p);

private:
  static const GpuAdaptiveLogStats& statsFor(const GpuScalingEntry& entry,
                                             GpuCostMetric metric);
  static GpuAdaptiveLogStats& statsFor(GpuScalingEntry& entry, GpuCostMetric metric);

  bool logRateScore(const GpuKernelKey& kernel, uint64_t gpuType, GpuCostMetric metric,
                    double& logRate, bool& learned) const;

  static GpuPredictionSource weaker(GpuPredictionSource left,
                                    GpuPredictionSource right);

  std::unordered_map<uint64_t, GpuDeviceTypeInfo> gpuTypes_;
  std::unordered_map<GpuKernelTypeKey, GpuScalingEntry, GpuKernelTypeKeyHash> entries_;
  uint64_t referenceType_ = 0;
  bool hasReference_ = false;
  double alphaMin_ = 0.0;
  uint64_t minSamples_ = 1;
};

// One prediction made in a previous epoch, held until the object has run again
// so the claim can be scored against what actually happened.
struct GpuShadowPrediction
{
  uint64_t destinationType = 0;
  double predictedCost = 0.0;
  GpuPredictionSource source = GpuPredictionSource::Invalid;
  uint64_t epoch = 0;
};

// Accuracy of predictions that were later checked against what actually
// happened. Kept separate from the model so shadow mode can be evaluated
// without any of it feeding back into placement, and so the estimator can be
// judged independently of whichever strategy consumes it.
class GpuPredictionAccuracy
{
public:
  // Beyond this many retained samples per bucket the counters keep advancing
  // but percentiles stop seeing new data. Reported by `truncated` so a summary
  // never silently claims to describe more than it measured.
  static constexpr std::size_t retainedPerSource() { return 8192; }

  bool observe(double predicted, double actual, GpuPredictionSource source);

  uint64_t samples(GpuPredictionSource source) const;
  uint64_t truncated(GpuPredictionSource source) const;
  double meanAbsolutePercentageError(GpuPredictionSource source) const;
  // quantile in [0, 1]; 0.5 is the median. Returns false with no samples.
  bool quantileAbsolutePercentageError(GpuPredictionSource source, double quantile,
                                       double& value) const;

  void clear();

private:
  struct Bucket
  {
    uint64_t samples = 0;
    uint64_t truncated = 0;
    double sumAbsolutePercentageError = 0.0;
    std::vector<double> retained;
  };

  static std::size_t indexOf(GpuPredictionSource source);
  const Bucket& bucketFor(GpuPredictionSource source) const;

  Bucket buckets_[4];
};

#endif
