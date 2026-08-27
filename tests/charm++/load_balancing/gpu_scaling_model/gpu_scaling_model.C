#include "GpuScalingModel.h"
#include "charm++.h"
#include "gpu_scaling_model.decl.h"
#include "lbdb.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>

namespace
{

void fail(const char* test, const char* detail)
{
  CkPrintf("GpuScalingModel test failed: %s: %s\n", test, detail);
  CkExit(1);
}

void expect(bool condition, const char* test, const char* detail)
{
  if (!condition)
    fail(test, detail);
}

void expectNear(double actual, double expected, double tolerance, const char* test,
                const char* detail)
{
  if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
    fail(test, detail);
}

void testRunningMoments()
{
  const char* test = "running moments";
  GpuRunningMoments serial;
  for (double value : {1.0, 2.0, 3.0, 4.0})
    expect(serial.observe(value), test, "serial observation rejected");

  GpuRunningMoments left;
  GpuRunningMoments right;
  expect(left.observe(1.0) && left.observe(2.0), test, "left observation rejected");
  expect(right.observe(3.0) && right.observe(4.0), test, "right observation rejected");
  expect(left.merge(right), test, "merge rejected");

  expect(left.samples == serial.samples, test, "merged sample count differs");
  expectNear(left.mean, serial.mean, 1.0e-14, test, "merged mean differs");
  expectNear(left.M2, serial.M2, 1.0e-14, test, "merged M2 differs");
  expectNear(serial.sampleVariance(), 5.0 / 3.0, 1.0e-14, test,
             "sample variance is wrong");
}

void testKeyHashing()
{
  const char* test = "key hashing";
  const GpuKernelKey kernel(3, 9);
  const GpuKernelTypeKey typed(kernel, 17);
  GpuKernelKeyHash kernelHash;
  GpuKernelTypeKeyHash typeHash;

  expect(kernelHash(kernel) == kernelHash(GpuKernelKey(3, 9)), test,
         "equal kernel keys have different hashes");
  expect(typeHash(typed) == typeHash(GpuKernelTypeKey(GpuKernelKey(3, 9), 17)), test,
         "equal typed keys have different hashes");

  std::unordered_map<GpuKernelTypeKey, int, GpuKernelTypeKeyHash> values;
  values.emplace(typed, 42);
  expect(values.at(GpuKernelTypeKey(GpuKernelKey(3, 9), 17)) == 42, test,
         "typed key cannot be retrieved from a hash table");
}

void testCaptureIdentities()
{
  const char* test = "capture identities";
  GpuObjectTokenTable tokens;
  LDObjKey first{};
  LDObjKey second{};
  first.omID().id.idx = 10;
  second.omID().id.idx = 11;
  first.objID() = 7;
  second.objID() = 7;

  uint64_t firstToken = 0;
  uint64_t repeatedToken = 0;
  uint64_t secondToken = 0;
  expect(tokens.intern(first, firstToken), test, "first identity was rejected");
  expect(tokens.intern(first, repeatedToken), test, "repeated identity was rejected");
  expect(tokens.intern(second, secondToken), test, "second identity was rejected");
  expect(firstToken == repeatedToken, test, "an identity did not reuse its token");
  expect(firstToken != secondToken, test,
         "equal element IDs from different collections aliased");
  expect(firstToken != GpuObjectTokenTable::noObjectToken() &&
             secondToken != GpuObjectTokenTable::noObjectToken(),
         test, "a real object received the unattributed sentinel");

  LDObjKey resolved{};
  expect(tokens.resolve(secondToken, resolved), test, "token lookup failed");
  expect(resolved == second, test, "token resolved to the wrong full identity");
  expect(tokens.size() == 2, test, "token table has the wrong size");

  LDObjKeyHash objectHash;
  expect(objectHash(first) == objectHash(first), test,
         "equal object identities have different hashes");

  std::unordered_map<LDObjKey, double, LDObjKeyHash> loads;
  loads[first] = 1.0;
  loads[second] = 2.0;
  expect(loads.size() == 2 && loads.at(first) == 1.0 && loads.at(second) == 2.0, test,
         "full-key load map merged different collections");
}

void testStableCaptureMetadata()
{
  const char* test = "stable capture metadata";
  expect(gpuStableKernelClass("stepKernel") == gpuStableKernelClass("stepKernel"), test,
         "kernel class is not stable");
  expect(gpuStableKernelClass("stepKernel") != gpuStableKernelClass("otherKernel"), test,
         "distinct test names collided");

  GpuLaunchSignature launch;
  launch.gridX = 64;
  launch.gridY = 2;
  launch.gridZ = 1;
  launch.blockX = 128;
  launch.blockY = 1;
  launch.blockZ = 1;
  launch.staticSharedMemory = 256;
  launch.dynamicSharedMemory = 512;

  const uint64_t automatic = gpuStableWorkBucket(launch, false, 0);
  expect(automatic == gpuStableWorkBucket(launch, false, 0), test,
         "automatic work bucket is not stable");
  expect(automatic != gpuStableWorkBucket(launch, true, 0), test,
         "tagged and untagged launches aliased");
  expect(gpuStableWorkBucket(launch, true, 17) != gpuStableWorkBucket(launch, true, 18),
         test, "distinct explicit work tags aliased");

  GpuLaunchSignature changed = launch;
  changed.gridX++;
  expect(automatic != gpuStableWorkBucket(changed, false, 0), test,
         "different launch geometry aliased");

  const uint64_t deviceType =
      gpuStableDeviceType("Example GPU", 80, 8, 0, 1400000, UINT64_C(40000000000));
  expect(deviceType ==
             gpuStableDeviceType("Example GPU", 80, 8, 0, 1400000, UINT64_C(40000000000)),
         test, "device type is not stable");
  expect(deviceType !=
             gpuStableDeviceType("Example GPU", 40, 8, 0, 1400000, UINT64_C(20000000000)),
         test, "different device slices aliased");

  GpuDeviceDescriptor descriptor;
  descriptor.instanceId = 12;
  descriptor.typeId = deviceType;
  descriptor.peakRateScore = 112000000.0;
  descriptor.smCount = 80;
  descriptor.computeMajor = 8;
  descriptor.maxClockKHz = 1400000;
  descriptor.totalMemory = UINT64_C(40000000000);
  descriptor.peakRateSource = GpuPeakRateSource::UserOverride;

  PUP::sizer sizer;
  descriptor.pup(sizer);
  std::vector<char> buffer(sizer.size());
  PUP::toMem packer(buffer.data());
  descriptor.pup(packer);
  GpuDeviceDescriptor restored;
  PUP::fromMem unpacker(buffer.data());
  restored.pup(unpacker);
  expect(restored.instanceId == descriptor.instanceId &&
             restored.typeId == descriptor.typeId &&
             restored.peakRateScore == descriptor.peakRateScore &&
             restored.smCount == descriptor.smCount &&
             restored.computeMajor == descriptor.computeMajor &&
             restored.maxClockKHz == descriptor.maxClockKHz &&
             restored.totalMemory == descriptor.totalMemory &&
             restored.peakRateSource == descriptor.peakRateSource,
         test, "device descriptor changed after PUP");
}

void testPeakRatePrior()
{
  const char* test = "peak rate prior";

  // Lane counts per SM differ across architectures, so a prior built from SMs
  // and clock alone misranks a mixed allocation. A100 (8.0, 64 lanes/SM) and
  // H100 (9.0, 128 lanes/SM) are the case that matters.
  double ampere = 0.0;
  double hopper = 0.0;
  GpuPeakRateSource ampereSource = GpuPeakRateSource::Unknown;
  GpuPeakRateSource hopperSource = GpuPeakRateSource::Unknown;
  expect(gpuDerivePeakRateScore(108, 8, 0, 1410000, ampere, ampereSource), test,
         "A100 rate derivation failed");
  expect(gpuDerivePeakRateScore(132, 9, 0, 1980000, hopper, hopperSource), test,
         "H100 rate derivation failed");
  expect(ampereSource == GpuPeakRateSource::ArchitectureTable &&
             hopperSource == GpuPeakRateSource::ArchitectureTable,
         test, "a tabulated architecture was not reported as tabulated");

  // Reference FP32 peaks are 19.5 and 67 TFLOP/s, a ratio of about 3.44. The
  // SM-times-clock proxy this replaced gave about 1.72.
  const double ratio = hopper / ampere;
  expect(ratio > 3.2 && ratio < 3.7, test,
         "architecture-aware ratio does not track peak FP32 throughput");
  const double proxyRatio = (132.0 * 1980000.0) / (108.0 * 1410000.0);
  expect(proxyRatio < 2.0, test, "sm-clock proxy assumption no longer holds");

  // An unknown capability still produces a comparable score: it stands in a
  // nominal lane count rather than changing units.
  double futureArch = 0.0;
  GpuPeakRateSource futureSource = GpuPeakRateSource::Unknown;
  expect(gpuCoresPerSm(99, 0) == 0, test, "unknown capability was tabulated");
  expect(gpuDerivePeakRateScore(100, 99, 0, 1500000, futureArch, futureSource), test,
         "unknown capability was rejected");
  expect(futureSource == GpuPeakRateSource::SmClockProxy, test,
         "unknown capability was not reported as a proxy");
  expectNear(futureArch,
             100.0 * static_cast<double>(gpuNominalCoresPerSm()) * 1500000.0, 0.0,
             test, "unknown capability did not use the nominal lane count");

  // A device that reports no clock keeps the units of one that does, so the
  // two stay within the spread of real clock rates instead of differing by
  // six orders of magnitude.
  double noClock = 0.0;
  GpuPeakRateSource noClockSource = GpuPeakRateSource::ArchitectureTable;
  expect(gpuDerivePeakRateScore(108, 8, 0, 0, noClock, noClockSource), test,
         "clockless device was rejected");
  expect(noClockSource == GpuPeakRateSource::Unknown, test,
         "a substituted clock was not reported as unknown");
  const double clockRatio = ampere / noClock;
  expect(clockRatio > 0.25 && clockRatio < 4.0, test,
         "clockless fallback left the score incomparable");

  // Too poorly described to rank at all.
  double noSms = 0.0;
  GpuPeakRateSource noSmsSource = GpuPeakRateSource::ArchitectureTable;
  expect(!gpuDerivePeakRateScore(0, 8, 0, 1410000, noSms, noSmsSource), test,
         "a device with no SMs was accepted");
  expectNear(noSms, 1.0, 0.0, test, "neutral score is wrong");
  expect(noSmsSource == GpuPeakRateSource::Unknown, test,
         "undescribed device was not reported as unknown");

  // registerGpuType compares scores exactly, so two reporters of one device
  // type must agree bit for bit.
  double again = 0.0;
  GpuPeakRateSource againSource = GpuPeakRateSource::Unknown;
  expect(gpuDerivePeakRateScore(108, 8, 0, 1410000, again, againSource), test,
         "repeat derivation failed");
  expect(again == ampere && againSource == ampereSource, test,
         "rate derivation is not bit-reproducible");

  GpuScalingModel model;
  expect(model.registerGpuType(1, ampere) && model.registerGpuType(1, again), test,
         "identical rederived score was rejected as a conflict");
}

void testEpochComponents()
{
  const char* test = "epoch components";

  GpuKernelEpochCost component;
  component.key = GpuKernelKey(5, 2);
  expect(component.observe(2.0, 4.0) && component.observe(6.0, 12.0), test,
         "observation rejected");
  expect(component.calls == 2, test, "call count is wrong");
  expectNear(component.normalizedDemand, 8.0, 1.0e-14, test, "demand did not sum");
  expectNear(component.logNormalizedDemand.mean,
             (std::log(2.0) + std::log(6.0)) / 2.0, 1.0e-14, test,
             "log demand mean is wrong");
  expectNear(component.logDuration.mean, (std::log(4.0) + std::log(12.0)) / 2.0,
             1.0e-14, test, "log duration mean is wrong");

  // A rejected sample must leave every field untouched, or the summary stops
  // reconciling with the scalar load.
  const uint64_t callsBefore = component.calls;
  const double demandBefore = component.normalizedDemand;
  const double meanBefore = component.logNormalizedDemand.mean;
  expect(!component.observe(0.0, 4.0), test, "zero demand accepted");
  expect(!component.observe(2.0, -1.0), test, "negative duration accepted");
  expect(component.calls == callsBefore &&
             component.normalizedDemand == demandBefore &&
             component.logNormalizedDemand.mean == meanBefore,
         test, "a rejected sample changed the component");

  GpuKernelEpochCost other;
  other.key = GpuKernelKey(5, 2);
  expect(other.observe(4.0, 8.0), test, "second component observation rejected");
  expect(component.merge(other), test, "merge of matching keys rejected");
  expect(component.calls == 3, test, "merged call count is wrong");
  expectNear(component.normalizedDemand, 12.0, 1.0e-14, test,
             "merged demand is wrong");

  GpuKernelEpochCost mismatched;
  mismatched.key = GpuKernelKey(5, 3);
  expect(!component.merge(mismatched), test, "merge across keys accepted");
}

void testComponentCapAndResidual()
{
  const char* test = "component cap and residual";

  GpuObjectEpochCosts costs;
  costs.sourceTypeId = 77;
  const double demands[] = {1.0, 5.0, 3.0, 9.0, 2.0};
  for (int i = 0; i < 5; ++i)
  {
    GpuKernelEpochCost component;
    component.key = GpuKernelKey(100 + i, 0);
    expect(component.observe(demands[i], demands[i]), test, "observation rejected");
    costs.components.push_back(component);
  }
  const double total = 1.0 + 5.0 + 3.0 + 9.0 + 2.0;
  expectNear(costs.totalDemand(), total, 1.0e-14, test, "uncapped total is wrong");

  costs.enforceComponentCap(2);
  expect(costs.components.size() == 2, test, "cap was not applied");
  // The two largest survive; everything else becomes residual, so the total is
  // unchanged. That invariant is what lets a load balancer trust the summary.
  expectNear(costs.modeledDemand(), 14.0, 1.0e-14, test,
             "cap did not keep the largest components");
  expectNear(costs.unmodeledGpuTime, total - 14.0, 1.0e-14, test,
             "residual does not account for the dropped components");
  expectNear(costs.totalDemand(), total, 1.0e-14, test,
             "cap changed the object total");

  // Deterministic across input orderings: a replica that saw the same
  // components in a different order must keep the same ones.
  GpuObjectEpochCosts reversed;
  reversed.sourceTypeId = 77;
  for (int i = 4; i >= 0; --i)
  {
    GpuKernelEpochCost component;
    component.key = GpuKernelKey(100 + i, 0);
    expect(component.observe(demands[i], demands[i]), test, "observation rejected");
    reversed.components.push_back(component);
  }
  reversed.enforceComponentCap(2);
  expect(reversed.components.size() == costs.components.size(), test,
         "cap kept a different number of components");
  for (std::size_t i = 0; i < reversed.components.size(); ++i)
    expect(reversed.components[i].key == costs.components[i].key, test,
           "cap is sensitive to input order");

  GpuObjectEpochCosts everything;
  GpuKernelEpochCost only;
  only.key = GpuKernelKey(1, 1);
  expect(only.observe(4.0, 4.0), test, "observation rejected");
  everything.components.push_back(only);
  everything.enforceComponentCap(0);
  expect(everything.components.empty(), test, "zero cap kept a component");
  expectNear(everything.totalDemand(), 4.0, 1.0e-14, test,
             "zero cap lost the demand");
}

void testEpochCostsPupRoundTrip()
{
  const char* test = "epoch costs PUP round trip";

  GpuObjectEpochCosts original;
  original.sourceInstanceId = 9;
  original.sourceTypeId = 12345;
  original.unmodeledGpuTime = 0.25;
  for (int i = 0; i < 3; ++i)
  {
    GpuKernelEpochCost component;
    component.key = GpuKernelKey(200 + i, i);
    expect(component.observe(1.0 + i, 2.0 + i) && component.observe(3.0 + i, 4.0 + i),
           test, "observation rejected");
    original.components.push_back(component);
  }

  PUP::sizer sizer;
  original.pup(sizer);
  std::vector<char> buffer(sizer.size());
  PUP::toMem packer(buffer.data());
  original.pup(packer);

  GpuObjectEpochCosts restored;
  PUP::fromMem unpacker(buffer.data());
  restored.pup(unpacker);

  expect(restored.sourceInstanceId == original.sourceInstanceId &&
             restored.sourceTypeId == original.sourceTypeId,
         test, "source identity changed after PUP");
  expectNear(restored.unmodeledGpuTime, original.unmodeledGpuTime, 0.0, test,
             "residual changed after PUP");
  expect(restored.components.size() == original.components.size(), test,
         "component count changed after PUP");
  for (std::size_t i = 0; i < restored.components.size(); ++i)
  {
    expect(restored.components[i].key == original.components[i].key, test,
           "component key changed after PUP");
    expect(restored.components[i].calls == original.components[i].calls, test,
           "component call count changed after PUP");
    expectNear(restored.components[i].normalizedDemand,
               original.components[i].normalizedDemand, 0.0, test,
               "component demand changed after PUP");
    expectNear(restored.components[i].logNormalizedDemand.mean,
               original.components[i].logNormalizedDemand.mean, 0.0, test,
               "component log demand mean changed after PUP");
    expectNear(restored.components[i].logDuration.M2,
               original.components[i].logDuration.M2, 0.0, test,
               "component log duration M2 changed after PUP");
  }
  expectNear(restored.totalDemand(), original.totalDemand(), 0.0, test,
             "total demand changed after PUP");
}

void testStationaryEstimator()
{
  const char* test = "stationary estimator";
  GpuScalingModel model;
  const GpuKernelKey kernel(11, 7);
  expect(model.registerGpuType(1, 10.0), test, "reference registration failed");
  expect(model.registerGpuType(2, 20.0), test, "target registration failed");
  expect(model.setReference(1), test, "reference selection failed");

  const double observations[] = {8.0, 4.0, 2.0, 1.0};
  for (uint64_t i = 0; i < 4; ++i)
    expect(
        model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, observations[i], i + 1),
        test, "observation rejected");

  const GpuScalingEntry* entry = model.findEntry(kernel, 1);
  expect(entry != nullptr, test, "entry missing");
  const double expected =
      (std::log(8.0) + std::log(4.0) + std::log(2.0) + std::log(1.0)) / 4.0;
  expectNear(entry->normalizedDemand.estimateMean, expected, 1.0e-14, test,
             "1/n estimate does not equal batch log mean");
}

void testNoiselessScaling()
{
  const char* test = "noiseless scaling";
  GpuScalingModel model;
  const GpuKernelKey kernel(21, 3);
  expect(model.registerGpuType(1, 10.0), test, "reference registration failed");
  expect(model.registerGpuType(2, 20.0), test, "target registration failed");
  expect(model.setReference(1), test, "reference selection failed");
  expect(model.observe(kernel, 1, 8.0, 8.0, 1), test, "reference observation failed");
  expect(model.observe(kernel, 2, 5.0, 5.0, 1), test, "target observation failed");

  double logE = 0.0;
  expect(model.derivedLogE(kernel, 2, GpuCostMetric::NormalizedDemand, logE), test,
         "logE unavailable");
  expectNear(std::exp(logE), 0.8, 1.0e-14, test, "learned E is wrong");

  double predicted = 0.0;
  GpuPredictionSource source = GpuPredictionSource::Invalid;
  expect(model.predictCost(kernel, 1, 2, GpuCostMetric::NormalizedDemand, 8.0, predicted,
                           &source),
         test, "prediction failed");
  expectNear(predicted, 5.0, 1.0e-13, test, "destination cost is wrong");
  expect(source == GpuPredictionSource::Calibrated, test,
         "calibrated prediction was not classified as calibrated");
}

void testPriorAndMinimumSamples()
{
  const char* test = "prior and minimum samples";
  GpuScalingModel model(0.0, 2);
  const GpuKernelKey kernel(31, 4);
  expect(model.registerGpuType(1, 10.0), test, "source registration failed");
  expect(model.registerGpuType(2, 20.0), test, "destination registration failed");
  expect(model.setReference(1), test, "reference selection failed");

  double predicted = 0.0;
  GpuPredictionSource source = GpuPredictionSource::Invalid;
  expect(model.predictCost(kernel, 1, 2, GpuCostMetric::NormalizedDemand, 8.0, predicted,
                           &source),
         test, "prior prediction failed");
  expectNear(predicted, 4.0, 1.0e-14, test, "peak-rate prior is wrong");
  expect(source == GpuPredictionSource::PriorOnly, test,
         "unseen prediction was not prior-only");

  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 8.0, 1) &&
             model.observe(kernel, 2, GpuCostMetric::NormalizedDemand, 5.0, 1),
         test, "first observations failed");
  expect(model.predictCost(kernel, 1, 2, GpuCostMetric::NormalizedDemand, 8.0, predicted,
                           &source),
         test, "low-sample prediction failed");
  expectNear(predicted, 4.0, 1.0e-14, test,
             "low-sample entry replaced the prior too early");

  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 8.0, 2) &&
             model.observe(kernel, 2, GpuCostMetric::NormalizedDemand, 5.0, 2),
         test, "second observations failed");
  expect(model.predictCost(kernel, 1, 2, GpuCostMetric::NormalizedDemand, 8.0, predicted,
                           &source),
         test, "calibrated prediction failed");
  expectNear(predicted, 5.0, 1.0e-13, test, "calibrated prediction is wrong");
}

void testMixedPrediction()
{
  const char* test = "mixed prediction";
  GpuScalingModel model;
  const GpuKernelKey kernel(36, 4);
  expect(model.registerGpuType(1, 10.0) && model.registerGpuType(2, 20.0) &&
             model.registerGpuType(3, 40.0),
         test, "type registration failed");
  expect(model.setReference(1), test, "reference selection failed");
  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 8.0, 1) &&
             model.observe(kernel, 2, GpuCostMetric::NormalizedDemand, 5.0, 1),
         test, "calibration observations failed");

  double predicted = 0.0;
  GpuPredictionSource source = GpuPredictionSource::Invalid;
  expect(model.predictCost(kernel, 2, 3, GpuCostMetric::NormalizedDemand, 5.0, predicted,
                           &source),
         test, "prediction failed");
  expectNear(predicted, 2.0, 1.0e-13, test, "mixed prediction is wrong");
  expect(source == GpuPredictionSource::Mixed, test,
         "prediction was not classified as mixed");
}

void testAdaptiveFloor()
{
  const char* test = "adaptive floor";
  GpuScalingModel model(0.5, 1);
  const GpuKernelKey kernel(41, 5);
  expect(model.registerGpuType(1, 1.0), test, "type registration failed");
  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 10.0, 1), test,
         "first observation failed");
  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 2.0, 2), test,
         "second observation failed");
  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 2.0, 3), test,
         "third observation failed");

  const GpuScalingEntry* entry = model.findEntry(kernel, 1);
  expect(entry != nullptr, test, "entry missing");
  const double afterTwo = 0.5 * std::log(10.0) + 0.5 * std::log(2.0);
  const double expected = 0.5 * afterTwo + 0.5 * std::log(2.0);
  expectNear(entry->normalizedDemand.estimateMean, expected, 1.0e-14, test,
             "EWMA floor was not applied");
}

void testRebasing()
{
  const char* test = "rebasing";
  GpuScalingModel model;
  const GpuKernelKey kernel(51, 6);
  expect(model.registerGpuType(1, 10.0) && model.registerGpuType(2, 20.0) &&
             model.registerGpuType(3, 5.0),
         test, "type registration failed");
  expect(model.setReference(1), test, "initial reference failed");
  expect(model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 12.0, 1) &&
             model.observe(kernel, 2, GpuCostMetric::NormalizedDemand, 6.0, 1) &&
             model.observe(kernel, 3, GpuCostMetric::NormalizedDemand, 24.0, 1),
         test, "observations failed");

  double before = 0.0;
  double after = 0.0;
  expect(model.predictCost(kernel, 2, 3, GpuCostMetric::NormalizedDemand, 6.0, before),
         test, "prediction before rebase failed");
  expect(model.setReference(2), test, "new reference failed");
  expect(model.predictCost(kernel, 2, 3, GpuCostMetric::NormalizedDemand, 6.0, after),
         test, "prediction after rebase failed");
  expectNear(before, 24.0, 1.0e-12, test, "pre-rebase prediction is wrong");
  expectNear(after, before, 1.0e-12, test, "rebasing changed a pairwise prediction");

  // The old reference disappeared; the most common remaining type wins and a
  // tie is deterministic.
  expect(model.setReference(1), test, "reference reset failed");
  expect(model.selectReference(std::vector<uint64_t>{3, 2, 2, 3}), test,
         "automatic reference selection failed");
  expect(model.referenceType() == 2, test, "reference tie was not broken by type id");
}

void testInvalidInputs()
{
  const char* test = "invalid inputs";
  GpuScalingModel model;
  const GpuKernelKey kernel(61, 7);
  expect(!model.configure(-0.1, 1), test, "negative alpha accepted");
  expect(!model.configure(0.0, 0), test, "zero minSamples accepted");
  expect(!model.registerGpuType(1, 0.0), test, "zero peak rate accepted");
  expect(model.registerGpuType(1, 1.0), test, "valid type rejected");
  expect(!model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, 0.0, 1), test,
         "zero cost accepted");
  expect(!model.observe(kernel, 1, GpuCostMetric::NormalizedDemand, -1.0, 1), test,
         "negative cost accepted");
  expect(!model.observe(kernel, 1, GpuCostMetric::NormalizedDemand,
                        std::numeric_limits<double>::quiet_NaN(), 1),
         test, "NaN cost accepted");
  expect(model.findEntry(kernel, 1) == nullptr, test,
         "invalid observations created an entry");
  expect(!model.observe(kernel, 1, 2.0, 0.0, 1), test,
         "partially invalid paired observation accepted");
  expect(model.findEntry(kernel, 1) == nullptr, test,
         "partially invalid paired observation changed the model");
}

void testPupRoundTrip()
{
  const char* test = "PUP round trip";
  GpuScalingModel original(0.1, 2);
  const GpuKernelKey kernel(71, 8);
  expect(original.registerGpuType(1, 12.0) && original.registerGpuType(2, 30.0), test,
         "type registration failed");
  expect(original.setReference(1), test, "reference selection failed");
  for (uint64_t epoch = 1; epoch <= 3; ++epoch)
  {
    expect(original.observe(kernel, 1, 9.0, 10.0, epoch) &&
               original.observe(kernel, 2, 4.5, 5.0, epoch),
           test, "observation failed");
  }

  PUP::sizer sizer;
  original.pup(sizer);
  std::vector<char> buffer(sizer.size());
  PUP::toMem packer(buffer.data());
  original.pup(packer);

  GpuScalingModel restored;
  PUP::fromMem unpacker(buffer.data());
  restored.pup(unpacker);

  double originalPrediction = 0.0;
  double restoredPrediction = 0.0;
  expect(original.predictCost(kernel, 1, 2, GpuCostMetric::NormalizedDemand, 9.0,
                              originalPrediction),
         test, "original prediction failed");
  expect(restored.predictCost(kernel, 1, 2, GpuCostMetric::NormalizedDemand, 9.0,
                              restoredPrediction),
         test, "restored prediction failed");
  expectNear(restoredPrediction, originalPrediction, 1.0e-14, test,
             "prediction changed after PUP");
  expect(restored.referenceType() == original.referenceType(), test,
         "reference changed after PUP");
  expect(restored.entryCount() == original.entryCount(), test,
         "entry count changed after PUP");
  expectNear(restored.alphaMin(), original.alphaMin(), 0.0, test,
             "alpha changed after PUP");
  expect(restored.minSamples() == original.minSamples(), test,
         "minimum sample count changed after PUP");
}

}  // namespace

class Main : public CBase_Main
{
public:
  explicit Main(CkArgMsg* msg)
  {
    delete msg;
    testRunningMoments();
    testKeyHashing();
    testCaptureIdentities();
    testStableCaptureMetadata();
    testPeakRatePrior();
    testEpochComponents();
    testComponentCapAndResidual();
    testEpochCostsPupRoundTrip();
    testStationaryEstimator();
    testNoiselessScaling();
    testPriorAndMinimumSamples();
    testMixedPrediction();
    testAdaptiveFloor();
    testRebasing();
    testInvalidInputs();
    testPupRoundTrip();
    CkPrintf("GpuScalingModel: all tests passed\n");
    CkExit();
  }
};

#include "gpu_scaling_model.def.h"
