#include "GpuScalingModel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

using gpu_scaling_detail::validPositive;

bool validAlpha(double alphaMin)
{
  return std::isfinite(alphaMin) && alphaMin >= 0.0 && alphaMin <= 1.0;
}

}  // namespace

bool GpuAdaptiveLogStats::observeLogCost(double logCost, uint64_t epoch, double alphaMin)
{
  if (!std::isfinite(logCost) || !validAlpha(alphaMin))
    return false;

  if (!lifetime.observe(logCost))
    return false;

  if (lifetime.samples == 1)
  {
    estimateMean = logCost;
  }
  else
  {
    const double runningMeanAlpha = 1.0 / static_cast<double>(lifetime.samples);
    const double alpha = std::max(alphaMin, runningMeanAlpha);
    estimateMean = (1.0 - alpha) * estimateMean + alpha * logCost;
  }
  lastEpoch = epoch;
  return std::isfinite(estimateMean);
}

GpuScalingModel::GpuScalingModel(double alphaMin, uint64_t minSamples)
{
  configure(alphaMin, minSamples);
}

bool GpuScalingModel::configure(double alphaMin, uint64_t minSamples)
{
  if (!validAlpha(alphaMin) || minSamples == 0)
    return false;
  alphaMin_ = alphaMin;
  minSamples_ = minSamples;
  return true;
}

bool GpuScalingModel::registerGpuType(uint64_t typeId, double peakRateScore)
{
  if (!validPositive(peakRateScore))
    return false;

  auto found = gpuTypes_.find(typeId);
  if (found != gpuTypes_.end())
  {
    // A type id is a fingerprint of the rate-relevant device properties. If
    // two reporters disagree on its score, accepting the later one would make
    // model replicas depend on message order.
    return found->second.peakRateScore == peakRateScore;
  }

  gpuTypes_.emplace(typeId, GpuDeviceTypeInfo(typeId, peakRateScore));
  return true;
}

bool GpuScalingModel::hasGpuType(uint64_t typeId) const
{
  return gpuTypes_.find(typeId) != gpuTypes_.end();
}

const GpuDeviceTypeInfo* GpuScalingModel::findGpuType(uint64_t typeId) const
{
  auto found = gpuTypes_.find(typeId);
  return found == gpuTypes_.end() ? nullptr : &found->second;
}

bool GpuScalingModel::selectReference(const std::vector<uint64_t>& availableTypes)
{
  std::unordered_map<uint64_t, uint64_t> counts;
  for (uint64_t typeId : availableTypes)
  {
    if (hasGpuType(typeId))
      ++counts[typeId];
  }
  if (counts.empty())
    return false;

  if (hasReference_ && counts.find(referenceType_) != counts.end())
    return true;

  uint64_t selected = counts.begin()->first;
  uint64_t selectedCount = counts.begin()->second;
  for (const auto& item : counts)
  {
    if (item.second > selectedCount ||
        (item.second == selectedCount && item.first < selected))
    {
      selected = item.first;
      selectedCount = item.second;
    }
  }

  referenceType_ = selected;
  hasReference_ = true;
  return true;
}

bool GpuScalingModel::setReference(uint64_t typeId)
{
  if (!hasGpuType(typeId))
    return false;
  referenceType_ = typeId;
  hasReference_ = true;
  return true;
}

GpuAdaptiveLogStats& GpuScalingModel::statsFor(GpuScalingEntry& entry,
                                               GpuCostMetric metric)
{
  return metric == GpuCostMetric::NormalizedDemand ? entry.normalizedDemand
                                                   : entry.rawDuration;
}

const GpuAdaptiveLogStats& GpuScalingModel::statsFor(const GpuScalingEntry& entry,
                                                     GpuCostMetric metric)
{
  return metric == GpuCostMetric::NormalizedDemand ? entry.normalizedDemand
                                                   : entry.rawDuration;
}

bool GpuScalingModel::observe(const GpuKernelKey& kernel, uint64_t gpuType,
                              GpuCostMetric metric, double cost, uint64_t epoch)
{
  if (!hasGpuType(gpuType) || !validPositive(cost))
    return false;

  GpuScalingEntry& entry = entries_[GpuKernelTypeKey(kernel, gpuType)];
  return statsFor(entry, metric).observeLogCost(std::log(cost), epoch, alphaMin_);
}

bool GpuScalingModel::observe(const GpuKernelKey& kernel, uint64_t gpuType,
                              double normalizedDemand, double rawDuration, uint64_t epoch)
{
  // Validate both values before updating either series so a malformed paired
  // observation cannot leave only one metric advanced.
  if (!hasGpuType(gpuType) || !validPositive(normalizedDemand) ||
      !validPositive(rawDuration))
    return false;

  GpuScalingEntry& entry = entries_[GpuKernelTypeKey(kernel, gpuType)];
  const bool normalizedAccepted =
      entry.normalizedDemand.observeLogCost(std::log(normalizedDemand), epoch, alphaMin_);
  const bool durationAccepted =
      entry.rawDuration.observeLogCost(std::log(rawDuration), epoch, alphaMin_);
  return normalizedAccepted && durationAccepted;
}

bool GpuScalingModel::observeLog(const GpuKernelKey& kernel, uint64_t gpuType,
                                 double logNormalizedDemand, double logDuration,
                                 uint64_t epoch)
{
  if (!hasGpuType(gpuType) || !std::isfinite(logNormalizedDemand) ||
      !std::isfinite(logDuration))
    return false;

  GpuScalingEntry& entry = entries_[GpuKernelTypeKey(kernel, gpuType)];
  const bool normalizedAccepted =
      entry.normalizedDemand.observeLogCost(logNormalizedDemand, epoch, alphaMin_);
  const bool durationAccepted =
      entry.rawDuration.observeLogCost(logDuration, epoch, alphaMin_);
  return normalizedAccepted && durationAccepted;
}

std::size_t GpuScalingModel::observeObjectCosts(const GpuObjectEpochCosts& costs,
                                                uint64_t epoch,
                                                std::size_t* rejected)
{
  std::size_t accepted = 0;
  std::size_t refused = 0;
  if (!hasGpuType(costs.sourceTypeId))
  {
    // Nothing to learn: without a registered source device there is no rate to
    // divide out, so the observation carries no cross-GPU information.
    if (rejected != nullptr) *rejected = costs.components.size();
    return 0;
  }

  for (const GpuKernelEpochCost& component : costs.components)
  {
    if (component.calls == 0 || component.logNormalizedDemand.samples == 0 ||
        component.logDuration.samples == 0)
    {
      refused++;
      continue;
    }
    if (observeLog(component.key, costs.sourceTypeId,
                   component.logNormalizedDemand.mean, component.logDuration.mean,
                   epoch))
      accepted++;
    else
      refused++;
  }

  if (rejected != nullptr) *rejected = refused;
  return accepted;
}

bool GpuScalingModel::priorScale(uint64_t sourceType, uint64_t destinationType,
                                 double& scale) const
{
  const GpuDeviceTypeInfo* source = findGpuType(sourceType);
  const GpuDeviceTypeInfo* destination = findGpuType(destinationType);
  if (source == nullptr || destination == nullptr)
    return false;
  if (!validPositive(source->peakRateScore) ||
      !validPositive(destination->peakRateScore))
    return false;

  scale = source->peakRateScore / destination->peakRateScore;
  return std::isfinite(scale) && scale > 0.0;
}

GpuPredictionSource GpuScalingModel::weaker(GpuPredictionSource left,
                                            GpuPredictionSource right)
{
  // Invalid < PriorOnly < Mixed < Calibrated, so the smaller enumerator is the
  // weaker claim.
  return static_cast<uint8_t>(left) <= static_cast<uint8_t>(right) ? left : right;
}

bool GpuScalingModel::predictObjectCost(const GpuObjectEpochCosts& costs,
                                        uint64_t destinationType,
                                        GpuCostMetric metric, double& predictedCost,
                                        GpuPredictionSource* weakest) const
{
  if (weakest != nullptr) *weakest = GpuPredictionSource::Invalid;

  double residualScale = 0.0;
  if (!priorScale(costs.sourceTypeId, destinationType, residualScale))
    return false;

  double total = 0.0;
  // An object with no modeled components is still a valid prediction: its whole
  // demand is residual, priced by the hardware prior.
  GpuPredictionSource worst = GpuPredictionSource::Calibrated;
  bool anyResidual = costs.unmodeledGpuTime > 0.0;

  for (const GpuKernelEpochCost& component : costs.components)
  {
    if (!validPositive(component.normalizedDemand))
      continue;

    double componentCost = 0.0;
    GpuPredictionSource componentSource = GpuPredictionSource::Invalid;
    if (predictCost(component.key, costs.sourceTypeId, destinationType, metric,
                    component.normalizedDemand, componentCost, &componentSource))
    {
      total += componentCost;
      worst = weaker(worst, componentSource);
    }
    else
    {
      // Fall back to the hardware prior for this component rather than losing
      // its demand; a partial prediction that silently omits work is worse than
      // an uncalibrated one.
      total += component.normalizedDemand * residualScale;
      worst = weaker(worst, GpuPredictionSource::PriorOnly);
    }
  }

  if (anyResidual)
  {
    total += costs.unmodeledGpuTime * residualScale;
    worst = weaker(worst, GpuPredictionSource::PriorOnly);
  }

  if (costs.components.empty() && !anyResidual)
    worst = GpuPredictionSource::PriorOnly;

  if (!std::isfinite(total) || total < 0.0)
    return false;

  predictedCost = total;
  if (weakest != nullptr) *weakest = worst;
  return true;
}

const GpuScalingEntry* GpuScalingModel::findEntry(const GpuKernelKey& kernel,
                                                  uint64_t gpuType) const
{
  auto found = entries_.find(GpuKernelTypeKey(kernel, gpuType));
  return found == entries_.end() ? nullptr : &found->second;
}

bool GpuScalingModel::derivedLogE(const GpuKernelKey& kernel, uint64_t gpuType,
                                  GpuCostMetric metric, double& logE) const
{
  if (!hasReference_)
    return false;

  const GpuDeviceTypeInfo* reference = findGpuType(referenceType_);
  const GpuDeviceTypeInfo* target = findGpuType(gpuType);
  if (reference == nullptr || target == nullptr)
    return false;

  if (gpuType == referenceType_)
  {
    logE = 0.0;
    return true;
  }

  const GpuScalingEntry* referenceEntry = findEntry(kernel, referenceType_);
  const GpuScalingEntry* targetEntry = findEntry(kernel, gpuType);
  if (referenceEntry == nullptr || targetEntry == nullptr)
    return false;

  const GpuAdaptiveLogStats& referenceStats = statsFor(*referenceEntry, metric);
  const GpuAdaptiveLogStats& targetStats = statsFor(*targetEntry, metric);
  if (referenceStats.lifetime.samples < minSamples_ ||
      targetStats.lifetime.samples < minSamples_)
    return false;

  logE = std::log(reference->peakRateScore) - std::log(target->peakRateScore) +
         referenceStats.estimateMean - targetStats.estimateMean;
  return std::isfinite(logE);
}

bool GpuScalingModel::logRateScore(const GpuKernelKey& kernel, uint64_t gpuType,
                                   GpuCostMetric metric, double& logRate,
                                   bool& learned) const
{
  const GpuDeviceTypeInfo* type = findGpuType(gpuType);
  if (type == nullptr || !validPositive(type->peakRateScore))
    return false;

  logRate = std::log(type->peakRateScore);
  learned = false;

  double logE = 0.0;
  if (derivedLogE(kernel, gpuType, metric, logE))
  {
    logRate += logE;

    const GpuScalingEntry* entry = findEntry(kernel, gpuType);
    if (entry != nullptr)
      learned = statsFor(*entry, metric).lifetime.samples >= minSamples_;
  }
  return std::isfinite(logRate);
}

bool GpuScalingModel::predictCost(const GpuKernelKey& kernel, uint64_t sourceType,
                                  uint64_t destinationType, GpuCostMetric metric,
                                  double observedSourceCost, double& predictedCost,
                                  GpuPredictionSource* predictionSource) const
{
  if (predictionSource != nullptr)
    *predictionSource = GpuPredictionSource::Invalid;
  if (!validPositive(observedSourceCost))
    return false;

  if (sourceType == destinationType)
  {
    predictedCost = observedSourceCost;
    if (predictionSource != nullptr)
      *predictionSource = hasGpuType(sourceType) ? GpuPredictionSource::Calibrated
                                                 : GpuPredictionSource::PriorOnly;
    return true;
  }

  double sourceLogRate = 0.0;
  double destinationLogRate = 0.0;
  bool sourceLearned = false;
  bool destinationLearned = false;
  if (!logRateScore(kernel, sourceType, metric, sourceLogRate, sourceLearned) ||
      !logRateScore(kernel, destinationType, metric, destinationLogRate,
                    destinationLearned))
    return false;

  const double logPrediction =
      std::log(observedSourceCost) + sourceLogRate - destinationLogRate;
  if (!std::isfinite(logPrediction) ||
      logPrediction > std::log(std::numeric_limits<double>::max()) ||
      logPrediction < std::log(std::numeric_limits<double>::min()))
    return false;

  predictedCost = std::exp(logPrediction);
  if (!validPositive(predictedCost))
    return false;

  if (predictionSource != nullptr)
  {
    if (sourceLearned && destinationLearned)
      *predictionSource = GpuPredictionSource::Calibrated;
    else if (sourceLearned || destinationLearned)
      *predictionSource = GpuPredictionSource::Mixed;
    else
      *predictionSource = GpuPredictionSource::PriorOnly;
  }
  return true;
}

void GpuScalingModel::clear()
{
  gpuTypes_.clear();
  entries_.clear();
  referenceType_ = 0;
  hasReference_ = false;
}

std::size_t GpuPredictionAccuracy::indexOf(GpuPredictionSource source)
{
  const std::size_t index = static_cast<std::size_t>(source);
  return index < 4 ? index : 0;
}

const GpuPredictionAccuracy::Bucket& GpuPredictionAccuracy::bucketFor(
    GpuPredictionSource source) const
{
  return buckets_[indexOf(source)];
}

bool GpuPredictionAccuracy::observe(double predicted, double actual,
                                    GpuPredictionSource source)
{
  // A zero actual cost has no meaningful percentage error, and a negative or
  // non-finite one is malformed either way.
  if (!std::isfinite(predicted) || predicted < 0.0 || !validPositive(actual))
    return false;

  const double error = std::fabs(predicted - actual) / actual;
  if (!std::isfinite(error))
    return false;

  Bucket& bucket = buckets_[indexOf(source)];
  bucket.samples++;
  bucket.sumAbsolutePercentageError += error;
  if (bucket.retained.size() < retainedPerSource())
    bucket.retained.push_back(error);
  else
    bucket.truncated++;
  return true;
}

uint64_t GpuPredictionAccuracy::samples(GpuPredictionSource source) const
{
  return bucketFor(source).samples;
}

uint64_t GpuPredictionAccuracy::truncated(GpuPredictionSource source) const
{
  return bucketFor(source).truncated;
}

double GpuPredictionAccuracy::meanAbsolutePercentageError(
    GpuPredictionSource source) const
{
  const Bucket& bucket = bucketFor(source);
  if (bucket.samples == 0)
    return 0.0;
  return bucket.sumAbsolutePercentageError / static_cast<double>(bucket.samples);
}

bool GpuPredictionAccuracy::quantileAbsolutePercentageError(
    GpuPredictionSource source, double quantile, double& value) const
{
  const Bucket& bucket = bucketFor(source);
  if (bucket.retained.empty() || !std::isfinite(quantile) || quantile < 0.0 ||
      quantile > 1.0)
    return false;

  std::vector<double> sorted = bucket.retained;
  std::sort(sorted.begin(), sorted.end());
  // Nearest-rank: for a diagnostic this is easier to reason about than an
  // interpolating definition, and it always returns a value that was measured.
  std::size_t rank = static_cast<std::size_t>(quantile * (sorted.size() - 1) + 0.5);
  if (rank >= sorted.size()) rank = sorted.size() - 1;
  value = sorted[rank];
  return true;
}

void GpuPredictionAccuracy::clear()
{
  for (Bucket& bucket : buckets_)
    bucket = Bucket();
}

void GpuScalingModel::pup(PUP::er& p)
{
  p | gpuTypes_;
  p | entries_;
  p | referenceType_;
  p | hasReference_;
  p | alphaMin_;
  p | minSamples_;
}
