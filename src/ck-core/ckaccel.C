#include <stdio.h>

#include "charm++.h"
#include "ckaccel.h"
#include "ckaccel.decl.h"

#if CMK_CUDA
  extern "C" void cleanup_registration(); // NOTE: Generated code for CUDA registration cleanup
#endif


//#define ACCELD 1                       // DEBUG FLAG
#define MALLOC_WITH_NML__NUM_EXTRA_INTS  (4)
#define MALLOC_WITH_NML__PATTERN         (0xDEADCAFE)

#include <set>
std::set<void*> *mallocPtrs = NULL;
std::set<void*> *freePtrs = NULL;

/* Simply mallocs memory and appends it to set of mallocPtrs. It returns the address of 5th index and puts some pattern in 1-4 and last 4 indexes. */

void* mallocWithNML(int size) {

  if (mallocPtrs == NULL) { mallocPtrs = new std::set<void*>(); }
  if (freePtrs == NULL) { freePtrs = new std::set<void*>(); }

  // Round up the size to sizeof(int)
  if (size % sizeof(int) != 0) { size += sizeof(int) - (size % sizeof(int)); }

  // Allocate the memory with some extra words
  int *intPtr = (int*)(malloc(size + (sizeof(int) * (2 * MALLOC_WITH_NML__NUM_EXTRA_INTS + 1))));
  if (intPtr == NULL) { return NULL; }

  // Fill in the extra words (start and end) with bit patterns
  int size_int = (size / sizeof(int)) + (2 * MALLOC_WITH_NML__NUM_EXTRA_INTS) + 1;
  for (int i = 1; i <= MALLOC_WITH_NML__NUM_EXTRA_INTS; i++) {
    intPtr[i] = MALLOC_WITH_NML__PATTERN;
    intPtr[size_int - i] = MALLOC_WITH_NML__PATTERN;
  }
  intPtr[0] = size;

  // Calculate the pointer value to be returned
  void *rtn = (void*)(intPtr + MALLOC_WITH_NML__NUM_EXTRA_INTS + 1);

  // Add the pointer into the malloc'ed set and remove it from the free'd set
  mallocPtrs->insert(rtn);
  freePtrs->erase(rtn);

  // Return a pointer first the start NML
  return rtn;
}

/* This function grabs the size and checks for change in pattern which was set earlier during malloc. */

void checkWithNML(void *ptr) {

  // Check if ptr is in the free set (accessing an already free'd buffer)
  if (freePtrs->end() != freePtrs->find(ptr)) { printf("[ERROR] :: ACCESSING AN ALREADY FREE'D BUFFER DETECTED IN checkWithNML !!\n"); }
  if (mallocPtrs->end() == mallocPtrs->find(ptr)) { printf("[ERROR] :: ACCESSING A POINTER NOT ALLOCATED WITH mallocWithNML DETECTED IN checkWithNML !!\n"); }

  // Move the pointer back the start the of the NML and grab the allocated size
  int *intPtr = ((int*)ptr) - MALLOC_WITH_NML__NUM_EXTRA_INTS - 1;
  int size = intPtr[0] + (sizeof(int) * (2 * MALLOC_WITH_NML__NUM_EXTRA_INTS + 1));

  // Verify the pattern
  int size_int = size / sizeof(int);
  for (int i = 1; i <= MALLOC_WITH_NML__NUM_EXTRA_INTS; i++) {
    if (intPtr[i] != MALLOC_WITH_NML__PATTERN) { printf("[ERROR] :: MEMORY CORRUPTION DETECTED IN checkWithNML !!! (header)\n"); fflush(NULL); }
    if (intPtr[size_int - i] != MALLOC_WITH_NML__PATTERN) { printf("[ERROR] :: MEMORY CORRUPTION DETECTED IN checkWithNML !!! (footer)\n"); fflush(NULL); }
  }
}

void freeWithNML(void *ptr) {

  // Check the pointer
  checkWithNML(ptr);

  // Remove the pointer from the malloc'ed set and put it in the free'd set
  mallocPtrs->erase(ptr);
  freePtrs->insert(ptr);

  // Free the memory
  int *intPtr = ((int*)ptr) - MALLOC_WITH_NML__NUM_EXTRA_INTS - 1;
  free(intPtr);
}


#if CMK_ACCEL_SMP != 0
  AccelManager** AccelManager::accelManager = NULL;
#else
  AccelManager* AccelManager::accelManager = NULL;
#endif
/* readonly */ CProxy_AccelManagerGroup accelManagerGroupProxy;

extern "C" void initAccelManager() {

  // Create the AccelManager object itself
  AccelManager::getAccelManager();

  // Create the AccelManagerGroup, used to communicate
  if (CkMyPe() == 0) {
    accelManagerGroupProxy = CProxy_AccelManagerGroup::ckNew();
  }
}

extern "C" void exitAccelManager() {

  // Broadcast shutdown to all the PEs
  accelManagerGroupProxy.shutdown();
}


void markKernelStart() {
  AccelManager *accelMgr = AccelManager::getAccelManager();
  if (accelMgr != NULL) { accelMgr->markKernelStart(); }
}
void markKernelEnd() {
  AccelManager *accelMgr = AccelManager::getAccelManager();
  if (accelMgr != NULL) { accelMgr->markKernelEnd(); }
}


AccelError AccelStrategy_AllOnDevice::decide(int funcIndex,
                                             AEMRecord *record,
                                             AccelDecision &decision,
                                             void *objPtr,
                                             AccelManager* manager
                                            ) {

  // Verify the parameters
  if (record == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }
  if (manager == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }

  // Set the decision to run the AEM on the device if one is present
  // NOTE: If there are multiple accelerator types, just give some preference
  //   over the others for now

  // Try issuing to the GPU
  if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0) {

    decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
    decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
    if (record->isTriggered) {
      if ((record->checkInCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||
          (record->checkInCount >= record->numLocalElements)
         ) {
        decision.issueFlag = ACCEL_ISSUE_TRUE;
      } else {
        decision.issueFlag = ACCEL_ISSUE_FALSE;
      }
    } else {
      // Always false.  If the batch size reaches the maximum, then the generated code
      //   will force an issue.  Otherwise, wait for the timeout to occur.
      decision.issueFlag = ACCEL_ISSUE_FALSE;
    }

  } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {

    decision.deviceType = ACCEL_DEVICE_SPE;
    decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
    decision.issueFlag = ACCEL_ISSUE_TRUE;

  } else {

    decision.deviceType = ACCEL_DEVICE_HOST;
    decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
    decision.issueFlag = ACCEL_ISSUE_TRUE;
  }

  // Return success
  return ACCEL_SUCCESS;
}


AccelError AccelStrategy_AllOnHost::decide(int funcIndex,
                                           AEMRecord *record,
                                           AccelDecision &decision,
                                           void *objPtr,
                                           AccelManager* manager
                                          ) {

  // Run on the host
  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
  decision.issueFlag = ACCEL_ISSUE_TRUE;

  // Return success
  return ACCEL_SUCCESS;
}

void AccelStrategy_PercentDevice::resetNewSampleCount() { newSampleCount = -2; }
void AccelStrategy_PercentDevice::incrementNewSampleCount(AccelManager *manager) {
  newSampleCount++;
  if (newSampleCount == 0) {
    manager->resetHostTime();
    manager->resetKernelTime();
  }
}


AccelError AccelStrategy_PercentDevice::decide(int funcIndex,
                                               AEMRecord *record,
                                               AccelDecision &decision,
                                               void *objPtr,
                                               AccelManager* manager
                                              ) {

  // Verify the parameters
  if (record == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }
  if (manager == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }

  // Assume the element should be executed on the host
  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
  decision.issueFlag = ACCEL_ISSUE_TRUE;

  AccelError err = ACCEL_SUCCESS;
  if (staticAssignFlag) {
    err = decideStatic(funcIndex, record, decision, objPtr, manager);
  } else {
    err = decideDynamic(funcIndex, record, decision, objPtr, manager);
  }
  return err;
}

AccelError AccelStrategy_PercentDevice::decideStatic(int funcIndex,
                                                     AEMRecord *record,
                                                     AccelDecision &decision,
                                                     void *objPtr,
                                                     AccelManager* manager
                                                    ) {

  // Grab the strategy variables for this funcIndex
  Variables *vars = (Variables*)(record->strategyVars);

  // Grab the object data
  AccelObjectData *objData = manager->getObjectData(objPtr);
  if (objData == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; } // Invalid object

  // If this is the start of another set...
  if (record->checkInCount == 1) {

    // Reset the counters
    vars->deviceCount = 0;
    vars->hostCount = 0;

    // If the percent device value has changed (or a forced assignment because of invalid deviceTarget), then reset the targets as well
    if (vars->percentDevice != percentDevice || vars->deviceTarget < 0) {
      vars->percentDevice = percentDevice;
      vars->deviceTarget = (int)(vars->percentDevice * record->numLocalElements);
      vars->hostTarget = record->numLocalElements - vars->deviceTarget;
      vars->assigningFlag = 1;
    }
  }

  // If we are currently assigning locations, check the objects location
  if (vars->assigningFlag) {

    // If the current object is already assigned to the device and the device target
    //   hasn't been reached, just keep it on the device
    if ((objData->location == ACCEL_OBJECT_DATA_LOCATION_DEVICE) && (vars->deviceCount < vars->deviceTarget)) {
      // NOTE: Do nothing, just leave the object where it is

    // Otherwise, if the device is short objects, migrate this object to the device
    } else if (vars->deviceCount < vars->deviceTarget) {
#if ACCELD
      if (objData->location != ACCEL_OBJECT_DATA_LOCATION_UNSET) {
        printf("[ACCEL-DEBUG] :: PE %d :: Moving element from HOST (%d) to DEVICE (static assignment change)... pre-add - d:%d(%d), h:%d(%d)\n",
               CkMyPe(),
               objData->location,
               vars->deviceCount, vars->deviceTarget,
               vars->hostCount, vars->hostCount
              );fflush(NULL);
      }
#endif
      manager->pushObjectData(objPtr);
      objData->location = ACCEL_OBJECT_DATA_LOCATION_DEVICE;

    // Otherwise, if the current object is already assigned to the host and the host
    //   target hasn't been reached, just keep it on the host
    } else if ((objData->location == ACCEL_OBJECT_DATA_LOCATION_HOST) && (vars->hostCount < vars->hostTarget)) {
      // NOTE: Do nothing, just leave the object where it is

    // Otherwise, place the object on the host
    } else {
#if ACCELD
      if (objData->location != ACCEL_OBJECT_DATA_LOCATION_UNSET) {
        printf("[ACCEL-DEBUG] :: PE %d :: Moving element from DEVICE (%d) to HOST (static assignment change)... pre-add - d:%d(%d), h:%d(%d)\n",
               CkMyPe(),
               objData->location,
               vars->deviceCount, vars->deviceTarget,
               vars->hostCount, vars->hostCount
              );
      }
#endif
      manager->pullObjectData(objPtr);
      objData->location = ACCEL_OBJECT_DATA_LOCATION_HOST;
    }

    // If this is the last object of the set, clear the assigningFlag
    if (record->checkInCount >= record->numLocalElements) {
      vars->assigningFlag = 0;

#if ACCELD
      printf("[ACCEL-DEBUG] :: PE %d :: percent device static assignment - "
             "fi:%d, device: %d(%d), host:%d(%d), numLocalElements:%d\n",
             CkMyPe(),
             record->funcIndex,
             vars->deviceCount, vars->deviceTarget,
             vars->hostCount, vars->hostTarget,
             record->numLocalElements
            );fflush(NULL);
#endif

    }
  } // end if (vars->assigningFlag)

  // Based on the object's location, set the decision
  // NOTE: The calling function (AccelStrategy_PercentDevice::decide()) will default the
  //   decision to "issue on the host," so just redirect to device if required
  if (objData->location == ACCEL_OBJECT_DATA_LOCATION_DEVICE) {

    // Try issuing to the GPU
    if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0 && record->isTriggered) {
      vars->deviceCount++;
      decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      if ((vars->deviceCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||  // Max batch size... or
          (vars->deviceCount >= vars->deviceTarget)                // Last device element
         ) {
          decision.issueFlag = ACCEL_ISSUE_TRUE;
#if ACCELD
             traceUserEvent(37893);
#endif

      } else {
        decision.issueFlag = ACCEL_ISSUE_FALSE;
      }

    } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {
      vars->deviceCount++;
      decision.deviceType = ACCEL_DEVICE_SPE;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      decision.issueFlag = ACCEL_ISSUE_TRUE;
    }
  }

  // NOTE: At this point, if objData->location != ACCEL_OBJECT_DATA_LOCATION_DEVICE, then the
  //   current decision will direct it to the host core.

  // If the decision.deviceType wasn't changed from host, count it as running on the host
  if (decision.deviceType == ACCEL_DEVICE_HOST) { vars->hostCount++; }

  if (decision.deviceType == ACCEL_DEVICE_HOST) {  // Make host into host-delayed
    decision.deviceType = ACCEL_DEVICE_HOST_DELAY;
  }
  if (record->checkInCount >= record->numLocalElements) { // Issued delayed with last element
    decision.issueDelayedFlag = ACCEL_ISSUE_TRUE;
  } else {
    decision.issueDelayedFlag = ACCEL_ISSUE_FALSE;
  }
#if ACCELD
  if (record->checkInCount >= record->numLocalElements) {
      printf("[ACCEL-DEBUG] :: ------------ last element ---------- v->hc = %d (%d), v->dc = %d (%d), r->cic = %d, r->nle = %d\n",
             vars->hostCount, vars->hostTarget, vars->deviceCount, vars->deviceTarget, record->checkInCount, record->numLocalElements
          );
  }

  if (record->checkInCount >= record->numLocalElements) {
    traceUserEvent(37892);
  }
#endif
    // Return success
  return ACCEL_SUCCESS;
}

AccelError AccelStrategy_PercentDevice::decideDynamic(int funcIndex,
                                                      AEMRecord *record,
                                                      AccelDecision &decision,
                                                      void *objPtr,
                                                      AccelManager* manager
                                                     ) {

  // Grab the strategy variables for this funcIndex
  Variables *vars = (Variables*)(record->strategyVars);

  // Calculate the number of elements that should directed to the device and
  //   direct this element to the device if is within that count
  // NOTE: For now, if multiple accelerators are present, give priority to some types
  int numDeviceElements = (int)(vars->percentDevice * record->numLocalElements);
  if (record->checkInCount <= numDeviceElements) {

    // Try issuing to the GPU
    if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0 && record->isTriggered) {

      decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      if ((record->checkInCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||
          (record->checkInCount >= numDeviceElements)
         ) {
        decision.issueFlag = ACCEL_ISSUE_TRUE;
      } else {
        decision.issueFlag = ACCEL_ISSUE_FALSE;
      }

    } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {

      decision.deviceType = ACCEL_DEVICE_SPE;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      decision.issueFlag = ACCEL_ISSUE_TRUE;
    }

  } // end if (record->checkInCount <= numDeviceElements)

  // Check to see if this was the last element to be issued on this host, and
  //   if so, update the percent device value incase it has changed
  if (record->checkInCount >= record->numLocalElements) {
    vars->percentDevice = percentDevice;
  }

  // Return success
  return ACCEL_SUCCESS;
}

void AccelStrategy_PercentDevice::notifyLBFinished() {

  AccelManager *manager = AccelManager::getAccelManager();
  if (manager != NULL) {

    int numFuncIndexes = manager->getNumFuncIndexes();
    for (int i = 0; i < numFuncIndexes; i++) {
      AEMRecord* record = manager->getFuncIndexRecord(i);
      if (record == NULL) { continue; }
      Variables *vars = (Variables*)(record->strategyVars);
      if (vars == NULL) { continue; }
      vars->deviceTarget = -1;
      vars->hostTarget = -1;
      vars->assigningFlag = 1;
    }

    manager->resetHostTime();
    manager->resetKernelTime();
  }
  resetNewSampleCount();
}


void AccelStrategy_AdjustBusy::notifyIdle(AccelManager *manager) {

  // Make sure there are enough samples
  if (getNewSampleCount() < ACCELMGR_PERFVALS_MIN_NEW_SAMPLE_COUNT) { return; }
  resetNewSampleCount(); // Reset the sample count

  // Grab the host and kernel busy times
  float hostBusyTime = (float)(manager->getHostBusyTime() * ACCEL_STRATEGY_ADJUST_BUSY_HOST_BIAS_FACTOR);
  float kernelBusyTime = (float)(manager->getKernelBusyTime());

  float hostOverlappedBusyTime = (float)(manager->getHostOverlappedBusyTime() * ACCEL_STRATEGY_ADJUST_BUSY_HOST_BIAS_FACTOR);

  // Remove the fraction of the base load that does not overlap with device busy time
  float baseLoad = (float)(manager->getBaseLoad());
  float baseLoadOverlapped = (float)(manager->getBaseLoadOverlapped());
  if (baseLoad > 0.0f && baseLoadOverlapped >= 0.0f) { hostBusyTime -= (baseLoad - baseLoadOverlapped); }
  if (hostBusyTime < 0.0f) { hostBusyTime = 0.0f; }

  // Calculate a step factor (ratio increases as difference between busy values increases)
  if (kernelBusyTime <= 0) { kernelBusyTime = hostBusyTime / 2.0; }  // NOTE: If there was no work on the device, arbitrarily set the kernelBusyTime to something that push work towards the accelerator device
  float stepFactor = hostBusyTime / kernelBusyTime;
  if (stepFactor < 1.0f) { stepFactor = 1.0f / stepFactor; }
  stepFactor -= 1.0f;
  stepFactor *= ACCEL_STRATEGY_ADJUST_BUSY_STEP_MULTIPLIER;
  stepFactor += 1.0f;

  // Set the amount which percent device will change by, without passing a maximum amount
  float percentDeviceStep = ACCEL_STRATEGY_ADJUST_BUSY_MIN_STEP * stepFactor;
  if (percentDeviceStep > ACCEL_STRATEGY_ADJUST_BUSY_MAX_STEP) {
    percentDeviceStep = ACCEL_STRATEGY_ADJUST_BUSY_MAX_STEP;
  }

#if ACCELD
  float hostIdleTime = (float)(manager->getHostIdleTime());
  float hostCallbackAdjustTime = (float)(manager->getHostCallbackAdjustTime());
  float kernelIdleTime = (float)(manager->getKernelIdleTime());
  int myPE = CkMyPe();
  double curTime = CmiWallTimer();
  printf("[ACCELMGR] :: PE %d :: BUSY VALUES @ time %lf sec - host: %f (%f) sec, kernel: %f sec, stepFactor: %f\n", myPE, curTime, hostBusyTime, baseLoad - baseLoadOverlapped, kernelBusyTime, stepFactor);
printf("[ACCELMGR] :: PE %d :: BUSY VALUES @ time %lf sec - host: %f (%f) sec, kernel: %f sec, stepFactor: %f\n", myPE, curTime, hostOverlappedBusyTime, hostBusyTime, kernelBusyTime, stepFactor);
  printf("[ACCELMGR] :: PE %d :: IDLE VALUES @ time %lf sec - host: %f sec, kernel: %f sec - hostCallbackAdjust: %f sec\n",myPE, curTime, hostIdleTime, kernelIdleTime, hostCallbackAdjustTime);
  fflush(NULL);
#endif

  // Adjust percent device as long as the kernel and host busy times aren't too close
  float newPercentDevice = getPercentDevice();
  if (kernelBusyTime > (hostBusyTime * ACCEL_STRATEGY_ADJUST_BUSY_MARGIN_FACTOR)) {
    newPercentDevice -= percentDeviceStep;
    if (newPercentDevice < 0.0f) { newPercentDevice = 0.0f; }
  } else if (hostBusyTime > (kernelBusyTime * ACCEL_STRATEGY_ADJUST_BUSY_MARGIN_FACTOR)) {
    newPercentDevice += percentDeviceStep;
    if (newPercentDevice > 1.0f) { newPercentDevice = 1.0f; }
  }
#if ACCELD
  printf("[ACCEL-DEBUG] :: oldPD:%f, newPD:%f\n", getPercentDevice(), newPercentDevice);
#endif
  setPercentDevice(newPercentDevice);
  

  // Reset the manager's performance variables
  manager->resetKernelTime();
  manager->resetHostTime();
}


void AccelStrategy_Step::notifyIdle(AccelManager *manager) {

  if (newSampleCount >= ACCELMGR_PERFVALS_MIN_NEW_SAMPLE_COUNT) {

    // Step the percent device value down by the defined step size
    float percentDevice = getPercentDevice();
    percentDevice -= stepSize;
    if (percentDevice < 0.0f) { percentDevice = 0.0f; }
    setPercentDevice(percentDevice);

    // Reset the manager's performance variables
    manager->resetKernelTime();
    manager->resetHostTime();

    // Ignore the first new sample (i.e. -1 instead of 0)
    newSampleCount = -1;
  }
}


void AccelStrategy_Sampling::initialize(int centralFlag, float percentDeviceBias) {

  isCentral = ((centralFlag) ? (1) : (0));
  this->percentDeviceBias = percentDeviceBias;

  // Allocate the sample data and zero out the sample counts
  sample = new double[(ACCELMGR_PERFVALS_RESOLUTION + 1) * ACCELMGR_PERFVALS_SAMPLE_SIZE];
  sampleAvg = new double[ACCELMGR_PERFVALS_RESOLUTION + 1];
  sampleCount = new int[ACCELMGR_PERFVALS_RESOLUTION + 1];
  memset(sampleCount, 0, sizeof(int) * (ACCELMGR_PERFVALS_RESOLUTION + 1));

  // Clear the pending percent device values (length = 0)
  const float resStep = 1.0f / ACCELMGR_PERFVALS_RESOLUTION;
  percentDevice_pendingLen = 0;
  percentDevice_pending[percentDevice_pendingLen++] = 1.0f - (5 * resStep);
  percentDevice_pending[percentDevice_pendingLen++] = 1.0f - (10 * resStep);
}

void AccelStrategy_Sampling::cleanup() {
  if (sample != NULL) { delete [] sample; sample = NULL; }
  if (sampleAvg != NULL) { delete [] sampleAvg; sampleAvg = NULL; }
  if (sampleCount != NULL) { delete [] sampleCount; sampleCount = NULL; }
}

void AccelStrategy_Sampling::notifyIdle(AccelManager *manager) {

  // If the sampling should be done at a centralized location (PE 0) then only
  //   process the idle notification on PE 0
  if (isCentral && CkMyPe() != 0) { return; }

  // Make sure there are enough samples
  if (getNewSampleCount() < ACCELMGR_PERFVALS_MIN_NEW_SAMPLE_COUNT) { return; }
  resetNewSampleCount(); // Reset the sample count

  // If there are no more pending percent device values, generate more
  if (percentDevice_pendingLen <= 0) {

    int prevAvgIndex = -1;
    float prevAvg = 0.0f;
    int minAvgIndex = -1;
    float minAvg = 0.0f;

    // Fillin the avg values
    for (int i = 0; i < ACCELMGR_PERFVALS_RESOLUTION + 1; i++) {

      // Check for a samples to create a legit value from
      if (sampleCount[i] > 0) {

	// Calculate the avg value from the actual samples
	double *sampleSet = sample + (i * ACCELMGR_PERFVALS_SAMPLE_SIZE);
        double sum = sampleSet[0];
        double min = sum;
        double max = sum;
        for (int j = 1; j < sampleCount[i]; j++) {
	  sum += sampleSet[j];
          if (sampleSet[j] < min) { min = sampleSet[j]; }
          if (sampleSet[j] > max) { max = sampleSet[j]; }
	}
        if (sampleCount[i] > 3) {
	  sampleAvg[i] = (sum - min - max) / (sampleCount[i] - 2);
	} else {
	  sampleAvg[i] = sum / sampleCount[i];
	}

        // With this new value added, fill in any missing values
        float fillinAvg = prevAvg;
        if (prevAvgIndex < 0 || sampleAvg[i] < fillinAvg) { fillinAvg = sampleAvg[i]; }
        for (int j = prevAvgIndex + 1; j < i; j++) { sampleAvg[j] = fillinAvg; }

        // Update the last legit avg
        prevAvg = sampleAvg[i];
        prevAvgIndex = i;

        // Track the global min
        if (minAvgIndex < 0 || sampleAvg[i] < minAvg) {
	  minAvgIndex = i;
          minAvg = sampleAvg[i];
	}

      // Otherwise, if this is the last value and not a legit value
      } else if (i == ACCELMGR_PERFVALS_RESOLUTION) {

	// Back fill with the last legit value
	for (int j = prevAvgIndex; j <= ACCELMGR_PERFVALS_RESOLUTION; j++) {
	  sampleAvg[j] = prevAvg;
        }
      }

    } // end for (i < ACCELMGR_PERFVALS_RESOLUTION)

    // Use the avg values and global avg min to generate some new
    //   pending percent device values

    float resStep = 1.0f / ACCELMGR_PERFVALS_RESOLUTION;

    #define ADD_INDEX_TO_PERCENTDEVICE_PENDING(index, bias) {            \
      float newPercent = (float)((index) * resStep) + bias;              \
      if (newPercent <   0.0f) { newPercent =   0.0f; }                  \
      if (newPercent > 100.0f) { newPercent = 100.0f; }                  \
      percentDevice_pending[percentDevice_pendingLen++] = newPercent;    \
    }

    // Try to detect a valley around the min value or near either end of the range
    int hiIndex = minAvgIndex + 15; if (hiIndex > ACCELMGR_PERFVALS_RESOLUTION) { hiIndex = ACCELMGR_PERFVALS_RESOLUTION; }
    int loIndex = minAvgIndex - 15; if (loIndex < 0) { loIndex = 0; }
    if ((sampleAvg[hiIndex] > minAvg && sampleAvg[loIndex] > minAvg) ||
        (minAvgIndex < 5 || minAvgIndex > ACCELMGR_PERFVALS_RESOLUTION - 5)
       ) {

      // Make small movements in the valley
      // NOTE: Since this case is the "in a valley" case, apply the specified percent device bias value
      if (minAvgIndex + 1 <= ACCELMGR_PERFVALS_RESOLUTION) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex + 1, percentDeviceBias); }
      ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex, percentDeviceBias);
      ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex, percentDeviceBias);
      ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex, percentDeviceBias);
      ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex, percentDeviceBias);
      ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex, percentDeviceBias);
      if (minAvgIndex - 1 >= 0) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex - 1, percentDeviceBias); }

    // Otherwise, test if the lower index is <= current while higher index is greater (move towards low)
    } else if (sampleAvg[loIndex] <= minAvg && sampleAvg[hiIndex] > minAvg) {

      // Test some values to the lower side
      if (minAvgIndex -  5 >= 0) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex -  5, 0.0f); }
      if (minAvgIndex - 10 >= 0) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex - 10, 0.0f); }
      if (minAvgIndex - 15 >= 0) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex - 15, 0.0f); }

    // Otherwise, test if the higher index is <= current while lower index is greater (move towards high)
    } else if (sampleAvg[hiIndex] <= minAvg && sampleAvg[loIndex] > minAvg) {

      // Test some values to the higher side
      if (minAvgIndex +  5 <= ACCELMGR_PERFVALS_RESOLUTION) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex +  5, 0.0f); }
      if (minAvgIndex + 10 <= ACCELMGR_PERFVALS_RESOLUTION) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex + 10, 0.0f); }
      if (minAvgIndex + 15 <= ACCELMGR_PERFVALS_RESOLUTION) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex + 15, 0.0f); }

    // Otherwise
    } else {

      // Test values to both sides
      if (minAvgIndex + 10 <= ACCELMGR_PERFVALS_RESOLUTION) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex + 10, 0.0f); }
      if (minAvgIndex +  5 <= ACCELMGR_PERFVALS_RESOLUTION) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex +  5, 0.0f); }
      ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex, 0.0f);
      if (minAvgIndex -  5 >= 0) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex -  5, 0.0f); }
      if (minAvgIndex - 10 >= 0) { ADD_INDEX_TO_PERCENTDEVICE_PENDING(minAvgIndex - 10, 0.0f); }
    }

    #undef ADD_INDEX_TO_PERCENTDEVICE_PENDING

  } // end if (percentDevice_pendingLen <= 0)


  // Grab the first pending percent device value
  if (isCentral) {
    accelManagerGroupProxy.accelStrategy_setPercentDevice(percentDevice_pending[0]);
  }
  setPercentDevice(percentDevice_pending[0]);
  for (int i = 1; i < percentDevice_pendingLen; i++) {
    percentDevice_pending[i - 1] = percentDevice_pending[i];
  }
  percentDevice_pendingLen--;
}

void AccelStrategy_Sampling::periodicSample(double value, AccelManager *manager) {

  // If the sampling should be done at a centralized location (PE 0) then only
  //   process the idle notification on PE 0
  if (isCentral && CkMyPe() != 0) { return; }

  // Calculate which sample set this will go into
  int setIndex = (int)(getPercentDevice() * ACCELMGR_PERFVALS_RESOLUTION);
  double *sampleSet = sample + (setIndex * ACCELMGR_PERFVALS_SAMPLE_SIZE);

  // Shift the current samples back one to make room fro the new value
  for (int i = ACCELMGR_PERFVALS_SAMPLE_SIZE - 2; i >= 0; i--) {
    sampleSet[i + 1] = sampleSet[i];
  }
  sampleSet[0] = value;
  sampleCount[setIndex]++;
  if (sampleCount[setIndex] > ACCELMGR_PERFVALS_SAMPLE_SIZE) {
    sampleCount[setIndex] = ACCELMGR_PERFVALS_SAMPLE_SIZE;
  }

  // Count the new sample
  incrementNewSampleCount(manager);

  // Create an artificial idle notification to this strategy object
  notifyIdle(manager);
}

void AccelStrategy_Sampling::resetData() {
  for (int i = 0; i < ACCELMGR_PERFVALS_RESOLUTION + 1; i++) {
    sampleCount[i] = 0;
  }
  percentDevice_pendingLen = 0;
  resetNewSampleCount();
}

void AccelStrategy_Sampling::notifyLBFinished() {

  // Notify the parent class also
  AccelStrategy_PercentDevice::notifyLBFinished();

  if (isCentralized() && CkMyPe() != 0) { return; }

  const float resStep = 1.0f / ACCELMGR_PERFVALS_RESOLUTION;

}

void AccelStrategy_Profiler::controlActiveFuncIndex(AccelManager *manager, AEMRecord *record) {

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: controlActiveFuncIndex() - Called...\n", CkMyPe());fflush(NULL);
#endif

  int numFuncIndexes = manager->getNumFuncIndexes();

  if (numRotations > 0) {

    // Set the state to 'waiting for a restart' and 'active function index is invalid'
    waitingForRestart = 1;
    activeFuncIndex = -1;

  // Otherwise, if we are holding back device calls until its a good time to restart
  //   timing, then go ahead and restart the timing
  } else if (waitingForRestart) {

    // Clear the timing info in the manager
    manager->resetHostTime();
    manager->resetKernelTime();

    // Indicate that timing can start
    waitingForRestart = 0;

  // If this is the last step in the rotation set
 } else if (activeFuncIndex >= numFuncIndexes) {

    // Only let execution pass if there are enought host samples for each AEM type
    for (int i = 0; i < numFuncIndexes; i++) {
      AEMRecord *record = manager->getFuncIndexRecord(i);
      if (record == NULL) { return; }
      Variables *vars = (Variables*)(record->strategyVars);
      if (vars == NULL || vars->hostTimeCount < ACCEL_PROFILE_MIN_NUM_NEW_SAMPLES) { return; }
    }

    // End the rotation
    newSampleCount = 0;
    numRotations++;
    activeFuncIndex = -1;
    waitingForRestart = 1;

#if ACCELD
    printProfileData(manager);
#endif


  // Otherwise, this is a new sample, so count it
  } else {

    // Check to see if we have enough samples
    newSampleCount++;
    if (newSampleCount >= ACCEL_PROFILE_MIN_NUM_NEW_SAMPLES) {

      // Grab the timing info
      double kernelBusyTime = manager->getKernelBusyTime();
      Variables *vars = (Variables*)(record->strategyVars);
      if (vars != NULL) {
        vars->kernelTime += kernelBusyTime;
        vars->kernelTimeCount += (record->numLocalElements * newSampleCount);
      }
#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: Grabbed profile data for funcIndex %d...\n", CkMyPe(), record->funcIndex);
#endif

      // Reset the new sample count
      newSampleCount = 0;

      // Move on to the next function index
      activeFuncIndex++;
      waitingForRestart = 1;
    }
  }
}

AccelError AccelStrategy_Profiler::decide(int funcIndex,
                                          AEMRecord *record,
                                          AccelDecision &decision,
                                          void *objPtr,
                                          AccelManager *manager
                                         ) {

  // DMK - TODO - Add parameter checks here

  // Check to see if this is the first element on the node, and if so, attempt and update to activeFuncIndex
  if ((funcIndex == activeFuncIndex && record->checkInCount == 1) || (activeFuncIndex >= manager->getNumFuncIndexes())) {
    controlActiveFuncIndex(manager, record);
  }

  // Direct to the host
  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
  decision.issueFlag = ACCEL_ISSUE_TRUE;

  // If we are waiting for a restart, just stick to running on the host
  if (waitingForRestart) { return ACCEL_SUCCESS; }

  // If the incoming recquest is for the active function index, then direct the call
  //   to an available device.  Otherwise, leave it on the host.
  if (activeFuncIndex == funcIndex) {

    // Direct the work request to an available device
    // NOTE: For now, multiple accelerators on a single PE is not supported, so just give priority based on type

    // CUDA-Based GPU
    if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0) {

      decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      if ((record->checkInCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||
          (record->checkInCount >= record->numLocalElements)
         ) {
        decision.issueFlag = ACCEL_ISSUE_TRUE;
      } else {
        decision.issueFlag = ACCEL_ISSUE_FALSE;
      }

    // Cell SPE
    } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {

      decision.deviceType = ACCEL_DEVICE_SPE;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      decision.issueFlag = ACCEL_ISSUE_TRUE;
    }
  }

  return ACCEL_SUCCESS;
}

void AccelStrategy_Profiler::aboutToRemove(AccelManager *manager) {

  // Since this is a profiler, attempt to write the kernelToHostRatios out
  //   to the AEMRecords within the manager
  int numFuncIndexes = manager->getNumFuncIndexes();
  for (int i = 0; i < numFuncIndexes; i++) {

    // Get the record and variables
    AEMRecord *record = manager->getFuncIndexRecord(i);
    if (record == NULL) { continue; }
    Variables *vars = (Variables*)(record->strategyVars);
    if (vars == NULL) { continue; }

    // Calculate and set the kernelToHostRatio
    double kernelTimePerElement = vars->kernelTime / (double)(vars->kernelTimeCount);
    double hostTimePerElement = vars->hostTime / (double)(vars->hostTimeCount);
    record->hostToKernelRatio = hostTimePerElement / kernelTimePerElement;
    record->avgHostTime = hostTimePerElement;
  }
}

void AccelStrategy_Profiler::printProfileData(AccelManager *manager, FILE *fout) {

  // Loop through the function indexes and print the profile data for each
  int numFuncIndexes = manager->getNumFuncIndexes();
  for (int i = 0; i < numFuncIndexes; i++) {

    // Get the record and variables
    AEMRecord *record = manager->getFuncIndexRecord(i);
    if (record == NULL) { continue; }
    Variables *vars = (Variables*)(record->strategyVars);
    if (vars == NULL) { continue; }

    // Print the data
    fprintf(fout,
           "[ACCEL-STRATEGY] :: PE %d :: PROFILER :: funcIndex:%d - "
           "kernel-time/element: %lf sec (samples:%d), "
           "host-time/element %lf sec (samples:%d), device-speedup/element: %lf ...\n",
	   CkMyPe(), i,
           vars->kernelTime / (double)(vars->kernelTimeCount), vars->kernelTimeCount,
           vars->hostTime / (double)(vars->hostTimeCount), vars->hostTimeCount,
           (vars->hostTime / (double)(vars->hostTimeCount)) / (vars->kernelTime / (double)(vars->kernelTimeCount))
	  );
  }
}


void AccelStrategy_BaseLoadProfiler::takeSample(AccelManager *manager, AEMRecord *record) {

  // Grab this record's data
  if (record == NULL) { return; }
  Variables *vars = (Variables*)(record->strategyVars);
  if (vars == NULL) { return; }

  // If we are trying to stop sampling for this function index, then stop
  if (vars->pendingStop) {

    // If we haven't already stopped this function index, then stop it and count the stop
    if (vars->samplingStarted) {
      vars->samplingStarted = 0;
      numStopped++;
      if (numStopped >= manager->getNumFuncIndexes()) { allStopped = 1; }
    }
  // If we haven't started sampling for this function index, then start
  } else if (!(vars->samplingStarted)) {

    vars->samplingStarted = 1;

    // If this is the last function index to checkin ...
    numStarted++;
    if (numStarted >= manager->getNumFuncIndexes()) {

      // ... reset the time measurements and indicate that all have started
      manager->resetKernelTime();
      manager->resetHostTime();
      allStarted = 1;
    }

  } else if (allStarted) {

    vars->numSamples += 1;
    if (vars->numSamples == ACCEL_PROFILE_MIN_NUM_NEW_SAMPLES) {

      numFinished++;  // NOTE: Should only happen once per function index because of '==' test above

      // If this is the last function index to finish, grap the host's busy measurement
      if (numFinished >= manager->getNumFuncIndexes()) {


        // Read the base load (and overlapped load during this time period)
        baseLoad = manager->getHostBusyTime();
        baseLoadOverlapped = manager->getHostOverlappedBusyTime();

        // If there are periodic samples, normalize to one sample period
        if (periodicSampleCount > 0) {
          double hostIdleTime = manager->getHostIdleTime();
          double periodicSampleAvg = periodicSampleSum / periodicSampleCount;
          double baseLoadRatio = baseLoad / (baseLoad + hostIdleTime);
          double baseLoadOverlappedRatio = baseLoadOverlapped / (baseLoad + hostIdleTime);
          baseLoad = periodicSampleAvg * baseLoadRatio;
          baseLoadOverlapped = periodicSampleAvg * baseLoadOverlappedRatio;
	}

#if ACCELD
	printf("[ACCEL-DEBUG] :: PE %d :: BaseLoadProfiler :: baseLoad = %lf ...\n", CkMyPe(), baseLoad);fflush(NULL);
#endif

        // Set pending stop for all the AEM types
        int numFuncIndexes = manager->getNumFuncIndexes();
        for (int i = 0; i < numFuncIndexes; i++) {

          // Get the record and variables
          AEMRecord *record = manager->getFuncIndexRecord(i);
          if (record == NULL) { continue; }
          Variables *vars = (Variables*)(record->strategyVars);
          if (vars == NULL) { continue; }

          vars->pendingStop = 1;
        }

      }
    }

  }
}

AccelError AccelStrategy_BaseLoadProfiler::decide(int funcIndex,
						  AEMRecord *record,
						  AccelDecision &decision,
                                                  void *objPtr,
						  AccelManager *manager
                                                 ) {

  // If this is the first element to check in for this function index, count the sample
  if (record->checkInCount == 1) {
    takeSample(manager, record);
  }

  // Direct to the host if we should be sampling for this function index ...

  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
  decision.issueFlag = ACCEL_ISSUE_TRUE;
  Variables *vars = (Variables*)(record->strategyVars);

  #if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: vars = %p...\n", CkMyPe(), vars); fflush(NULL);
  #endif

  if (!(vars->samplingStarted)) { return ACCEL_SUCCESS; }

  // ... but try to push the work to any available accelerator

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: manager = %p...\n", CkMyPe(), manager); fflush(NULL);

  printf("[ACCEL-DEBUG] :: PE %d :: record = %p...\n", CkMyPe(), record);   fflush(NULL);
#endif

  // CUDA-Based GPU
  if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0) {

    decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
    decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
    if ((record->checkInCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||
        (record->checkInCount >= record->numLocalElements)
       ) {
      decision.issueFlag = ACCEL_ISSUE_TRUE;
    } else {
      decision.issueFlag = ACCEL_ISSUE_FALSE;
    }

  // Cell SPE
  } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {

    decision.deviceType = ACCEL_DEVICE_SPE;
    decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
    decision.issueFlag = ACCEL_ISSUE_TRUE;
  }

  return ACCEL_SUCCESS;
}

void AccelStrategy_BaseLoadProfiler::periodicSample(double value, AccelManager *manager) {
  if (allStarted) {
    periodicSampleSum += value;
    periodicSampleCount++;
  }
}

void AccelStrategy_BaseLoadProfiler::aboutToRemove(AccelManager *manager) {
  manager->setBaseLoad(baseLoad);
  manager->setBaseLoadOverlapped(baseLoadOverlapped);
}


void AccelStrategy_Greedy::defaultPercents(AccelManager *manager) {

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d ::   greedy - defaulting\n", CkMyPe());
  fflush(NULL);
#endif 
  // Default all of the percent device values to "all on host"
  int numFuncIndexes = manager->getNumFuncIndexes();
  for (int i = 0; i < numFuncIndexes; i++) {
    AEMRecord *record = manager->getFuncIndexRecord(i);
    if (record == NULL) { continue; }
    Variables *vars = (Variables*)(record->strategyVars);
    if (vars == NULL) { continue; }
    vars->percentDevice_pending = 0.0f;
  }

  // Indicate that percents have been calculated
  percentsSet = 1;
  numPercentsToApply = numFuncIndexes;
}

void AccelStrategy_Greedy::calculatePercents(AccelManager *manager) {


  float kernelTime = 0.0f;
  float hostTime = 0.0f;

  int numFuncIndexes = manager->getNumFuncIndexes();

  // Verify that performance values exist, and if not, default to some known values
  for (int i = 0; i < numFuncIndexes; i++) {
    AEMRecord *record = manager->getFuncIndexRecord(i);
    if (record == NULL) { continue; }
    Variables *vars = (Variables*)(record->strategyVars);
    if (vars == NULL) { continue; }
  }

  // Initially place everything on the host
  for (int i = 0; i < numFuncIndexes; i++) {

    // Sum the host times
    AEMRecord *record = manager->getFuncIndexRecord(i);
    if (record == NULL) { continue; }
    Variables *vars = (Variables*)(record->strategyVars);
    if (vars == NULL) { continue; }

    // Skip any records with invalid performance data
    if (record->avgHostTime < 0.0f || record->hostToKernelRatio < 0.0f) { continue; }

    hostTime += (record->avgHostTime * record->numLocalElements);
    vars->percentDevice_pending = 0.0f;  // While we are at it, clear pending
  }

  // Add in the base load
  float baseLoad = (float)(manager->getBaseLoad());
  float baseLoadOverlapped = (float)(manager->getBaseLoadOverlapped());
  if (baseLoad > 0.0f && baseLoadOverlapped >= 0.0f) {
    hostTime += baseLoad;
    kernelTime += (baseLoad - baseLoadOverlapped);
  }

  // Subract work from the accelerator while the load is imbalanced, starting
  //   with the loads that are best suited for the accelerator and working
  //   backwards to the loads worst suited for the accelerator
  int *funcIndexVisited = new int[numFuncIndexes];
  memset(funcIndexVisited, 0, sizeof(int) * numFuncIndexes);
  int numVisited = 0;

  // While there are still more function indexes to consider
  while (numVisited < numFuncIndexes) {

    // Find the bested suited function index of those not visited yet
    int maxFuncIndex = -1;
    float maxHostToKernelRatio = -1.0f;
    AEMRecord *maxRecord = NULL;
    for (int i = 0; i < numFuncIndexes; i++) {

      // Skip if already visited
      if (funcIndexVisited[i]) { continue; }

      // Skip any function indexes that don't have records and variables
      AEMRecord *record = manager->getFuncIndexRecord(i);
      if (record == NULL) { numVisited++; funcIndexVisited[i] = 1; continue; }
      Variables *vars = (Variables*)(record->strategyVars);
      if (vars == NULL) { numVisited++; funcIndexVisited[i] = 1; continue; }

      // Skip any records with invalid performance data
      if (record->avgHostTime < 0.0f || record->hostToKernelRatio < 0.0f) {
	numVisited++; funcIndexVisited[i] = 1; continue;
      }

      if (maxFuncIndex < 0 || maxHostToKernelRatio < record->hostToKernelRatio) {
	maxFuncIndex = i;
	maxHostToKernelRatio = record->hostToKernelRatio;
        maxRecord = record;
      }
    }

    // If control flow ever reaches this point without maxRecord being set, then
    //   it indicates that there are no valid function indexes with valid data
    if (maxRecord == NULL) { defaultPercents(manager); return; }

    // Mark the max as visited and count another function index as visited
    funcIndexVisited[maxFuncIndex] = 1;
    numVisited++;

    // Calculate how much of this function index to shift to the device
    Variables *maxVars = (Variables*)(maxRecord->strategyVars);
    float funcIndexHostTotalTime = maxRecord->avgHostTime * maxRecord->numLocalElements;
    float funcIndexKernelTotalTime = funcIndexHostTotalTime / maxRecord->hostToKernelRatio;

    // If shifting everything would fit
    if (hostTime - funcIndexHostTotalTime > kernelTime + funcIndexKernelTotalTime) {

      // Then shift everything
      hostTime -= funcIndexHostTotalTime;
      kernelTime += funcIndexKernelTotalTime;
      maxVars->percentDevice_pending = 1.0f;

    // Otherwise, if only a partial shift will fit, then do a partial shift and quit
    } else {

      // Subtract this function index's time from the host time
      hostTime -= funcIndexHostTotalTime;

      float d = hostTime - kernelTime;
      float r = maxRecord->hostToKernelRatio;
      float X = funcIndexHostTotalTime;
      float p = ((r * d) + (r * X)) / (X + (r * X));
      //float p = (d + X) / ((X/r) + X);  // (2 divides)

      if (p > 1.0f) { p = 1.0f; printf("[ACCEL-ERROR] :: PE %d :: GREEDY - percent device calculated too high, max-ing at 1.0f\n", CkMyPe()); }
      if (p < 0.0f) { p = 0.0f; printf("[ACCEL-ERROR] :: PE %d :: GREEDY - percent device calculated too low, min-ing at 0.0f\n", CkMyPe()); }

      maxVars->percentDevice_pending = p;

      break; // Nothing else should fit, so stop shifting
    }
  }

#if ACCELD
  printf("[ACCEL-STRATEGY] :: PE %d :: GREED - hostTime: %lf sec, kernelTime: %lf sec\n",CkMyPe(), hostTime, kernelTime);
  for (int i = 0; i < numFuncIndexes; i++) {
    AEMRecord *record = manager->getFuncIndexRecord(i);
    Variables *vars = (Variables*)(record->strategyVars);
    printf("[ACCEL-STRATEGY] :: PE %d :: GREEDY - percentDevice_pending[%d] = %f\n",CkMyPe(), i, vars->percentDevice_pending);
  }
  printf("[ACCEL-DEBUG] :: PE %d ::   greedy - percent device pending values calculated\n", CkMyPe());
  fflush(NULL);
#endif

  delete [] funcIndexVisited;


  // Indicate that percents have been calculated
  percentsSet = 1;
  numPercentsToApply = numFuncIndexes;
}

AccelError AccelStrategy_Greedy::decide(int funcIndex,
                                        AEMRecord *record,
                                        AccelDecision &decision,
                                        void *objPtr,
                                        AccelManager *manager
                                       ) {

  // DMK - TODO - Add parameter checks here

  // Assume the element should be executed on the host
  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
  decision.issueFlag = ACCEL_ISSUE_TRUE;

  // Grab the strategy variables for this function index
  Variables *vars = (Variables*)(record->strategyVars);
  if (vars == NULL) { return ACCEL_SUCCESS; } // Default to all on host

  // Calculate the number of elements that should directed to the device and
  //   direct this element to the device if is within that count
  // NOTE: For now, if multiple accelerators are present, give priority to some types
  int numDeviceElements = (int)(vars->percentDevice * record->numLocalElements);
  if (record->checkInCount <= numDeviceElements) {

    // Try issuing to the GPU
    if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0 && record->isTriggered) {

      decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      if ((record->checkInCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||
          (record->checkInCount >= numDeviceElements)
         ) {
        decision.issueFlag = ACCEL_ISSUE_TRUE;
      } else {
        decision.issueFlag = ACCEL_ISSUE_FALSE;
      }

    } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {

      decision.deviceType = ACCEL_DEVICE_SPE;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      decision.issueFlag = ACCEL_ISSUE_TRUE;
    }

  } // end if (record->checkInCount <= numDeviceElements)

  // Check to see if this is the last element, and if so, update the percent device value
  if (record->checkInCount >= record->numLocalElements) {
    vars->percentDevice = vars->percentDevice_pending;
    if (percentsSet && !(vars->appliedFlag)) {
      numPercentsApplied++;
      vars->appliedFlag = 1;
    }
  }
}

float AccelStrategy_Greedy::getPassedPercentDevice(AEMRecord *record) {
  if (record == NULL) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: PE %d ::   greedy - record = NULL\n", CkMyPe());
    fflush(NULL);
#endif
    return 0.0f;
 }
  Variables *vars = (Variables*)(record->strategyVars);
  if (vars == NULL) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: PE %d ::   greedy - vars = NULL\n", CkMyPe());
    fflush(NULL);
#endif
    return 0.0f;
  }
  return vars->percentDevice; // Make sure to return the active value, not the pending
}


AccelError AccelStrategy_Crawler::decide(int funcIndex,
                                         AEMRecord *record,
                                         AccelDecision &decision,
                                         void *objPtr,
                                         AccelManager* manager
                                        ) {

  // Assume the element should be executed on the host
  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
  decision.issueFlag = ACCEL_ISSUE_TRUE;

  // Grab the strategy variables for this function index
  Variables *vars = (Variables*)(record->strategyVars);
  if (vars == NULL) { return ACCEL_SUCCESS; } // Default to all on host

  // Check to see if this is a safe place to adjust the percent device amount
  //   to reflect any pending udpates
  if (record->checkInCount == 1) {
    float p  = vars->percentDeviceBase + vars->percentDeviceOffset;
    if (p < 0.0f) { p = 0.0f; }
    if (p > 1.0f) { p = 1.0f; }

#if ACCELD
    if (p != vars->percentDevice) {
      printf("[ACCEL-DEBUG] :: PE %d ::   crawler - funcIndex:%d, %%device:%f\n",
             CkMyPe(), funcIndex, p
            );
    }
#endif

    vars->percentDevice = p;
  }

  // Calculate the number of elements that should directed to the device and
  //   direct this element to the device if is within that count
  // NOTE: For now, if multiple accelerators are present, give priority to some types
  int numDeviceElements = (int)(vars->percentDevice * record->numLocalElements);
  if (record->checkInCount <= numDeviceElements) {

    // Try issuing to the GPU
    if (manager->getDeviceCount(ACCEL_DEVICE_GPU_CUDA) > 0 && record->isTriggered) {

      decision.deviceType = ACCEL_DEVICE_GPU_CUDA;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      if ((record->checkInCount % ACCEL_AEMs_PER_GPU_KERNEL == 0) ||
          (record->checkInCount >= numDeviceElements)
         ) {
        decision.issueFlag = ACCEL_ISSUE_TRUE;
      } else {
        decision.issueFlag = ACCEL_ISSUE_FALSE;
      }

    } else if (manager->getDeviceCount(ACCEL_DEVICE_SPE) > 0) {

      decision.deviceType = ACCEL_DEVICE_SPE;
      decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;
      decision.issueFlag = ACCEL_ISSUE_TRUE;
    }

  } // end if (record->checkInCount <= numDeviceElements)
}

void AccelStrategy_Crawler::applyNewOffsets(AccelManager *manager,
                                            float numFuncIndexes,
                                            float *offset
                                           ) {

#if ACCELD
  printf("[ACCEL-STRATEGY-DEBUG] :: PE %d :: CRAWLER - applying new offsets = { 0:%f, ... }\n",CkMyPe(), offset[0]);
  fflush(NULL);
#endif
  for (int i = 0; i < numFuncIndexes; i++) {
    AEMRecord *record = manager->getFuncIndexRecord(i);
    if (record == NULL) { continue; }
    Variables *vars = (Variables*)(record->strategyVars);
    if (vars == NULL) { continue; }
    vars->percentDeviceOffset = offset[i];
  }
}

void AccelStrategy_Crawler_Master::initialize(AccelManager *manager) {

  numSamples = -2;
  sampleAccum = 0.0f;

  int numFuncIndexes = manager->getNumFuncIndexes();
  currentOffsets = new float[numFuncIndexes];
  accumOffsets = new float[numFuncIndexes];

  rotationLen = numFuncIndexes * 2 + 1;  // + and - for each func index and "here"
  rotationIndex = 0;
  sampleAvg = new float[rotationLen];

  stepSize = ACCEL_CRAWLER_INIT_PERCENT_DEVICE_STEP;
}

void AccelStrategy_Crawler_Master::cleanup() {
  if (currentOffsets != NULL) { delete [] currentOffsets; }
  if (accumOffsets != NULL) { delete [] accumOffsets; }
  if (sampleAvg != NULL) { delete [] sampleAvg; }
}

AccelError AccelStrategy_Crawler_Master::decide(int funcIndex,
                                                AEMRecord *record,
                                                AccelDecision &decision,
                                                void *objPtr,
                                                AccelManager* manager
                                               ) {
  // DMK - TODO - Add parameter checks here

  if (numSamples >= ACCEL_CRAWLER_MIN_NEW_SAMPLES) {

    int numFuncIndexes = manager->getNumFuncIndexes();

    // Record the sample average for this rotation index
    sampleAvg[rotationIndex] = sampleAccum / numSamples;

    // Advance the rotation index
    rotationIndex++;
    while (rotationIndex > 0 && rotationIndex < rotationLen) {

#if ACCELD
      printf("[ACCEL-STRATEGY-DEBUG] :: PE %d :: CRAWLER trying rotation index %d/%d...\n",CkMyPe(), rotationIndex, rotationLen);
      fflush(NULL);
#endif

      // If it is a value movement, keep it (break the while loop)
      float offset = accumOffsets[calcFuncIndex(rotationIndex)] + calcOffset(rotationIndex);
      Variables* vars = (Variables*)(record->strategyVars);
      float pendingPercentDevice = vars->percentDeviceBase + offset;
      if (pendingPercentDevice <= 100.0f && pendingPercentDevice >= 0.0f) { break; }
      sampleAvg[rotationIndex] = -1.0;  // Set to an invalid value so it will be ignored

      // Move to the next rotation index
      rotationIndex++;
    }

    if (rotationIndex >= rotationLen) {  // Was the last one in this rotation

      // Search through the sample averages for the lowest, and apply that
      //   "winner's" offset to the accum offsets
      int minIndex = -1;
      float minAvg = -1.0f;
      for (int i = 0; i < rotationLen; i++) {
        if (sampleAvg[i] < 0.0) { continue; }
	if (minIndex < 0 || minAvg > sampleAvg[i]) {
	  minIndex = i;
          minAvg = sampleAvg[i];
        }
      }

      // Treat an invalid result as "stay still"
      if (minIndex < 0 || minIndex >= rotationLen) { minIndex = 0; }

      // Add the winner's offset to the accum offset array
      accumOffsets[calcFuncIndex(minIndex)] += calcOffset(minIndex);

#if ACCELD
      printf("[ACCEL-STRATEGY-DEBUG] :: PE %d :: CRAWLER - minIndex: %d, minAvg: %f, stepSize: %f, accum offsets = { 0:%f, 1:%f, 2:%f, ... }\n",CkMyPe(), minIndex, minAvg, stepSize, accumOffsets[0], accumOffsets[1], accumOffsets[2]);
      fflush(NULL);
#endif

      // Reset the rotation index
      rotationIndex = 0;
    }

#if ACCELD
    printf("[ACCEL-STRATEGY-DEBUG] :: PE %d :: CRAWLER @ rotation index %d...\n", CkMyPe(), rotationIndex);
    fflush(NULL);
#endif

    // Calculate the new "currentOffset" array (accum + adjustment for this rotation index)
    for (int i = 0; i < numFuncIndexes; i++) { currentOffsets[i] = accumOffsets[i]; }
    currentOffsets[calcFuncIndex(rotationIndex)] += calcOffset(rotationIndex);

    // DMK - Continue here ... broadcast out the content of currentOffsets to all
    //   of the strategy objects on all the cores (will only apply to crawler derivatives,
    //   but that should be alright)
    accelManagerGroupProxy.accelStrategy_crawler_setOffsets(numFuncIndexes, currentOffsets);

    // Start the next sample set
    sampleAccum = 0.0f;
    numSamples = -2;    // Ignore two (one sample period for offsets to transmit)
  }

  // Decide what to do with this AEM
  AccelStrategy_Crawler::decide(funcIndex, record, decision, objPtr, manager);
}

AccelPersistBuffer::AccelPersistBuffer(void *oPtr, void* hPtr, int s) {
  initialize(oPtr, hPtr, s);
}

AccelPersistBuffer::~AccelPersistBuffer() {
  cleanup();
}

#if CMK_CUDA

void AccelPersistBuffer::initialize(void *oPtr, void *hPtr, int s) {

  objPtr = oPtr;
  hostBufPtr = hPtr;
  deviceBufPtr = NULL;
  size = s;
  isDeviceDirty = 1;
  isHostDirty = 0;

  // Verify the parameters
  if (oPtr == NULL || hPtr == NULL || size <= 0) { return; }

  // Try to allocate a buffer on the device of size "size"
  //if (cudaSuccess != cudaMalloc(&deviceBufPtr, size)) {
  deviceBufPtr = newDeviceBuffer(size);  // NOTE: Set to NULL on failure
}

void AccelPersistBuffer::cleanup() {

  // Cleanup the device buffer
  deleteDeviceBuffer(deviceBufPtr);
  deviceBufPtr = NULL;
}

void AccelPersistBuffer::pushToDevice() {

  if (isDeviceDirty == 0) { return; }

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: AccelPersistBuffer::pushToDevice() - %p -> %p\n",CkMyPe(), hostBufPtr, deviceBufPtr);fflush(NULL);
#endif


  if (hostBufPtr == NULL || deviceBufPtr == NULL || size <= 0) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelPersistBuffer::pushToDevice() - Invalid persist state.\n", CkMyPe());
    return;
  }

  if (0 != ::pushToDevice(hostBufPtr, deviceBufPtr, size)) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelPersistBuffer::pushToDevice() - Copy failure.\n", CkMyPe());
    return;
  }

  isDeviceDirty = 0;
}

void AccelPersistBuffer::pullFromDevice() {

  if (isHostDirty == 0) { return; }

#if ACCELD
    printf("[ACCEL-DEBUG] :: PE %d :: AccelPersistBuffer::pullFromDevice() - %p -> %p\n",CkMyPe(), deviceBufPtr, hostBufPtr);
    fflush(NULL);
#endif

  if (hostBufPtr == NULL || deviceBufPtr == NULL || size <= 0) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelPersistBuffer::pullFromDevice() - Invalid persist state.\n", CkMyPe());
    return;
  }

  if (0 != ::pullFromDevice(hostBufPtr, deviceBufPtr, size)) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelPersistBuffer::pullFromDevice() - Copy failure.\n", CkMyPe());
    return;
  }

  isHostDirty = 0;
}

#else

void AccelPersistBuffer::initialize(void *oPtr, void *hPtr, int s) {
  objPtr = oPtr;
  hostBufPtr = hPtr;
  deviceBufPtr = NULL;
  size = s;
  isDeviceDirty = 1;
  isHostDirty = 0;  // NOTE: Data will remain on host, so host will never be dirty
}

void AccelPersistBuffer::cleanup() {
  objPtr = NULL;
  hostBufPtr = NULL;
  deviceBufPtr = NULL;
  size = -1;
  isDeviceDirty = 1;
  isHostDirty = 1;
}

void AccelPersistBuffer::pushToDevice() { }
void AccelPersistBuffer::pullFromDevice() { }

#endif


AccelManager::AccelManager() {
  initialize();
}

AccelManager::~AccelManager() {
  cleanup();
}

AccelManager* AccelManager::getAccelManager() {

  #if CMK_ACCEL_SMP != 0

    const int rank = CmiMyRank();
    if (accelManager[rank] == NULL) {
      accelManager[rank] = new AccelManager();
    }
    return accelManager[rank];

  #else

    if (accelManager == NULL) {
      accelManager = new AccelManager();
    }
    return accelManager;

  #endif
}

void AccelManager::destroyAccelManager() {

  #if CMK_ACCEL_SMP != 0

    const int rank = CmiMyRank();
    if (accelManager[rank] != NULL) { delete accelManager[rank]; accelManager[rank] = NULL; }

  #else

    if (accelManager != NULL) { delete accelManager; }
    accelManager = NULL;
    #if CMK_CUDA
      cleanup_registration(); // NOTE: Generated code for CUDA
    #endif

  #endif
}

void AccelManager::initialize() {

  // Default the flags
  flags = 0;

  // Start with an empty record list
  recordList = NULL;
  recordListLen = 0;

  // Assume that there are no devices for any device type
  for (int i = 0; i < ACCEL_NUM_DEVICE_TYPES; i++) { numDevices[i] = 0; }

  #if CMK_CELL
    numDevices[ACCEL_DEVICE_SPE] += 1;  // Just do one for now (present or not)
  #endif

  #if CMK_CUDA
    numDevices[ACCEL_DEVICE_GPU_CUDA] += 1;  // Just do one for now (present or not)
  #endif


  baseLoad = -1.0;
  baseLoadOverlapped = -1.0;

  // IDLE time measurement initialization
  idleTime[0] = idleTime[1] = idleTime[2] = idleTime[3] = idleTime[4] = 0.0;
  lastIdleNotifyTime = 0.0;
  lastIdleNotifyType = ACCELMGR_IDLE_NOTIFY_TYPE_UNKNOWN;
  lastIdleAdjustmentTime = 0.0;

  // Initialize the kernel performance measurement variables
  kernelTime[ACCELMGR_KERNEL_TIME_IDLE] = kernelTime[ACCELMGR_KERNEL_TIME_BUSY] = 0.0;
  lastKernelTime = CmiWallTimer();
  activeKernelCount = 0;

  callbackStructList = new std::list<AccelCallbackStruct*>();
  if (callbackStructList == NULL) {
    printf("[ACCEL-ERROR] :: AccelManager::initialize() - Unable to allocate memory for callback struct list.\n");
  }
  pendingCallbackStructList = new std::list<AccelCallbackStruct*>();
  if (pendingCallbackStructList == NULL) {
    printf("[ACCEL-ERROR] :: AccelManager::initialize() - Unable to allocate memory for pending callback struct list.\n");
  }

  takePeriodicSample_prevTime = -1.0;

  pendingStrategies = NULL;

  // Process the command line parameters accociated with the Accel Manager
  char **argv = CkGetArgv();
  CmiArgGroup("Charm++", "Accelerator");

  int hostOnlyFlag = CmiGetArgFlagDesc(argv, "+accelHostOnly", "Execute all AEMs on the host cores");
  int deviceOnlyFlag = CmiGetArgFlagDesc(argv, "+accelDeviceOnly", "Execute all AEMs on the device cores");
  char *gpuMapStr = NULL;
  char *strTokRTmp = NULL;
  if (0 != CmiGetArgStringDesc(argv, "+accelHostOnlyMap", &gpuMapStr, "Indicate which PEs should make use of the GPUs")) {
    char *hostOnlyArray = new char[strlen(gpuMapStr) + 1];
    char *tok = strtok_r(gpuMapStr, ",", &strTokRTmp);
    int i = 0;
    while (tok != NULL) {
      hostOnlyArray[i] = ((atoi(tok) == 0) ? (0) : (1));
      tok = strtok_r(NULL, ",", &strTokRTmp);
      i++;
    }
    if (hostOnlyArray[CkMyPe() % i] != 0) { hostOnlyFlag = 1; }
    delete [] hostOnlyArray;
  }

  double percentDevice = -1.0;
  int percentDevicePresent = CmiGetArgDoubleDesc(argv, "+accelPercentDevice", &percentDevice, "Use a fixed percent device when load balancing between hosts and accelerators");
  if (percentDevicePresent && (percentDevice < 0.0 || percentDevice > 1.0)) {
    printf("[ACCEL-ERROR] :: AccelManager::initialize() - Invalid percentDevice specified... ignoring.\n");
    percentDevicePresent = 0;
  }

  double stepSize = ACCEL_STRATEGY_ADJUST_STEP_STEPSIZE;
  int stepSizePresent = CmiGetArgDoubleDesc(argv, "+accelStep", &stepSize, "Step the percent device value by this constant");
  if (stepSizePresent && (stepSize < 0.0 || stepSize > 1.0)) {
    printf("[ACCEL-ERROR] :: AccelManager::initialize() - Invalid step specified... ignoring.\n");
    stepSizePresent = 0;
  }

  int adjustBusyFlag = CmiGetArgFlagDesc(argv, "+accelAdjustBusy", "Balance the busy times for the host and accelerator");
  int adjustBusyBaseFlag = 0;
  if (0 != CmiGetArgFlagDesc(argv, "+accelAdjustBusyBase", "Balance the busy times for the host and accelerator, after applying the Base Load Profiler")) {
    adjustBusyFlag = 1;
    adjustBusyBaseFlag = 1;
  }

  int samplingFlag = CmiGetArgFlagDesc(argv, "+accelSampling", "Use sampling to measure performance and adjust percent device");
  int samplingCentralizedFlag = 0;
  if (CmiGetArgFlagDesc(argv, "+accelSamplingCentralized", "Use centralized sampling to measure performance and adjust percent device"))  {
    samplingFlag = 1;
    samplingCentralizedFlag = 1;
  }

  int profilerFlag = CmiGetArgFlagDesc(argv, "+accelProfiler", "Use the profiler to measure kernel and host timings of the AEMs");
  int baseLoadProfilerFlag = CmiGetArgFlagDesc(argv, "+accelBaseLoadProfiler", "Use the base load profiler to measure non-AEM workload on the host");

  int greedyFlag = CmiGetArgFlagDesc(argv, "+accelGreedy", "Profile the base load, profile the kernels, and then apply a greedy strategy");
  int greedyCrawlerFlag = CmiGetArgFlagDesc(argv, "+accelGreedyCrawler", "Profile the base load, profile the kernels, apply a greedy strategy, and then start a crawler");
  if (greedyCrawlerFlag) { greedyFlag = 1; }

  if (CmiGetArgFlagDesc(argv, "+accelAutoMeasure", "Using this flag will cause the AccelManager to automate time sampling")) {
    flags |= ACCELMGR_FLAGMASK_AUTO_MEASURE_PERF;
  }

  float percentDeviceBias = 0.0f;
  double percentDeviceBias_raw = 0.0;
  if (CmiGetArgDoubleDesc(argv, "+accelBias", &percentDeviceBias_raw, "A bias amount for percent device values (supported by Sampling; max 0.2; positive is towards device)")) {
    if (percentDeviceBias_raw >= -0.2 && percentDeviceBias_raw <= 2.0) {
      percentDeviceBias = (float)(percentDeviceBias_raw);
    }
  }

  int staticAssignFlag = CmiGetArgFlagDesc(argv, "+accelStaticAssign", "Statically assign objects (Note: strategy must support this mode to have an effect)");

  // Create the strategy
  if (hostOnlyFlag) {
    strategy = new AccelStrategy_AllOnHost();
  } else if (deviceOnlyFlag) {
    strategy = new AccelStrategy_AllOnDevice();
  } else if (percentDevicePresent) {
    strategy = new AccelStrategy_PercentDevice(percentDevice, staticAssignFlag);
  } else if (stepSizePresent) {
    strategy = new AccelStrategy_Step(stepSize);
    ((AccelStrategy_PercentDevice*)strategy)->setStaticMode(staticAssignFlag);
  } else if (adjustBusyFlag) {
    if (adjustBusyBaseFlag) {
      strategy = new AccelStrategy_BaseLoadProfiler();
      pendingStrategies = new std::list<AccelStrategy*>();
      AccelStrategy_AdjustBusy *adjustBusyStrategy = new AccelStrategy_AdjustBusy();
      adjustBusyStrategy->setStaticMode(staticAssignFlag);
      pendingStrategies->push_back(adjustBusyStrategy);
    } else {
      strategy = new AccelStrategy_AdjustBusy();
      ((AccelStrategy_PercentDevice*)strategy)->setStaticMode(staticAssignFlag);
    }
  } else if (samplingFlag) {
    strategy = new AccelStrategy_Sampling(samplingCentralizedFlag, percentDeviceBias);
    ((AccelStrategy_PercentDevice*)strategy)->setStaticMode(staticAssignFlag);
  } else if (profilerFlag) {
    strategy = new AccelStrategy_Profiler();
  } else if (baseLoadProfilerFlag) {
    strategy = new AccelStrategy_BaseLoadProfiler();
  } else if (greedyFlag) {
    strategy = new AccelStrategy_BaseLoadProfiler();
    pendingStrategies = new std::list<AccelStrategy*>();
    pendingStrategies->push_back(new AccelStrategy_Profiler());
    pendingStrategies->push_back(new AccelStrategy_Greedy());
    if (greedyCrawlerFlag) {
      if (CkMyPe() == 0) {
	pendingStrategies->push_back(new AccelStrategy_Crawler_Master());
      } else {
	pendingStrategies->push_back(new AccelStrategy_Crawler_Slave());
      }
    }
  } else {
    CmiAbort("Invalid strategy selection");
  }

  // DMK - DEBUG
  if (strategy == NULL) {
#if ACCELD
    printf("[ACCEL-ERROR] :: AccelManager::initialize() - Unable to create strategy."); 
    fflush(NULL);
#endif
  }

  // Call the newly created strategy's aboutToAdd() function since it has just been created
  strategy->aboutToAdd(this);

  // Initialize the persistMap data structure
  persistMap = new std::map<void*,AccelPersistBuffer*>();
  objectMap = new std::map<void*,AccelObjectData*>();

  // Initialized the locMgrMap data structure
  locMgrMap = new std::map<void*,int>();

  // Initialize the delayed calls data structure
  delayCallDataStructs = new std::list<AccelDelayCallData*>();

  numAbandonedRequests = 0;
  abandonedPersistBuffers = new std::list<AccelPersistBuffer*>();

  // DMK - DEBUG
#if ACCELD
  strategy->printConfig();
#endif


  // Register for the BEGIN_IDLE and END_IDLE conditions with the
  //   runtime system
  conditionIndex[0] = CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, accelMgr_beginIdle_callback, (void*)this);
  conditionIndex[1] = CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE, accelMgr_endIdle_callback, (void*)this);
  conditionIndex[2] = CcdCallOnConditionKeep(CcdPERIODIC_1s, accelMgr_periodic_1s_callback, (void*)this);
  conditionIndex[3] = CcdCallOnConditionKeep(CcdPERIODIC, accelMgr_periodic_callback, (void*)this);
  conditionIndex[4] = CcdCallOnConditionKeep(CcdPERIODIC_5minute, accelMgr_periodic_5m_callback, (void*)this);

  // DMK - DEBUG - User events
#if ACCELD
  traceRegisterUserEvent("_kernel_issue_", 19482);
  traceRegisterUserEvent("LAST_TRGR", 37892);
  traceRegisterUserEvent("GPU_ISSUE", 37893);
  traceRegisterUserEvent("HD_ISSUE", 37894);
  traceRegisterUserEvent("HD_EXEC", 37895);
#endif
}

void AccelManager::cleanup() {

  // Cancel the conditions causing calls to notify_idle
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, conditionIndex[0]);
  CcdCancelCallOnConditionKeep(CcdPROCESSOR_END_IDLE, conditionIndex[1]);
  CcdCancelCallOnConditionKeep(CcdPERIODIC_1s, conditionIndex[2]);
  CcdCancelCallOnConditionKeep(CcdPERIODIC, conditionIndex[3]);
  CcdCancelCallOnConditionKeep(CcdPERIODIC_5minute, conditionIndex[4]);

  if (recordList != NULL) {
    for (int i = 0; i < recordListLen; i++) {
      if (recordList[i] != NULL) {
	//cleanupStrategy(recordList[i]);
        if (recordList[i]->delayedCalls != NULL) { delete recordList[i]->delayedCalls; }
        delete recordList[i];
      }
    }
    delete [] recordList;
    recordList = NULL;
  }
  recordListLen = 0;

  if (strategy != NULL) { delete strategy; strategy = NULL; }

  if (callbackStructList != NULL) {
    while (!(callbackStructList->empty())) {
      AccelCallbackStruct *t = callbackStructList->front();
      callbackStructList->pop_front();
      if (t->sharedLookup != NULL) { delete t->sharedLookup; }
      if (t->impl_objs != NULL) { delete [] t->impl_objs; }
      delete t;
    }
    delete callbackStructList;
    callbackStructList = NULL;
  }
  if (pendingCallbackStructList != NULL) {
    delete pendingCallbackStructList;
    pendingCallbackStructList = NULL;
  }

  // Cleanup the persist map data structure
  {
    // Current persist buffers
    std::map<void*,AccelPersistBuffer*>::iterator i;
    for (i = persistMap->begin(); i != persistMap->end(); i++) {
      if (i->second != NULL) {
        (i->second)->pullFromDevice();
        delete (i->second);
      }
    }
    delete persistMap;
    persistMap = NULL;

    // Pending abandoned persist buffers
    cleanAbandonedPersistData();
    delete abandonedPersistBuffers;
    abandonedPersistBuffers = NULL;
  }

  // Cleanup the object data structure
  {
    std::map<void*,AccelObjectData*>::iterator i;
    for (i = objectMap->begin(); i != objectMap->end(); i++) {
      if (i->second != NULL) {
        delete [] ((i->second)->persistHostPtr);
        delete (i->second);
      }
    }
    delete objectMap;
    objectMap = NULL;
  }

  // Cleanup the location manager map
  if (locMgrMap != NULL) { delete locMgrMap; locMgrMap = NULL; }

  // Cleanup the delayed call data structure
  if (delayCallDataStructs != NULL) {
    while (!(delayCallDataStructs->empty())) {
      AccelDelayCallData *dataPtr = delayCallDataStructs->front();
      delayCallDataStructs->pop_front();
      delete dataPtr;
    }
  }
}


void accelMgr_beginIdle_callback(void *arg, double time) {
  AccelManager *accelMgr = (AccelManager*)arg;
  if (accelMgr != NULL) { accelMgr->notify_idle(ACCELMGR_IDLE_NOTIFY_TYPE_BEGIN, time); }
}

void accelMgr_endIdle_callback(void *arg, double time) {
  AccelManager *accelMgr = (AccelManager*)arg;
  if (accelMgr != NULL) { accelMgr->notify_idle(ACCELMGR_IDLE_NOTIFY_TYPE_END, time); }
}

void accelMgr_periodic_1s_callback(void *arg, double time) {
  AccelManager *accelMgr = (AccelManager*)arg;
  if (accelMgr != NULL) { accelMgr->notify_idle(ACCELMGR_IDLE_NOTIFY_TYPE_PERIODIC, time); }
}

void accelMgr_periodic_callback(void *arg, double time) {
  AccelManager *accelMgr = (AccelManager*)arg;
  if (accelMgr != NULL) { accelMgr->notifyPeriodic(time); }
}

void accelMgr_periodic_5m_callback(void *arg, double time) {
  AccelManager *accelMgr = (AccelManager*)arg;
  if (accelMgr != NULL) { accelMgr->notifyPeriodic_5m(time); }
}

void AccelManager::notify_idle(int type, double time) {

    ///// Update the IDLE time measurements /////

  // Check for the UNKNOWN type (i.e. initializing)
  if (lastIdleNotifyType == ACCELMGR_IDLE_NOTIFY_TYPE_UNKNOWN) {

    idleTime[0] = idleTime[1] = idleTime[2] = idleTime[3] = idleTime[4] = 0.0;
    lastIdleAdjustmentTime = lastIdleNotifyTime = time;
    lastIdleNotifyType = type;

    // NOTE: If the first notification was PERIODIC, then assume that the
    // processor is busy (starts busy and hasn't gone IDLE yet).  Do this
    // since the lastIdleNotifyType most only toggle between BEGIN (0) and
    // END (1) when accumulating the time into idleTime[] below.
    if (lastIdleNotifyType == ACCELMGR_IDLE_NOTIFY_TYPE_PERIODIC) {
      lastIdleNotifyType = ACCELMGR_IDLE_NOTIFY_TYPE_END;
    }

  // Check for a periodic notification
  } else if (type == ACCELMGR_IDLE_NOTIFY_TYPE_PERIODIC) {

    double elapsedTime = time - lastIdleNotifyTime;
    int idleTimeIndex = 1 - lastIdleNotifyType;
    idleTime[idleTimeIndex] += elapsedTime;  // NOTE: if last was BEGIN, then add to idle time... if last was END, then add to busy time
    if (activeKernelCount > 0) { idleTime[3 + idleTimeIndex] += elapsedTime; }


    lastIdleNotifyTime = time;
    // NOTE: Leave lastIdleNotifyType untouched (time has been accounted for, so just pretend the last
    //   notification actually occured "now")

  // Check to see if the IDLE state has toggled
  } else if (lastIdleNotifyType != type) {

    double elapsedTime = time - lastIdleNotifyTime;
    idleTime[type] += elapsedTime; // NOTE: if type == BEGIN then index 0... if type == END then index = 1
    if (activeKernelCount > 0) { idleTime[3 + type] += elapsedTime; }


    lastIdleNotifyType = type;
    lastIdleNotifyTime = time;
  }

  // NOTE: The case lastIdleNotifyType == type should be ignored

  ///// Periodically make load balancing adjustments based on the IDLE time measurements /////


  if (lastIdleAdjustmentTime + ACCELMGR_IDLE_PERIOD <= time) {
    strategy->notifyIdle(this);
  }
}

void AccelManager::resetHostTime() {
  idleTime[0] = idleTime[1] = idleTime[2] = idleTime[3] = idleTime[4] = 0.0;
  lastIdleAdjustmentTime = CmiWallTimer();
}

double AccelManager::getHostBusyTime() { return idleTime[0]; } // + idleTime[2]; }
double AccelManager::getHostIdleTime() { return idleTime[1]; } // - idleTime[2]; }
double AccelManager::getHostCallbackAdjustTime() { return idleTime[2]; }
double AccelManager::getHostOverlappedBusyTime() { return idleTime[3]; }
double AccelManager::getHostOverlappedIdleTime() { return idleTime[4]; }

void AccelManager::adjustCallbackTime(double ammount) {
  // If the runtime is currently in an IDLE state, then accumulate this
  //   callback time as part of the callback adjustment time
  if (lastIdleNotifyType == ACCELMGR_IDLE_NOTIFY_TYPE_BEGIN) {
    idleTime[2] += ammount;
  }
}

AccelError AccelManager::userHostCodeTiming(int funcIndex, double startTime, double endTime) {

  // Verify the parameters
  if (funcIndex < 0) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: AccelManager::userHostCodeTiming() - invalid funcIndex (%d)...\n", funcIndex); 
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }
  if (startTime > endTime || startTime < 0 || endTime < 0) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: AccelManager::userHostCodeTiming() - invalid timing values (start:%lf, end:%lf)...\n",
           startTime, endTime
          );
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }

  // Grab the AEM record
  if (funcIndex >= recordListLen) {
    AccelError err = growRecordListToInclude(funcIndex);
    if (err != ACCEL_SUCCESS) { return err; }
  }

  // Initialize the record if needed
  if (recordList[funcIndex] == NULL) {
    AccelError err = initializeRecord(funcIndex);
    if (err != ACCEL_SUCCESS) { return err; }
  }

  // Get the record
  AEMRecord *rec = recordList[funcIndex];
  if (rec == NULL) { return ACCEL_ERROR_OUT_OF_MEMORY; }

  // Notify the strategy of the timing data
  strategy->userHostCodeTiming(rec, startTime, endTime);

  return ACCEL_SUCCESS;
}

void AccelManager::setPercentDevice(float pd) {

//  // Set the percent device value
  int strategyType = strategy->getStrategyType();
  switch (strategyType) {
    case ACCEL_STRATEGY_PERCENT_DEVICE:
    case ACCEL_STRATEGY_SAMPLING:
      AccelStrategy_PercentDevice *samplingStrategy = (AccelStrategy_PercentDevice*)strategy;
      samplingStrategy->setPercentDevice(pd);
      break;
  }
}

int AccelManager::numElementsLookup(CkLocMgr *locMgr) {

  std::map<void*,int>::iterator i;

  // Default the return value to an invalid value
  int numElements = -1;

  // Check to see if there already is a buffered count
  i = locMgrMap->find((void*)locMgr);
  if (i != locMgrMap->end()) {  // If so, return the buffered count
    numElements = i->second;
  } else { // If not, create a buffered count
    numElements = locMgr->numLocalElements();
    locMgrMap->insert(std::pair<void*,int>((void*)locMgr, numElements));
  }

#if ACCELD
    printf("[ACCEL-DEBUG] :: PE %d :: AccelManager->numElementsLookup(%p) = %d...\n", CkMyPe(), locMgr, numElements);
    fflush(NULL);
#endif

  // Return the number of elements
  return numElements;
}

AccelError AccelManager::delayGeneralCall(AccelDelayCallData *dataPtr) {
  if (dataPtr == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }
  AEMRecord *record = getFuncIndexRecord(dataPtr->funcIndex);
  if (record == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }
  (record->delayedCalls)->push_back(dataPtr);
  return ACCEL_SUCCESS;
}

void AccelManager::issueDelayedGeneralCalls(int funcIndex) {

  if (funcIndex < 0) { return; }
  AEMRecord *record = getFuncIndexRecord(funcIndex);
  if (record == NULL) { return; }

#if ACCELD
  CkPrintf("[ACCEL-DEBUG] ::PE %d :: AccelManager->issueDelayedGeneralCalls(funcIndex:%d) - %d delayed calls...\n", CkMyPe(), funcIndex, record->delayedCalls->size());

  double __hd_exec_start = CmiWallTimer();
#endif

  while (!((record->delayedCalls)->empty())) {
    AccelDelayCallData *dataPtr = (record->delayedCalls)->front();
    (record->delayedCalls)->pop_front();
    (dataPtr->funcPtr)(dataPtr);
  }
#if ACCELD
  double __hd_exec_end = CmiWallTimer();
  traceUserBracketEvent(37895, __hd_exec_start, __hd_exec_end);
#endif
}

AccelDelayCallData* AccelManager::allocAccelDelayCallData() {
  AccelDelayCallData *rtn = NULL;
  if (delayCallDataStructs->empty()) {
    rtn = new AccelDelayCallData;
  } else {
    rtn = delayCallDataStructs->front();
    delayCallDataStructs->pop_front();
  }
  return rtn;
}

void AccelManager::freeAccelDelayCallData(AccelDelayCallData* ptr) {
  if (ptr == NULL) { return; }
  delayCallDataStructs->push_back(ptr);
}

void AccelManager::takePeriodicSample(double value) {


  // If the value is less than zero, use time by default
  if (value < 0.0) {

    double now = CmiWallTimer();
    if (takePeriodicSample_prevTime < 0.0) {
      takePeriodicSample_prevTime = now;
      return;
    }
    strategy->periodicSample(now - takePeriodicSample_prevTime, this);
    takePeriodicSample_prevTime = now;

  } else {

    strategy->periodicSample(value, this);
  }
}

void AccelManager::notifyLBFinished() {

  // Clear the numElements (location manager) lookup
  locMgrMap->clear();

  // Notify the accelerator strategy that load balancing has just completed
  // NOTE: For centralized strategies, only PE 0 will be notified.
  if (strategy != NULL) {

    if (!(strategy->isCentralized()) || CkMyPe() == 0) {
      strategy->notifyLBFinished();
    }

  }

  takePeriodicSample_prevTime = -1.0;
}

void AccelManager::notifyFlushAccel() {

#if ACCELD
  printf("[ACCELMGR] :: PE %d :: AccelManager::notifyFlushAccel() - Flushing accelerator...\n", CkMyPe()); fflush(NULL);
#endif

  // Abandon the currently executing requests
  abandonPendingRequests();

  // Flush out object data
  removeAllObjectData();

  // Remove any persistent data buffers
  abandonPersistData();

  // Clear the numElements (location manager) lookup
  locMgrMap->clear();

  // Reset AEM records
  for (int i = 0; i < recordListLen; i++) {
    recordList[i]->checkInCount = 0;
    recordList[i]->numLocalElements = 0;
    if (recordList[i]->delayedCalls != NULL) {
      std::list<AccelDelayCallData*>::iterator j;
      for (j = (recordList[i]->delayedCalls)->begin(); j != (recordList[i]->delayedCalls)->end(); j++) {
        freeAccelDelayCallData(*j);
      }
      (recordList[i]->delayedCalls)->clear();
    }
  }

  takePeriodicSample_prevTime = -1.0;

#if ACCELD
  printf("[ACCELMGR] :: PE %d :: AccelManager::notifyFlushAccel() - Finished.\n", CkMyPe());
#endif
  fflush(NULL);
}

void AccelManager::notifyAbandonedRequestCompletion(AccelCallbackStruct *cbStruct) {

  // Note that there is one less outstanding abandoned request pending
  numAbandonedRequests--;

  // DMK - DEBUG
#if ACCELD
  printf("[ACCELMGR] :: PE %d :: AccelManager::notifyAbandonedRequestCompletion() - %d remaining requests...\n", CkMyPe(), numAbandonedRequests); 
#endif
  fflush(NULL);

  // Attempt to remove any abandoned persist data
  cleanAbandonedPersistData();
}

void AccelManager::markKernelStart() {
  double now = CmiWallTimer();
  if (activeKernelCount <= 0) {

    notify_idle(ACCELMGR_IDLE_NOTIFY_TYPE_PERIODIC, now);

#if ACCELD
    printf("<< first active kernel @ %lf -> %lf (%lf, %d) >>\n",
           lastKernelTime, now, now - lastKernelTime, activeKernelCount
          );
    fflush(NULL);
#endif

    kernelTime[ACCELMGR_KERNEL_TIME_IDLE] += now - lastKernelTime;
    lastKernelTime = now;
  }
  activeKernelCount++;
}

void AccelManager::markKernelEnd() {
  double now = CmiWallTimer();
  if (activeKernelCount == 1) { notify_idle(ACCELMGR_IDLE_NOTIFY_TYPE_PERIODIC, now); }
  activeKernelCount--;
  if (activeKernelCount <= 0) {
#if ACCELD
    printf("<<  last active kernel @ %lf -> %lf (%lf, %d) >>\n",
    lastKernelTime, now, now - lastKernelTime, activeKernelCount);
    fflush(NULL);
#endif

    activeKernelCount = 0; // If this func called more than start func
    kernelTime[ACCELMGR_KERNEL_TIME_BUSY] += now - lastKernelTime;
    lastKernelTime = now;
  }
}

void AccelManager::resetKernelTime() {
  lastKernelTime = CmiWallTimer();
  kernelTime[ACCELMGR_KERNEL_TIME_IDLE] = kernelTime[ACCELMGR_KERNEL_TIME_BUSY] = 0.0;
}

double AccelManager::getKernelBusyTime() {
  if (activeKernelCount > 0) {
    double now = CmiWallTimer();
    kernelTime[ACCELMGR_KERNEL_TIME_BUSY] += now - lastKernelTime;
    lastKernelTime = now;
  }
  return kernelTime[ACCELMGR_KERNEL_TIME_BUSY];
}

double AccelManager::getKernelIdleTime() {
  if (activeKernelCount <= 0) {
    double now = CmiWallTimer();
    kernelTime[ACCELMGR_KERNEL_TIME_IDLE] += now - lastKernelTime;
    lastKernelTime = now;
  }
  return kernelTime[ACCELMGR_KERNEL_TIME_IDLE];
}

AccelError AccelManager::growRecordListToInclude(int funcIndex) {

#if ACCELD
  printf("[ACCEL-DEBUG] :: AccelManager::growRecordListToInclude(funcIndex = %d) - Called...\n", funcIndex);
    fflush(NULL);
#endif

  // Verify the funcIndex parameter
  if (funcIndex < 0) {
#if ACCELD
    printf("[ACCEL-ERROR] :: AccelManager::growRecordListToInclude() - Invalid funcIndex...\n"); 
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }

  // Check to see if the length of the recordList needs to increase and
  // increase the length of the recordList if needed
  int reqLen = funcIndex + 1;
  if (reqLen > recordListLen) {
    AEMRecord** newRecordList = new AEMRecord*[reqLen];
    if (newRecordList == NULL) { return ACCEL_ERROR_OUT_OF_MEMORY; }
    int i = 0;
    if (recordList != NULL) {
      for (; i < recordListLen; i++) { newRecordList[i] = recordList[i]; }
      delete [] recordList;
    }
    for (; i < reqLen; i++) { newRecordList[i] = NULL; }
    recordList = newRecordList;
    recordListLen = reqLen;
  }

  return ACCEL_SUCCESS;
}

AccelError AccelManager::initializeRecord(int funcIndex) {

#if ACCELD
  printf("[ACCEL-DEBUG] :: AccelManager@%p::initializeRecord(funcIndex = %d) - Called...\n", this, funcIndex);
  fflush(NULL);
#endif

  AccelError err = ACCEL_SUCCESS;

  // Verify the parameter
  if (funcIndex < 0) {

#if ACCELD
    printf("[ACCEL-ERROR] :: AccelManager::initializeRecord() - Invalid funcIndex...\n");
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }

  // Grow the record list or needed
  if (funcIndex >= recordListLen) {
    err = growRecordListToInclude(funcIndex);
    if (err != ACCEL_SUCCESS) { return err; }
  }

#if ACCELD
  for (int i = 0; i < recordListLen; i++) {
    printf("[ACCEL-DEBUG] :: AccelManager::initializeRecord(...) -   recordList[%d] = %p...\n", i, recordList[i]);
    fflush(NULL);
  }
#endif

  // Check to see if the record has already been initialized
  if (recordList[funcIndex] != NULL) {
    return ACCEL_SUCCESS;  // Nothing wrong, just nothing needs done
  }

  // Create and initialize the actual data structure
  AEMRecord *rec = new AEMRecord;
  if (rec == NULL) { return ACCEL_ERROR_OUT_OF_MEMORY; }

  rec->funcIndex = funcIndex;
  rec->checkInCount = 0;
  rec->numLocalElements = 0;
  rec->isTriggered = 0;
  rec->isSplittable = 0;
  rec->strategyVars = strategy->allocStrategyVars();

  rec->hostToKernelRatio = -1.0f;  // Invalid / Not-Set Value
  rec->avgHostTime = -1.0f;

  rec->currentCbStruct = NULL;

  rec->delayedCalls = new std::list<AccelDelayCallData*>;
  if (rec->delayedCalls == NULL) { CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::initializeRecord() - Unable to allocate memory for delayed call list.\n", CkMyPe()); }

  recordList[funcIndex] = rec;

  return ACCEL_SUCCESS;
}

AccelError AccelManager::setDeviceCount(AccelDeviceType deviceType, int deviceCount) {

  // Verify the parameters
  if (deviceType < 0 || deviceType >= ACCEL_NUM_DEVICE_TYPES) { return ACCEL_ERROR_INVALID_PARAMETER; }
  if (deviceCount < 0) { return ACCEL_ERROR_INVALID_PARAMETER; }

  // Set the value
  numDevices[deviceType] = deviceCount;

  return ACCEL_SUCCESS;
}

int AccelManager::getDeviceCount(AccelDeviceType deviceType) {

  // Verify the parameters
  if (deviceType < 0 || deviceType >= ACCEL_NUM_DEVICE_TYPES) { return ACCEL_ERROR_INVALID_PARAMETER; }

  return numDevices[deviceType];
}



AccelError AccelManager::decide(int funcIndex,
                                int checkInCount, int numLocalElements,
                                int isTriggered, int isSplittable,
                                AccelDecision &decision,
                                void *objPtr
                               ) {

#if ACCELD
  printf("[ACCEL-DEBUG] :: AccelManager::decide(funcIndex = %d, checkInCount = %d, numLocalElements = %d, isTriggered = %d, isSplittable = %d, ...) - Called...\n",funcIndex, checkInCount, numLocalElements, isTriggered, isSplittable);
  fflush(NULL);
#endif


  // Regardless of any errors that are encountered, make sure the decision
  //   to be returned is a valid decision (default it to executing on the host)
  decision.issueFlag = ACCEL_ISSUE_TRUE;
  decision.issueDelayedFlag = ACCEL_ISSUE_TRUE;
  decision.deviceType = ACCEL_DEVICE_HOST;
  decision.deviceIndex = ACCEL_DEVICE_INDEX_UNKNOWN;


  // Verify the parameters
  if (funcIndex < 0) { 
#if ACCELD
    printf("[ACCEL-DEBUG] :: AccelManager::decide() - invalid funcIndex (%d)...\n", funcIndex);
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }
  if (numLocalElements < 1) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: AccelManager::decide() - invalid numLocalElements (%d)...\n", numLocalElements);
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }
  if (checkInCount < 0 || checkInCount > numLocalElements) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: AccelManager::decide() - invalid checkInCount (%d of %d)...\n", checkInCount, numLocalElements); 
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }

  // DMK - NOTE - For now, the strategies are really only written to handle one element checkin at a time, so make sure checkInCount is 1 (for now)
  if (checkInCount != 1) {
#if ACCELD
    printf("[ACCEL-DEBUG] :: AccelManager::decide() - invalid checkInCount (%d)...\n", checkInCount);
    fflush(NULL);
#endif
    return ACCEL_ERROR_INVALID_PARAMETER;
  }

  // Make sure there is enough room for the record in the recordList
  if (funcIndex >= recordListLen) {
    AccelError err = growRecordListToInclude(funcIndex);
    if (err != ACCEL_SUCCESS) { return err; }
  }

  // Initialize the record if needed
  if (recordList[funcIndex] == NULL) {
    AccelError err = initializeRecord(funcIndex);
    if (err != ACCEL_SUCCESS) { return err; }
  }

  // If there are pending strategies and the current strategy is safe to remove, then
  //   move on to the next strategy
  if (pendingStrategies != NULL && !(pendingStrategies->empty()) && strategy->isSafeToRemove()) {
#if ACCELD
    printf("[ACCELMGR] :: PE %d :: Swapping strategies @ %lf sec...\n", CkMyPe(), CmiWallTimer());
    fflush(NULL);
#endif

    AccelStrategy *nextStrategy = pendingStrategies->front();
    pendingStrategies->pop_front();

    // Check to see if the current strategy isPercentDevicePassable() and the
    //   next strategy acceptsPercentDevicePassable(), and if so, grab the percent
    //   device values so they can be passed to the next strategy
    float *passedPercentDeviceValue = NULL;
    if (strategy->isPercentDevicePassable()) {
      if (!(nextStrategy->acceptsPercentDevicePassable())) {
        printf("[ACCEL-ERROR] :: PE %d :: Invalid strategy sequence detected (passable to doesn't accept passable)...\n", CkMyPe());
      } else {
	passedPercentDeviceValue = new float[recordListLen];
        for (int i = 0; i < recordListLen; i++) {
	  passedPercentDeviceValue[i] = strategy->getPassedPercentDevice(recordList[i]);
	}
      }
    }

    // Indicate to the strategy that it is about to be removed from this manager
    strategy->aboutToRemove(this);

    // Free the stategy-specific variables
    for (int i = 0; i < recordListLen; i++) {
      if (recordList[i] != NULL) {
        strategy->freeStrategyVars(recordList[i]->strategyVars);
        recordList[i]->strategyVars = NULL;
      }
    }

    // Swap the strategies
    delete strategy;
    strategy = nextStrategy;

    // Create the strategy-specific variables
    for (int i = 0; i < recordListLen; i++) {
      if (recordList[i] != NULL) {
        recordList[i]->strategyVars = strategy->allocStrategyVars();
      }
    }

    // If there are passed percent device values, then pass them on
    if (passedPercentDeviceValue != NULL) {
      for (int i = 0; i < recordListLen; i++) {
	strategy->setPassedPercentDevice(recordList[i], passedPercentDeviceValue[i]);
      }
      delete [] passedPercentDeviceValue;  // Finished with this array
    }

    // Indicate to the strategy that it is about to be added to this manager
    strategy->aboutToAdd(this);

    // DMK - DEBUG
#if ACCELD
    strategy->printConfig();
#endif
  }

  // Get the record
  AEMRecord *rec = recordList[funcIndex];
  if (rec == NULL) { return ACCEL_ERROR_INVALID_PARAMETER; }

  //// DMK - DEBUG
#if ACCELD
  if (CkMyPe() == 0 && funcIndex == 0 && (rec->checkInCount == 0 || rec->checkInCount >= numLocalElements - 1)) {  
    printf("[DEBUG] :: PE %d :: decide -> checkInCount = %d / %d\n",CkMyPe(), rec->checkInCount, numLocalElements);
    fflush(NULL);
  }
#endif

  // Take a periodic measurement if this is the last local element
  if (getAutoMeasurePerf() && funcIndex == 0 && rec->checkInCount <= 0) {
    takePeriodicSample();
  }

  // Update the record with the incoming values
  rec->checkInCount += checkInCount;
  rec->numLocalElements = numLocalElements;
  rec->isTriggered = isTriggered;
  rec->isSplittable = isSplittable;

  ///// Time To Be The Decider /////

  AccelError strategyError = strategy->decide(funcIndex, rec, decision, objPtr, this);

    // If all local elements have checked in for a triggered AEM, restart the check in count
  if (rec->isTriggered && rec->checkInCount >= rec->numLocalElements) {
    rec->checkInCount = 0;
  }
#if ACCELD
  printf("[DMK-DEBUG] ::   checkInCount = %d/%d, deviceType = %d, issuelFag = %d...\n",rec->checkInCount, rec->numLocalElements, decision.deviceType, decision.issueFlag);
  printf("[ACCEL-DEBUG] :: AccelManager::decide(funcIndex = %d, ...) - Finished.\n", funcIndex);
  fflush(NULL);
#endif

  return strategyError;
}

AccelCallbackStruct* AccelManager::allocAccelCallbackStruct() {

  AccelCallbackStruct *rtn = NULL;

  // Check to see if the list is empty (if so, allocated a new callback struct)
  if (callbackStructList->empty()) {

    rtn = new AccelCallbackStruct;
    if (rtn == NULL) {
      printf("[ACCEL-ERROR] :: AccelManager::allocAccelCallbackStruct() - Unable to allocate memory for callback struct.\n");
      return NULL;
    }

    rtn->impl_objs = new void*[ACCEL_AEMs_PER_GPU_KERNEL];
    if (rtn->impl_objs == NULL) {
      delete rtn;
      printf("[ACCEL-ERROR] :: AccelManager::allocAccelCallbackStruct() - Unable to allocated memory for object pointer array.\n");
      return NULL;
    }

    rtn->wrData = NULL;

    rtn->sharedLookup = new AccelSharedLookup();
    if (rtn->sharedLookup == NULL) {
      delete [] (rtn->impl_objs);
      delete rtn;
      printf("[ACCEL-ERROR] :: AccelManager::allocAccelCallbackStruct() - Unable to allocated memory for shared lookup.\n");
      return NULL;
    }

  // Otherwise, grab an existing callback struct from the list and return that
  } else {

    rtn = callbackStructList->front();
    callbackStructList->pop_front();
  }

  // Reset the callback data structure
  resetAccelCallbackStruct(rtn);

  // Add this request to the pending request list
  if (rtn != NULL) {
    rtn->lastContribTime = CmiWallTimer();
    pendingCallbackStructList->push_front(rtn);
  }

  return rtn;
}

void AccelManager::resetAccelCallbackStruct(AccelCallbackStruct *cbStruct) {
  cbStruct->wr = NULL;
  cbStruct->di = NULL;
  (cbStruct->sharedLookup)->reset();
  cbStruct->contribStartTime = -1.0;
  cbStruct->issueTime = -1.0;
  cbStruct->callbackStartTime = -1.0;
  cbStruct->lastContribTime = -1.0;
  // DMK - TODO | FIXME - The macro check fails because it is not being set in the runtime code even
  //   though it is being set within the Hybrid API, so for now, assume pooling is enabled
  #if CMK_CUDA != 0 // CMK_CUDA_USE_GPU_MEMPOOL != 0
    if (cbStruct->wrData != NULL) { hapi_poolFree(cbStruct->wrData); }
  #else
    if (cbStruct->wrData != NULL) { delete [] (char*)(cbStruct->wrData); }
  #endif
  cbStruct->wrData = NULL;
  cbStruct->wrDataLen = 0;
  cbStruct->wrDataLen_max = 0;
  cbStruct->numElements = 0;
  cbStruct->numElements_count = 0;
  cbStruct->abandonFlag = 0;

  cbStruct->funcIndex = -1;
  cbStruct->callbackPtr = NULL;
  cbStruct->threadCount = -1;
}

void AccelManager::freeAccelCallbackStruct(AccelCallbackStruct *cbStruct) {
  pendingCallbackStructList->remove(cbStruct);
  resetAccelCallbackStruct(cbStruct);
  callbackStructList->push_back(cbStruct);
}

void AccelManager::updateLastContribTime(AccelCallbackStruct *cbStruct) {
  cbStruct->lastContribTime = CmiWallTimer();
}

void AccelManager::abandonPendingRequests() {
  #if CMK_CUDA != 0 // Only required (meaningful) for CUDA-based GPGPUs as they are (currently) the only ones that batch (i.e. use cbStructs)

  // DMK - DEBUG
#if ACCELD
  printf("[ACCELMGR] :: PE %d :: AccelManager::abandonPendingRequests()...\n", CkMyPe()); 
  fflush(NULL);
#endif

  // While there are pending requests, keep draining the queue
  while (!(pendingCallbackStructList->empty())) {

    // Remove the first request
    AccelCallbackStruct *cbStruct = pendingCallbackStructList->front();
    pendingCallbackStructList->pop_front();

    // Depending on the state of the request, either deallocate it or abandon it
    if (cbStruct->issueTime > 0) {

      // Has been issued, so abandon in callback by setting abandonFlag
      cbStruct->abandonFlag = 1;
      numAbandonedRequests++;
      // NOTE: The callback function will call freeAccelCallbackStruct(), even if
      //   abandonFlag is set, returning the data structure to the free list

    } else {

      // Has not been issued, so just destroy the request
      kernelCleanup(cbStruct->wr, cbStruct->di);
      // DMK - TODO | FIXME - The macro check fails because it is not being set in the runtime code even
      //   though it is being set within the Hybrid API, so for now, assume pooling is enabled
      hapi_poolFree(cbStruct->wrData);
      cbStruct->wrData = NULL;
      freeAccelCallbackStruct(cbStruct);
    }
  }
#if ACCELD
  printf("[ACCELMGR] :: PE %d :: AccelManager::abandonPendingRequests() - numAbandonedRequests = %d...\n", CkMyPe(), numAbandonedRequests); fflush(NULL);
  fflush(NULL);
#endif
  #endif
}

void AccelManager::submitPendingRequests() {
  #if CMK_CUDA != 0 // Only required (meaningful) for CUDA-based GPGPUs as they are (currently) the only ones that batch (i.e. use cbStructs)

  double now = CmiWallTimer();
  std::list<AccelCallbackStruct*>::iterator i;
  for (i = pendingCallbackStructList->begin(); i != pendingCallbackStructList->end(); i++) {

    // Grab the batch set's data structure and associated record
    AccelCallbackStruct *cbStruct = *i;
    AEMRecord *record = getFuncIndexRecord(cbStruct->funcIndex);
    if (cbStruct == NULL || record == NULL) {
      CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::submitPendingRequests() - "
               "Unable to retrive batch set data... (cbStruct:%p, record:%p, fi:%d)...\n",
               CkMyPe(), cbStruct, record, ((cbStruct == NULL) ? (-13) : (cbStruct->funcIndex))
              );
      fflush(NULL);
    }

    // Calcualte the timeout period used for this type of AEM and test if the timeout period has elapsed
    double timeoutPeriod = ((record->isTriggered != 0) ? (ACCEL_CONTRIB_TRIGGERED_TIMEOUT) : (ACCEL_CONTRIB_NONTRIGGERED_TIMEOUT));
    if (cbStruct->issueTime < 0.0 && cbStruct->lastContribTime + timeoutPeriod < now) {
#if ACCELD
      printf("[ACCEL-DEBUG] :: PE %d :: Kernel timeout occured... submitting...\n", CkMyPe());
      fflush(NULL);
#endif

      submitPendingRequest(cbStruct); // If the timeout period has elapsed, submit the batch set
    }

  } // end for (i != pendingCallbackStructList->end())

  #endif
}

void AccelManager::submitPendingRequest(AccelCallbackStruct *cbStruct) {
  #if CMK_CUDA != 0 // Only required (meaningful) for CUDA-based GPGPUs as they are (currently) the only ones that batch (i.e. use cbStructs)

  if (cbStruct == NULL) { return; }

  // Save the issue time
  cbStruct->issueTime = CmiWallTimer();

  // Calculate the actual thread count
  int threadCount = 0;
  int numSplitsFlag = ((int*)(cbStruct->wrData))[ACCEL_CUDA_KERNEL_NUM_SPLITS];
  if (numSplitsFlag < 0) {  // Unequal splits were detected
    for (int i = 0; i < cbStruct->numElements_count; i++) {
      threadCount += cbStruct->numSplitsSubArray[i];
    }
  } else { // All numSplits values are equal in this kernel set
    threadCount = cbStruct->numElements_count * numSplitsFlag; // All equal, so just multiply out to avoid loop
  }
  cbStruct->threadCount = threadCount;

  // Remove this request as the active request, if it is the active request
  AccelCallbackStruct *activeRequest = getCurrentCallbackStruct(cbStruct->funcIndex);
  if (cbStruct == activeRequest) {
    setCurrentCallbackStruct(cbStruct->funcIndex, NULL);
  }
  // Issue the request to the device
  kernelSetup(cbStruct->funcIndex,
              cbStruct->wrData,
              cbStruct->wrDataLen,
              cbStruct->callbackPtr,
              (void*)cbStruct,
              cbStruct->threadCount,
              &(cbStruct->wr),
              &(cbStruct->di)
	     );

  #endif
}

AccelCallbackStruct* AccelManager::getCurrentCallbackStruct(int funcIndex) {
  AEMRecord *record = getFuncIndexRecord(funcIndex);
  if (record == NULL) { return NULL; }
  return record->currentCbStruct;
}

void AccelManager::setCurrentCallbackStruct(int funcIndex, AccelCallbackStruct *cbStruct) {
  AEMRecord *record = getFuncIndexRecord(funcIndex);
  if (record == NULL) { return; }
  record->currentCbStruct = cbStruct;
}

void AccelManager::notifyPeriodic(double time) {
  submitPendingRequests();
}

void AccelManager::notifyPeriodic_5m(double time) {
  cleanStaleObjectData();
}

void AccelManager::cleanStaleObjectData() {

  // TODO | FIXME : Fill this in... any object data that has not been used in some
  //   time should be removed, along with the persistent data associated with the
  //   object so the device memory gets free'd periodically.
}


AccelPersistBuffer* AccelManager::newPersistBuffer(void *oPtr, void *hPtr, int size) {

  // TODO - Add more error checking in this method (if hPtr exists already, insert fails, etc.)

  if (oPtr == NULL || hPtr == NULL || size <= 0) { return NULL; }

  if (getPersistBuffer(hPtr) != NULL) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelManager::newPersistBuffer() - Attempt to create multiple persist buffers using same host pointer.\n", CkMyPe());
    return NULL;
  }

  // Create the persist buffer
  AccelPersistBuffer *persistBuf = new AccelPersistBuffer(oPtr, hPtr, size);
  if (persistBuf == NULL) { return NULL; }

  // Insert the new persist buffer into the persistMap and pust the data
  persistMap->insert(std::pair<void*,AccelPersistBuffer*>(hPtr, persistBuf));
  persistBuf->pushToDevice();

  // Register the host buffer with the object
  registerPersistHostBufferWithObject(oPtr, hPtr);

  // Return the new buffer to the caller
  return persistBuf;
}

AccelPersistBuffer* AccelManager::getPersistBuffer(void *hPtr) {
  if (hPtr == NULL) { return NULL; }
  std::map<void*,AccelPersistBuffer*>::iterator i;
  i = persistMap->find(hPtr);
  if (i != persistMap->end()) { return (i->second); }
  return NULL;
}

void AccelManager::pullPersistBuffer(void *hPtr) {
  AccelPersistBuffer* persistBuffer = getPersistBuffer(hPtr);
  if (persistBuffer == NULL) { return; }
  persistBuffer->pullFromDevice();
}

void AccelManager::pushPersistBuffer(void *hPtr) {
  AccelPersistBuffer* persistBuffer = getPersistBuffer(hPtr);
  if (persistBuffer == NULL) { return; }
  persistBuffer->pushToDevice();
}

void AccelManager::deletePersistBuffer(AccelPersistBuffer *persistBuf) {

  if (persistBuf == NULL) { return; }

  void *hostBufPtr = persistBuf->getHostBuffer();
  if (hostBufPtr == NULL) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelManager::deletePersistBuffer() - Persist buffer with NULL host pointer detected.\n", CkMyPe());
    return;
  }
  AccelPersistBuffer *lookupBuf = getPersistBuffer(hostBufPtr);
  if (lookupBuf != persistBuf) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelManager::deletePersistBuffer() - Persist buffer and host pointer mismatch detected.\n", CkMyPe()); fflush(NULL);
    return;
  }
  if (lookupBuf == NULL) {
    CkPrintf("[ACCEL-PERSIST-ERROR] :: PE %d :: AccelManager::deletePersistBuffer() - Attempt to delete a non-existent persist buffer.\n", CkMyPe()); fflush(NULL);
    return;
  }

  // Remove the persist buffer from the map and delete it
  lookupBuf->pullFromDevice();
  persistMap->erase(hostBufPtr);
  delete lookupBuf;
}

void AccelManager::abandonPersistData() {

  // Move all of the existing persist buffers to the abandoned list
  std::map<void*,AccelPersistBuffer*>::iterator i;
  for (i = persistMap->begin(); i != persistMap->end(); i++) {
    abandonedPersistBuffers->push_back(i->second);
  }
  persistMap->clear();

  // Try to clean the device memory
  cleanAbandonedPersistData();
}

void AccelManager::cleanAbandonedPersistData() {

  // Destroy any abandoned persist buffers, as long as there are not pending abandoned requests
  if (numAbandonedRequests <= 0) {

#if ACCELD
    printf("[ACCELMGR] :: PE %d :: AccelManager::cleanAbandonedPersistData() - Removing old persist buffers...\n", CkMyPe());
    fflush(NULL);
#endif

    std::list<AccelPersistBuffer*>::iterator i;
    for (i = abandonedPersistBuffers->begin(); i != abandonedPersistBuffers->end(); i++) {
      delete *i; // NOTE: Will cause deleteDeviceBuffer to be called, freeing device memory
    }
    abandonedPersistBuffers->clear();
  }
}


AccelObjectData* AccelManager::getObjectData(void* objectPtr) {

  // Check to see if there already is an entry for this object pointer, and if so, return it
  std::map<void*,AccelObjectData*>::iterator i;
  i = objectMap->find(objectPtr);
  if (i != objectMap->end()) { return (i->second); }

  // Otherwise, create a new object data record and return that
  AccelObjectData *rtn = new AccelObjectData;
  if (rtn == NULL) { CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::getObjectData() - Unable to allocate memory for object data.\n", CkMyPe()); fflush(NULL); }
  rtn->objectPtr = objectPtr;
  rtn->persistHostPtr = new void*[ACCEL_OBJECT_DATA_INIT_PERSIST_ARRAY_LEN];
  if (rtn->persistHostPtr == NULL) { CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::getObjectData() - Unable to allocate memory for object data's persist array.\n", CkMyPe()); fflush(NULL); }
  rtn->persistHostPtr_len = 0;
  rtn->persistHostPtr_maxLen = ACCEL_OBJECT_DATA_INIT_PERSIST_ARRAY_LEN;
  rtn->location = ACCEL_OBJECT_DATA_LOCATION_UNSET;
  objectMap->insert(std::pair<void*,AccelObjectData*>(objectPtr, rtn));
  return rtn;
}

void AccelManager::pullObjectData(void *objectPtr) {

  // Grab the object's record
  std::map<void*,AccelObjectData*>::iterator i;
  i = objectMap->find(objectPtr);
  if (i == objectMap->end()) { return; }  // Not found, so nothing to do

  // Loop through the host pointers and pull the data
  AccelObjectData *objectData = (i->second);
  for (int i = 0; i < objectData->persistHostPtr_len; i++) {
    pullPersistBuffer(objectData->persistHostPtr[i]);
  }
}

void AccelManager::pushObjectData(void *objectPtr) {

  // Grab the object's record
  std::map<void*,AccelObjectData*>::iterator i;
  i = objectMap->find(objectPtr);
  if (i == objectMap->end()) { return; }  // Not found, so nothing to do

  // Loop through the host pointers and pull the data
  AccelObjectData *objectData = (i->second);
  for (int i = 0; i < objectData->persistHostPtr_len; i++) {
    pushPersistBuffer(objectData->persistHostPtr[i]);
  }
}

void AccelManager::removeAllObjectData() {
  while (!(objectMap->empty())) { removeObjectData((objectMap->begin())->first); }
  objectMap->clear();  // Just to be safe
}

void AccelManager::removeObjectData(void *objectPtr) {

  // Grab the object's record
  std::map<void*,AccelObjectData*>::iterator i;
  i = objectMap->find(objectPtr);
  if (i == objectMap->end()) { return; }  // Not found, so nothing to do

  // Destroy the record
  AccelObjectData *objectData = (i->second);
  objectMap->erase(i);
  delete [] objectData->persistHostPtr;
  delete objectData;
}

void AccelManager::registerPersistHostBufferWithObject(void* objectPtr, void* hostPtr) {

  AccelObjectData *objectData = getObjectData(objectPtr);  // NOTE: Should be created if does not already exist
  if (objectData == NULL) { CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::registerPersistBufferWithObject() - Unable to retrieve object data.\n", CkMyPe()); fflush(NULL); }

  // Check to see if the persistHostPtr array needs to grow
  if (objectData->persistHostPtr_len >= objectData->persistHostPtr_maxLen) {
    int newPersistHostPtr_maxLen = objectData->persistHostPtr_maxLen + ACCEL_OBJECT_DATA_GROW_PERSIST_ARRAY_LEN;
    void **newPersistHostPtr = new void*[newPersistHostPtr_maxLen];
    if (newPersistHostPtr == NULL) { CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::registerPersistBufferWithObject() - Unable to allocate new persist host pointer array.\n", CkMyPe()); fflush(NULL); }
    memcpy((void*)newPersistHostPtr, (void*)objectData->persistHostPtr, sizeof(void*) * objectData->persistHostPtr_maxLen);
    delete [] objectData->persistHostPtr;
    objectData->persistHostPtr = newPersistHostPtr;
    objectData->persistHostPtr_maxLen = newPersistHostPtr_maxLen;
  }

  // Add the new entry
  objectData->persistHostPtr[(objectData->persistHostPtr_len)++] = hostPtr;

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: object @ %p persist host list = {", CkMyPe(), objectPtr);
  for (int i = 0; i < objectData->persistHostPtr_len; i++) {
    printf(" %p", objectData->persistHostPtr[i]);
  }
  printf(" }\n");
  fflush(NULL);
#endif

}

void AccelManager::unregisterPersistHostBufferWithObject(void* objectPtr, void* hostPtr) {

  AccelObjectData *objectData = getObjectData(objectPtr);
  if (objectData == NULL) { CkPrintf("[ACCEL-ERROR] :: PE %d :: AccelManager::unregisterPersistBufferWithObject() - Unable to retrieve object data.\n", CkMyPe()); fflush(NULL); }

  // If the passed hostPtr is registered with the given object, then remove it
  int i = 0;
  int foundFlag = 0;
  for (i = 0; i < objectData->persistHostPtr_len; i++) {
    if (objectData->persistHostPtr[i] == hostPtr) {
      foundFlag = 1;
      break;
    }
  }
  if (foundFlag) {
    (objectData->persistHostPtr_len)--;
    for (; i < objectData->persistHostPtr_len; i++) {
      objectData->persistHostPtr[i] = objectData->persistHostPtr[i + 1];
    }
  }

}

void AccelManager::notifyObjectIsPacking(void *objectPtr) {
  if (objectPtr == NULL) { return; }
  pullObjectData(objectPtr);

  // DMK - TODO | FIXME - At the moment, this is triggered from ArrayElement::pup when the
  //   PUP::er is unpacking.  In cases of load balancing, removing the object data is a
  //   valid course of action.  However, if checkpointing or fault tolerance is occuring,
  //   removing the object data is a waste of time.  Figure out how to tell the different
  //   situations apart from one another and act accordingly.  For now removal of the object
  //   data should work (provide correct behavior), but may hurt performance.
  removeObjectData(objectPtr);
}


const char * const ACCEL_SUCCESS_STR = "ACCEL_SUCCESS";
const char * const ACCEL_ERROR_UNKNOWN_STR = "ACCEL_ERROR_UNKNOWN_STR";
const char * const ACCEL_ERROR_INVALID_PARAMETER_STR = "ACCEL_ERROR_INVALID_PARAMETER";
const char * const ACCEL_ERROR_OUT_OF_MEMORY_STR = "ACCEL_ERROR_OUT_OF_MEMORY";
const char * const ACCEL_ERROR_INVALID_STRATEGY_STR = "ACCEL_ERROR_INVALID_STRATEGY";

const char * const accelErrorString(AccelError err) {
  switch (err) {
    case ACCEL_SUCCESS: return ACCEL_SUCCESS_STR;
    case ACCEL_ERROR_INVALID_PARAMETER: return ACCEL_ERROR_INVALID_PARAMETER_STR;
    case ACCEL_ERROR_OUT_OF_MEMORY: return ACCEL_ERROR_OUT_OF_MEMORY_STR;
    case ACCEL_ERROR_INVALID_STRATEGY: return ACCEL_ERROR_INVALID_STRATEGY_STR;
  }
  return ACCEL_ERROR_UNKNOWN_STR;
}

const char * const ACCEL_ISSUE_TRUE_STR = "ACCEL_ISSUE_TRUE";
const char * const ACCEL_ISSUE_FALSE_STR = "ACCEL_ISSUE_FALSE";

const char * const accelIssueFlagString(AccelIssueFlag aif) {
  if (aif == ACCEL_ISSUE_FALSE) {
    return ACCEL_ISSUE_FALSE_STR;
  }
  return ACCEL_ISSUE_TRUE_STR;
}

const char * const ACCEL_DEVICE_UNKNOWN_STR = "ACCEL_DEVICE_UNKNOWN";
const char * const ACCEL_DEVICE_HOST_STR = "ACCEL_DEVICE_HOST";
const char * const ACCEL_DEVICE_SPE_STR = "ACCEL_DEVICE_SPE";
const char * const ACCEL_DEVICE_GPU_CUDA_STR = "ACCEL_DEVICE_GPU_CUDA";
const char * const ACCEL_DEVICE_MIC_STR = "ACCEL_DEVICE_MIC";

const char * const accelDeviceTypeString(AccelDeviceType dt) {
  switch (dt) {
    case ACCEL_DEVICE_HOST: return ACCEL_DEVICE_HOST_STR;
    case ACCEL_DEVICE_SPE: return ACCEL_DEVICE_SPE_STR;
    case ACCEL_DEVICE_GPU_CUDA: return ACCEL_DEVICE_GPU_CUDA_STR;
    case ACCEL_DEVICE_MIC: return ACCEL_DEVICE_MIC_STR;
  }
  return ACCEL_DEVICE_UNKNOWN_STR;
}


AccelSharedLookup::AccelSharedLookup() {
  initialize();
}

AccelSharedLookup::~AccelSharedLookup() {
  cleanup();
}

void AccelSharedLookup::initialize() {

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: AccelSharedLookup@%p::initialize() - Called...\n", CkMyPe(), this); 
  fflush(NULL);
#endif


  // Initialize the recordList
  recordListLen = 0;
  recordListLen_max = ACCEL_SHARED_LOOKUP_INIT_LEN;
  recordList = new AccelSharedLookupRecord[recordListLen_max];
  if (recordList == NULL) {
#if ACCELD
    printf("[ACCEL-ERROR] :: AccelSharedLookup::initialize() - Unable to allocate record list.\n");
    fflush(NULL);
#endif
    recordListLen_max = 0;
    return;
  }

  // Initialize the recordTable
  recordTable = new AccelSharedLookupRecord*[ACCEL_SHARED_LOOKUP_TABLE_LEN];
  if (recordTable == NULL) {
#if ACCELD
    printf("[ACCEL-ERROR] :: AccelSharedLookup::initializ() - Unable to allocate record table.\n"); 
    fflush(NULL);
#endif
    return;
  }
  for (int i = 0; i < ACCEL_SHARED_LOOKUP_TABLE_LEN; i++) {
    recordTable[i] = NULL;
  }
}

void AccelSharedLookup::cleanup() {

  // Cleanup the recordTable
  if (recordTable != NULL) {
    delete [] recordTable;
    recordTable = NULL;
  }

  // Cleanup the recordList
  if (recordList != NULL) {
    delete [] recordList;
    recordList = NULL;
    recordListLen = 0;
    recordListLen_max = 0;
  }
}

void AccelSharedLookup::reset() {

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: AccelSharedLookup@%p::reset() - Called...\n", CkMyPe(), this); fflush(NULL);
  fflush(NULL);
#endif
  // If nothing has been inserted into this lookup, then nothing to do
  if (recordListLen == 0) { return; }

  // Move the recordListLen to zero (i.e. none in use)
  recordListLen = 0;

  memset(recordTable, 0, sizeof(AccelSharedLookupRecord*) * ACCEL_SHARED_LOOKUP_TABLE_LEN);
}

int AccelSharedLookup::hash(const AccelSharedLookupRecord * const sRec) {
  return AccelSharedLookup::hash(sRec->ptr, sRec->ptrLen);
}

union hash_index_union {
  void *p;
  int i[2];
  char c[8];
};

// Generates an 8-bit hash index into recordTable using ptr and ptrLen
int AccelSharedLookup::hash(const void * const ptr, const int ptrLen) {
  hash_index_union hiu;
  hiu.i[0] = hiu.i[1] = 0;
  hiu.p = (void*)ptr;
  hiu.i[0] |= ptrLen;
  return 0xFF & ((int)(hiu.c[0] ^ hiu.c[1] ^ hiu.c[2] ^ hiu.c[3] ^ hiu.c[4] ^ hiu.c[5] ^ hiu.c[6] ^ hiu.c[7]));
}

int AccelSharedLookup::lookupOffset(const void * const ptr, const int ptrLen) {

  // Verify the parameters
  if (ptr == NULL || ptrLen <= 0) { return ACCEL_INVALID_OFFSET; }

  // Generate the hash for this ptr and ptrLen pair
  int hash = AccelSharedLookup::hash(ptr, ptrLen);

  // DMK - DEBUG
  if (hash < 0 || hash >= ACCEL_SHARED_LOOKUP_TABLE_LEN) {
    printf("[ACCEL-ERROR] :: AccelSharedLookup::lookupOffset() - invalid hash detected (%d)...\n", hash); fflush(NULL);
  }

  // Search the recordTable for a record with this ptr and ptrLen combo
  AccelSharedLookupRecord *sRec = recordTable[hash];
  while (sRec != NULL) {
    if ((sRec->ptr == ptr) && (sRec->ptrLen == ptrLen)) { return sRec->offset; }
    sRec = sRec->next;
  }

  return ACCEL_INVALID_OFFSET;
}

void AccelSharedLookup::insertOffset(const void * const ptr, const int ptrLen, const int offset) {

  // Verify the parameters
  if (ptr == NULL || ptrLen <= 0 || offset <= 0) { return; }

  // Insert a new record with the (ptr, ptrLen, offset) combo.
  // NOTE: Because this is specifically meant to be used with the accelerated
  // entry method code generate in charmxi, don't check for duplicate records
  // in the table to save some performance (the code generate by charmxi does
  // this already, so don't dupliate the effort here).  The intended use is to
  // only insert of a previous call to lookup for the same record failed.

  // Make sure this is an available entry in recordList.  If not, grow the
  // list and rebuild the recordTable (new pointers for "next" members).
  if (recordListLen >= recordListLen_max) {

    // Create a new recordList
    int newRecordListLen_max = recordListLen_max + ACCEL_SHARED_LOOKUP_GROW_LEN;
    AccelSharedLookupRecord *newRecordList = new AccelSharedLookupRecord[newRecordListLen_max];
    if (newRecordList == NULL) { return; }

    // Recreate the recordTable
    int newRecordListLen = 0;
    for (int tableIndex = 0; tableIndex < ACCEL_SHARED_LOOKUP_TABLE_LEN; tableIndex++) {

      // Remove the old hash list from the table and replace it
      AccelSharedLookupRecord *sRec = recordTable[tableIndex];
      recordTable[tableIndex] = NULL;
      while (sRec != NULL) {
	newRecordList[newRecordListLen].next = recordTable[tableIndex];
        newRecordList[newRecordListLen].ptr = sRec->ptr;
        newRecordList[newRecordListLen].ptrLen = sRec->ptrLen;
        newRecordList[newRecordListLen].offset = sRec->offset;
        recordTable[tableIndex] = newRecordList + newRecordListLen;
        newRecordListLen++;
        sRec = sRec->next;
      }
    }

    // Cleanup the old recordList memory now that it's contents are copied
    delete [] recordList;
    recordList = newRecordList;
    recordListLen = newRecordListLen;
    recordListLen_max = newRecordListLen_max;
  }

  // Generate the hash for this ptr and ptrLen pair
  int hash = AccelSharedLookup::hash(ptr, ptrLen);

  // Insert the new offset record at the head of the table list
  recordList[recordListLen].next = recordTable[hash];
  recordList[recordListLen].ptr = (void*)ptr;
  recordList[recordListLen].ptrLen = ptrLen;
  recordList[recordListLen].offset = offset;
  recordTable[hash] = recordList + recordListLen;
  recordListLen++;

  return;
}


AccelManagerGroup::AccelManagerGroup() {

#if ACCELD
  printf("[ACCEL-DEBUG] :: PE %d :: AccelManagerGroup::AccelManagerGroup() - Called...\n", CkMyPe());fflush(NULL);
#endif
}

AccelManagerGroup::AccelManagerGroup(CkMigrateMessage *m) : CBase_AccelManagerGroup(m) { }

AccelManagerGroup::~AccelManagerGroup() {
}

void AccelManagerGroup::strategyAdjustPerf_setPercentDevice(float pd) {
}

void AccelManagerGroup::accelStrategy_setPercentDevice(float pd) {
  if (CkMyPe() != 0) {
    AccelManager *accelMgr = AccelManager::getAccelManager();
    if (accelMgr != NULL) { accelMgr->setPercentDevice(pd); }
  }
}

void AccelManagerGroup::accelStrategy_crawler_setOffsets(int numOffsets, float* offsets) {

  // Grab the manager and the manager's strategy
  AccelManager *manager = AccelManager::getAccelManager();
  if (manager == NULL) { printf("[ACCEL-ERROR] :: PE %d :: manager NULL in accelStrategy_crawler_applyNextOffset.\n", CkMyPe()); return; }
  AccelStrategy *strategy = manager->getStrategy();
  if (strategy == NULL) { printf("[ACCEL-ERROR] :: PE %d :: strategy NULL in accelStrategy_crawler_applyNextOffset.\n", CkMyPe()); return; }

  // The master might start transmitting offsets before every PE has become a slave.
  //   In these cases, just drop the offsets.  The master will measure the effect of
  //   the offsets not changing for this PE at first, but the PE will eventually
  //   become a slave and the offsets will matter.
  if (strategy->getStrategyType() != ACCEL_STRATEGY_CRAWLER_MASTER &&
      strategy->getStrategyType() != ACCEL_STRATEGY_CRAWLER_SLAVE
     ) {
    return;
  }

  // Since the strategy type is a crawler, cast to that and pass on the offsets
  ((AccelStrategy_Crawler*)strategy)->applyNewOffsets(manager, numOffsets, offsets);
}

void AccelManagerGroup::accelStrategy_notifyLBFinished() {
  if (CkMyPe() != 0) {
    AccelManager *accelManager = AccelManager::getAccelManager();
    if (accelManager != NULL) {
      accelManager->notifyLBFinished();
    }
  }
}

void AccelManagerGroup::shutdown() {

  // Cleanup after the AccelManager
  AccelManager::destroyAccelManager();

  // Checkin with the group member on PE 0
  thisProxy[0].shutdownCheckin();
}

void AccelManagerGroup::shutdownCheckin() {

  // Verify that this is only being called on PE 0
  if (CkMyPe() != 0) {
    printf("[ACCELMGR-ERROR] :: AccelManagerGroup::shutdownCheckin() called on PE other than zero!\n");
    return;
  }

  // Increment the counter for the number of PEs that have finished their
  // AccelManagerGroup::shutdown() calls.  Once all PEs have finished,
  // continue the overall shutdown process by calling CkExit() again.
  // NOTE: Since PE 0 should be the only PE reaching this code (per the if statement above),
  //   the use of a static variable should be SMP safe.
  static int checkInCount = 0;
  if ((++checkInCount) >= CkNumPes()) {
    CkExit();  // Continue the overall shutdown process
  }
}

void AccelManagerGroup::pup(PUP::er &p) {
  CBase_AccelManagerGroup::pup(p);
}


// DMK - DEBUG
void traceKernelIssueTime() {
  traceUserEvent(19482);
}


#include "ckaccel.def.h"
