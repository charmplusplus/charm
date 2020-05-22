#include "hapi_cupti.h"

CsvDeclare(uint64_t, cupti_start_time);

void cuptiInit() {
  // CUPTI Activity API needs to be initialized before cuInit() or
  // CUDA runtime call
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));

  // Enable CUPTI for memcpys and kernels
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

  // Register callbacks required by CUPTI
  CUPTI_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));

  // Get and set activity attributes, might be useful in the future
  /*
  size_t attr_value = 0, attr_value_size = sizeof(size_t);
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attr_value_size, &attr_value));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attr_value);
  attr_value *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attr_value_size, &attr_value));

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attr_value_size, &attr_value));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attr_value);
  attr_value *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attr_value_size, &attr_value));
  */

  CUPTI_CALL(cuptiGetTimestamp(&CsvAccess(cupti_start_time)));

  printf("HAPI> GPU tracing using CUPTI is enabled\n");
}

void CUPTIAPI cuptiBufferRequested(uint8_t **buffer, size_t *size, size_t *max_num_records)
{
  uint8_t *bfr = (uint8_t *) malloc(CUPTI_BUF_SIZE + CUPTI_ALIGN_SIZE);
  if (bfr == NULL) {
    CmiAbort("[CUPTI] Error: out of memory\n");
  }

  *size = CUPTI_BUF_SIZE;
  *buffer = CUPTI_ALIGN_BUFFER(bfr, CUPTI_ALIGN_SIZE);
  *max_num_records = 0;
}

void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  printf("========== CUPTI Activity Traces ==========\n");

  if (valid_size > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
      if (status == CUPTI_SUCCESS) {
        cuptiPrintActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
    if (dropped != 0) {
      printf("[CUPTI] Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}


static const char* cuptiMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    default:
      break;
  }

  return "<unknown>";
}

const char* cuptiActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }

  return "<unknown>";
}

const char *cuptiActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD:
      return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
      return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      return "STREAM";
    default:
      break;
  }

  return "<unknown>";
}

uint32_t cuptiActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      return id->pt.processId;
    case CUPTI_ACTIVITY_OBJECT_THREAD:
      return id->pt.threadId;
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
      return id->dcs.deviceId;
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      return id->dcs.contextId;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      return id->dcs.streamId;
    default:
      break;
  }

  return 0xffffffff;
}

static const char *cuptiComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
      return "CUDA";
    case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
      return "CUDA_MPS";
    default:
      break;
  }

  return "<unknown>";
}

static void cuptiPrintActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
    case CUPTI_ACTIVITY_KIND_DEVICE:
      {
        CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
        printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
            "multiprocessors %u, clock %u MHz\n",
            device->name, device->id,
            device->computeCapabilityMajor, device->computeCapabilityMinor,
            (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
            (unsigned int) (device->globalMemorySize / 1024 / 1024),
            device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
        break;
      }
    case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
      {
        CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
        printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
            attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
        break;
      }
    case CUPTI_ACTIVITY_KIND_CONTEXT:
      {
        CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
        printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
            context->contextId, context->deviceId,
            cuptiComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
            (int) context->nullStreamId);
        break;
      }
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      {
        CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
        printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
            cuptiMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
            (unsigned long long) (memcpy->start - CsvAccess(cupti_start_time)),
            (unsigned long long) (memcpy->end - CsvAccess(cupti_start_time)),
            memcpy->deviceId, memcpy->contextId, memcpy->streamId,
            memcpy->correlationId, memcpy->runtimeCorrelationId);
        break;
      }
    case CUPTI_ACTIVITY_KIND_MEMSET:
      {
        CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *) record;
        printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
            memset->value,
            (unsigned long long) (memset->start - CsvAccess(cupti_start_time)),
            (unsigned long long) (memset->end - CsvAccess(cupti_start_time)),
            memset->deviceId, memset->contextId, memset->streamId,
            memset->correlationId);
        break;
      }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      {
        const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
        CUpti_ActivityKernel3 *kernel = (CUpti_ActivityKernel3 *) record;
        printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
            kindString,
            kernel->name,
            (unsigned long long) (kernel->start - CsvAccess(cupti_start_time)),
            (unsigned long long) (kernel->end - CsvAccess(cupti_start_time)),
            kernel->deviceId, kernel->contextId, kernel->streamId,
            kernel->correlationId);
        printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
            kernel->gridX, kernel->gridY, kernel->gridZ,
            kernel->blockX, kernel->blockY, kernel->blockZ,
            kernel->staticSharedMemory, kernel->dynamicSharedMemory);
        break;
      }
    case CUPTI_ACTIVITY_KIND_DRIVER:
      {
        CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
        printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
            api->cbid,
            (unsigned long long) (api->start - CsvAccess(cupti_start_time)),
            (unsigned long long) (api->end - CsvAccess(cupti_start_time)),
            api->processId, api->threadId, api->correlationId);
        break;
      }
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      {
        CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
        printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
            api->cbid,
            (unsigned long long) (api->start - CsvAccess(cupti_start_time)),
            (unsigned long long) (api->end - CsvAccess(cupti_start_time)),
            api->processId, api->threadId, api->correlationId);
        break;
      }
    case CUPTI_ACTIVITY_KIND_NAME:
      {
        CUpti_ActivityName *name = (CUpti_ActivityName *) record;
        switch (name->objectKind)
        {
          case CUPTI_ACTIVITY_OBJECT_CONTEXT:
            printf("NAME  %s %u %s id %u, name %s\n",
                cuptiActivityObjectKindString(name->objectKind),
                cuptiActivityObjectKindId(name->objectKind, &name->objectId),
                cuptiActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
                cuptiActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
                name->name);
            break;
          case CUPTI_ACTIVITY_OBJECT_STREAM:
            printf("NAME %s %u %s %u %s id %u, name %s\n",
                cuptiActivityObjectKindString(name->objectKind),
                cuptiActivityObjectKindId(name->objectKind, &name->objectId),
                cuptiActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
                cuptiActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
                cuptiActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
                cuptiActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
                name->name);
            break;
          default:
            printf("NAME %s id %u, name %s\n",
                cuptiActivityObjectKindString(name->objectKind),
                cuptiActivityObjectKindId(name->objectKind, &name->objectId),
                name->name);
            break;
        }
        break;
      }
    case CUPTI_ACTIVITY_KIND_MARKER:
      {
        CUpti_ActivityMarker *marker = (CUpti_ActivityMarker *) record;
        printf("MARKER id %u [ %llu ], name %s\n",
            marker->id, (unsigned long long) marker->timestamp, marker->name);
        break;
      }
    case CUPTI_ACTIVITY_KIND_MARKER_DATA:
      {
        CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
        printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
            marker->id, marker->color, marker->category,
            (unsigned long long) marker->payload.metricValueUint64,
            marker->payload.metricValueDouble);
        break;
      }
    case CUPTI_ACTIVITY_KIND_OVERHEAD:
      {
        CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
        printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
            cuptiActivityOverheadKindString(overhead->overheadKind),
            (unsigned long long) overhead->start - CsvAccess(cupti_start_time),
            (unsigned long long) overhead->end - CsvAccess(cupti_start_time),
            cuptiActivityObjectKindString(overhead->objectKind),
            cuptiActivityObjectKindId(overhead->objectKind, &overhead->objectId));
        break;
      }
    default:
      printf("  <unknown>\n");
      break;
  }
}
