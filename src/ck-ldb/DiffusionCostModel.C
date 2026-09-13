// Implementation of the cost-model config loader and tier resolution.
// Compiled into libck, since MetisLB prices its edge cut from the same table
// DiffusionLB prices its moves from and the two are separate modules. The
// standalone lbsim still includes it textually after the header; the guards
// make that harmless.

#include "DiffusionCostModel.h"
#include "charm++.h"

const char* DiffusionCostConfig::tierName(DiffusionTier t)
{
  switch (t)
  {
    case DIFF_TIER_INTRA_PROCESS: return "intra_process";
    case DIFF_TIER_IPC_SAME_GPU: return "ipc_same_gpu";
    case DIFF_TIER_IPC_CROSS_GPU: return "ipc_cross_gpu";
    case DIFF_TIER_INTER_NODE: return "inter_node";
    default: return "unknown";
  }
}

// Devices per physical host, and hence which device a PE drives. Taken from the
// config file rather than probed: the balancer runs on every PE but can only ask
// CUDA about its own device, and the calibration benchmark that wrote the file
// measured this very machine. Zero means "not stated", in which case the
// same-GPU/cross-GPU distinction cannot be drawn and the cross-GPU tier is used
// -- the common one-device-per-process layout, and the more expensive of the two,
// so an unstated layout errs toward leaving objects where they are.
static int diffCfgDevicesPerHost = 0;

static int diffDeviceOfPe(int pe)
{
  if (diffCfgDevicesPerHost <= 0) return -1;
  const int host = CmiPhysicalNodeID(pe);
  const int pesHere = CmiNumPesOnPhysicalNode(host);
  if (pesHere <= 0) return -1;
  // The same block map HAPI applies from PE-rank to device, lifted to the host.
  return (CmiPhysicalRank(pe) * diffCfgDevicesPerHost) / pesHere;
}

DiffusionTier DiffusionCostConfig::tierBetween(int pe1, int pe2)
{
  if (CmiNodeOf(pe1) == CmiNodeOf(pe2)) return DIFF_TIER_INTRA_PROCESS;
  if (!CmiPeOnSamePhysicalNode(pe1, pe2)) return DIFF_TIER_INTER_NODE;
  const int d1 = diffDeviceOfPe(pe1), d2 = diffDeviceOfPe(pe2);
  if (d1 >= 0 && d1 == d2) return DIFF_TIER_IPC_SAME_GPU;
  return DIFF_TIER_IPC_CROSS_GPU;
}

bool DiffusionCostConfig::load(const char* path)
{
  calibrated = false;
  if (path == NULL || path[0] == '\0') return false;

  FILE* f = fopen(path, "r");
  if (f == NULL)
  {
    if (CkMyPe() == 0)
      CkPrintf("CharmLB> DiffusionLB cost model: cannot open '%s'; running "
               "without a cost model (every quota-driven move is accepted, as "
               "before).\n",
               path);
    return false;
  }

  // Which of the eight alpha/beta values have been seen. All are required: a
  // partially specified model silently prices one tier at zero, which is the
  // cheapest possible answer and would wave through exactly the moves the model
  // is meant to stop.
  bool seen[DIFF_TIER_COUNT][2] = {};
  bool seenMigHost = false, seenMigDev = false, seenMigAlpha = false;

  char line[512];
  int lineNo = 0;
  while (fgets(line, sizeof(line), f) != NULL)
  {
    lineNo++;
    char* p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\0') continue;

    char key[128];
    double val;
    if (sscanf(p, "%127[^= \t] = %lf", key, &val) != 2 &&
        sscanf(p, "%127[^= \t]=%lf", key, &val) != 2)
    {
      if (CkMyPe() == 0)
        CkPrintf("CharmLB> DiffusionLB cost model: ignoring unparsable line %d "
                 "of '%s'\n", lineNo, path);
      continue;
    }

    if (strcmp(key, "devices_per_host") == 0) { diffCfgDevicesPerHost = (int)val; continue; }
    if (strcmp(key, "migrate_alpha") == 0) { migrateAlpha = val; seenMigAlpha = true; continue; }
    if (strcmp(key, "migrate_beta_host") == 0) { migrateBetaHost = val; seenMigHost = true; continue; }
    if (strcmp(key, "migrate_beta_device") == 0) { migrateBetaDevice = val; seenMigDev = true; continue; }
    if (strcmp(key, "placement_lifetime_intervals") == 0) { placementLifetimeIntervals = val; continue; }

    bool matched = false;
    for (int t = 0; t < DIFF_TIER_COUNT && !matched; t++)
    {
      char ka[160], kb[160];
      snprintf(ka, sizeof(ka), "%s_alpha", tierName((DiffusionTier)t));
      snprintf(kb, sizeof(kb), "%s_beta", tierName((DiffusionTier)t));
      if (strcmp(key, ka) == 0) { tier[t].alpha = val; seen[t][0] = matched = true; }
      else if (strcmp(key, kb) == 0) { tier[t].beta = val; seen[t][1] = matched = true; }
    }
    if (!matched && CkMyPe() == 0)
      CkPrintf("CharmLB> DiffusionLB cost model: unknown key '%s' at line %d of "
               "'%s'\n", key, lineNo, path);
  }
  fclose(f);

  for (int t = 0; t < DIFF_TIER_COUNT; t++)
  {
    if (!seen[t][0] || !seen[t][1])
    {
      if (CkMyPe() == 0)
        CkPrintf("CharmLB> DiffusionLB cost model: '%s' does not give both "
                 "%s_alpha and %s_beta; refusing to run a partial model.\n",
                 path, tierName((DiffusionTier)t), tierName((DiffusionTier)t));
      return false;
    }
  }
  if (!seenMigAlpha || !seenMigHost || !seenMigDev)
  {
    if (CkMyPe() == 0)
      CkPrintf("CharmLB> DiffusionLB cost model: '%s' is missing one of "
               "migrate_alpha, migrate_beta_host, migrate_beta_device.\n", path);
    return false;
  }

  calibrated = true;
  if (CkMyPe() == 0)
  {
    CkPrintf("CharmLB> DiffusionLB cost model loaded from '%s'"
             " (devices_per_host=%d, placement lifetime %.1f intervals)\n",
             path, diffCfgDevicesPerHost, placementLifetimeIntervals);
    for (int t = 0; t < DIFF_TIER_COUNT; t++)
      CkPrintf("CharmLB>   %-14s alpha=%.6es beta=%.6es/byte\n",
               tierName((DiffusionTier)t), tier[t].alpha, tier[t].beta);
  }
  return true;
}
