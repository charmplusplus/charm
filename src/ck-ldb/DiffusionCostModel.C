// Implementation of the cost-model config loader and tier resolution.
// Compiled into libck, since MetisLB prices its edge cut from the same table
// DiffusionLB prices its moves from and the two are separate modules. The
// standalone lbsim still includes it textually after the header; the guards
// make that harmless.

#include "DiffusionCostModel.h"
#include <string>
#include "charm++.h"
#if CMK_GPU
#include "hapi.h"
#endif

const char* DiffusionCostConfig::tierName(DiffusionTier t)
{
  switch (t)
  {
    case DIFF_TIER_INTRA_PROCESS: return "intra_process";
    case DIFF_TIER_INTRA_PROCESS_PEER: return "intra_process_peer";
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
#if CMK_GPU
  // The runtime's own mapping first: exact for this process's PEs and the mapping's formula for
  // every other process, for any process/GPU split. The config arithmetic below is the fallback
  // for a build whose HAPI has not mapped yet.
  {
    const int d = hapiDeviceIndexOfPe(pe);
    if (d >= 0) return d;
  }
#endif
  if (diffCfgDevicesPerHost <= 0) return -1;
  const int host = CmiPhysicalNodeID(pe);
  const int pesHere = CmiNumPesOnPhysicalNode(host);
  if (pesHere <= 0) return -1;
  // The same block map HAPI applies from PE-rank to device, lifted to the host.
  return (CmiPhysicalRank(pe) * diffCfgDevicesPerHost) / pesHere;
}

DiffusionTier DiffusionCostConfig::tierBetween(int pe1, int pe2)
{
  if (!CmiPeOnSamePhysicalNode(pe1, pe2)) return DIFF_TIER_INTER_NODE;
  const int d1 = diffDeviceOfPe(pe1), d2 = diffDeviceOfPe(pe2);
  if (CmiNodeOf(pe1) == CmiNodeOf(pe2))
  {
    // One process: a memcpy on one device, a peer copy between two of its devices. An unknown
    // device (no mapping yet) is taken as the same one, the historical answer.
    return (d1 >= 0 && d2 >= 0 && d1 != d2) ? DIFF_TIER_INTRA_PROCESS_PEER
                                            : DIFF_TIER_INTRA_PROCESS;
  }
  if (d1 >= 0 && d1 == d2) return DIFF_TIER_IPC_SAME_GPU;
  return DIFF_TIER_IPC_CROSS_GPU;
}

DiffusionTier DiffusionCostConfig::tierBetween(int pe1, int pe2, uint64_t dev1, uint64_t dev2)
{
  if (!CmiPeOnSamePhysicalNode(pe1, pe2)) return DIFF_TIER_INTER_NODE;
  if (CmiNodeOf(pe1) == CmiNodeOf(pe2))
    return (dev1 != dev2) ? DIFF_TIER_INTRA_PROCESS_PEER : DIFF_TIER_INTRA_PROCESS;
  return (dev1 == dev2) ? DIFF_TIER_IPC_SAME_GPU : DIFF_TIER_IPC_CROSS_GPU;
}

// The name of the device this PE drives, for choosing a built-in table; empty
// without a GPU build or before HAPI has mapped the PEs to devices.
static std::string diffDeviceName()
{
#if CMK_GPU
  const int d = hapiDeviceIndexOfPe(CkMyPe());
  if (d >= 0)
  {
    hapiDeviceProp prop;
    if (hapiGetDeviceProperties(&prop, d) == hapiSuccess) return prop.name;
  }
#endif
  return std::string();
}

bool DiffusionCostConfig::loadBuiltin(const char* why)
{
  for (int c = 0; c < 2; c++)
    for (int k = 0; k <= DIFF_XCHG_MAX_FAN; k++) xchgSeen[c][k] = false;
  diffCfgDevicesPerHost = 4;
  const std::string dev = diffDeviceName();
  const char* which;
  if (dev.find("A100") != std::string::npos)
  {
    // benchmarks/charm++/cuda/gpudirect/lbcalib/lbcost.a100.2n.chan.conf (job 23637513,
    // 2026-11-07, gpua010 and gpua029: A100-SXM4-40GB x4 per host, every pair on four
    // NVLink lanes, Slingshot between hosts; 8 PEs, 64 B .. 4 MB tiers, 200 timed
    // iterations; the fan sweep by device channels, 4..128 MB, 10 rounds, with the
    // cross-host put announced and read by the owner). Two, three and four copies
    // exchanging on one host each cost a peer 93 GB/s; across hosts 12.1 GB/s a
    // peer (the one-sided write of the 2026-11-06 table ran at 3.0 with both hosts
    // sending: a write into device memory is the slow leg on this fabric).
    which = "Delta A100, lbcalib by channels 2026-11-07";
    tier[DIFF_TIER_INTRA_PROCESS].alpha = 3.539371984e-05;
    tier[DIFF_TIER_INTRA_PROCESS].beta = 9.644288548e-12;
    tier[DIFF_TIER_IPC_SAME_GPU].alpha = 3.539371984e-05;
    tier[DIFF_TIER_IPC_SAME_GPU].beta = 9.644288548e-12;
    tier[DIFF_TIER_IPC_CROSS_GPU].alpha = 3.539371984e-05;
    tier[DIFF_TIER_IPC_CROSS_GPU].beta = 9.644288548e-12;
    tier[DIFF_TIER_INTRA_PROCESS_PEER] = tier[DIFF_TIER_IPC_CROSS_GPU];
    tier[DIFF_TIER_INTER_NODE].alpha = 2.203699425e-05;
    tier[DIFF_TIER_INTER_NODE].beta = 5.175327229e-11;
    migrateAlpha = 2.219108047e-05;
    migrateBetaHost = 3.389214890e-10;
    migrateBetaDevice = 1.800491336e-11;
    xchg[0][2].alpha = 2.781751737e-05; xchg[0][2].beta = 1.071665299e-11; xchgSeen[0][2] = true;
    xchg[0][3].alpha = 1.722823328e-05; xchg[0][3].beta = 1.070247648e-11; xchgSeen[0][3] = true;
    xchg[0][4].alpha = 1.419923001e-05; xchg[0][4].beta = 1.072087753e-11; xchgSeen[0][4] = true;
    xchg[1][2].alpha = 9.446357613e-06; xchg[1][2].beta = 8.272221957e-11; xchgSeen[1][2] = true;
  }
  else
  {
    // benchmarks/charm++/cuda/gpudirect/lbcalib/lbcost.a40.2n.chan2.conf (job 23637512,
    // 2026-11-07, gpub011 and gpub095: A40 x4 per host on PCIe, Slingshot between
    // hosts; 8 PEs, 64 B .. 4 MB tiers, 200 timed iterations; the fan sweep by
    // device channels, 4..128 MB, 10 rounds, with the cross-host put announced and
    // read by the owner). alpha seconds per message, beta seconds per byte. The
    // intra-process tiers were not measured and take the cross-GPU IPC figures, as
    // load() does for a file without them. This is the table for the A40 nodes
    // and the default for any other device.
    which = "Delta A40, lbcalib by channels 2026-11-07";
    tier[DIFF_TIER_INTRA_PROCESS].alpha = 3.435739021e-05;
    tier[DIFF_TIER_INTRA_PROCESS].beta = 3.953828829e-11;
    tier[DIFF_TIER_IPC_SAME_GPU].alpha = 3.435739021e-05;
    tier[DIFF_TIER_IPC_SAME_GPU].beta = 3.953828829e-11;
    tier[DIFF_TIER_IPC_CROSS_GPU].alpha = 3.435739021e-05;
    tier[DIFF_TIER_IPC_CROSS_GPU].beta = 3.953828829e-11;
    tier[DIFF_TIER_INTRA_PROCESS_PEER] = tier[DIFF_TIER_IPC_CROSS_GPU];
    tier[DIFF_TIER_INTER_NODE].alpha = 2.031688685e-05;
    tier[DIFF_TIER_INTER_NODE].beta = 5.491234402e-11;
    migrateAlpha = 2.335390015e-05;
    migrateBetaHost = 3.297443444e-10;
    migrateBetaDevice = 4.707438766e-11;
    // What one peer costs a copy when k copies exchange 4..128 MB buffers all at
    // once, k PEs one per device on one host, or one per host: 24.1, 16.0 and
    // 13.9 GB/s a peer at two, three and four copies on one host (the PCIe
    // fabric's concurrent copies; 93 on the NVLink A100 nodes), 14.2 across hosts
    // (the entry-method sweep of 2026-11-05 read 18, 1.5 and 0.9 on one host: its
    // receive path's collapse, not the fabric's).
    xchg[0][2].alpha = 2.726571477e-05; xchg[0][2].beta = 4.157963980e-11; xchgSeen[0][2] = true;
    xchg[0][3].alpha = 1.786603263e-05; xchg[0][3].beta = 6.231011954e-11; xchgSeen[0][3] = true;
    xchg[0][4].alpha = 1.443725320e-05; xchg[0][4].beta = 7.210188391e-11; xchgSeen[0][4] = true;
    xchg[1][2].alpha = 4.439709791e-05; xchg[1][2].beta = 7.037423721e-11; xchgSeen[1][2] = true;
  }
  calibrated = true;
  if (CkMyPe() == 0)
    CkPrintf("CharmLB> transfer-cost model: built-in table (%s) for device \"%s\"%s%s; "
             "+LBCostConfig <file> overrides it\n", which, dev.empty() ? "unknown" : dev.c_str(),
             why ? " " : "", why ? why : "");
  return true;
}

void DiffusionCostConfig::loadDefault(const char* path)
{
  if (path != NULL && path[0] != '\0' && load(path)) return;
  loadBuiltin(path != NULL && path[0] != '\0' ? "because the named file did not load" : NULL);
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
    {
      // xchg_fan<k>_host_alpha | _beta, xchg_fan<k>_inter_alpha | _beta: optional
      int fan = 0; char cls[16], which[16];
      if (sscanf(key, "xchg_fan%d_%15[a-z]_%15[a-z]", &fan, cls, which) == 3 && fan >= 2 && fan <= DIFF_XCHG_MAX_FAN &&
          (strcmp(cls, "host") == 0 || strcmp(cls, "inter") == 0) && (strcmp(which, "alpha") == 0 || strcmp(which, "beta") == 0))
      {
        const int c = strcmp(cls, "host") == 0 ? 0 : 1;
        if (which[0] == 'a') xchg[c][fan].alpha = val; else xchg[c][fan].beta = val;
        xchgSeen[c][fan] = true;   // set by either; a table writes both
        continue;
      }
    }

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

  // Tables written before the peer tier existed do not name it. A same-process peer copy crosses
  // the same PCIe link as a cross-process cross-GPU transfer, so that tier stands in until
  // lbcalib measures it (the loader still insists on the cross-GPU tier itself below).
  const bool peerGiven = seen[DIFF_TIER_INTRA_PROCESS_PEER][0] && seen[DIFF_TIER_INTRA_PROCESS_PEER][1];
  if (!peerGiven)
  {
    tier[DIFF_TIER_INTRA_PROCESS_PEER] = tier[DIFF_TIER_IPC_CROSS_GPU];
    seen[DIFF_TIER_INTRA_PROCESS_PEER][0] = seen[DIFF_TIER_INTRA_PROCESS_PEER][1] = true;
  }
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
    if (!peerGiven)
      CkPrintf("CharmLB>   intra_process_peer not in the table: priced as ipc_cross_gpu until "
               "lbcalib measures it\n");
    for (int t = 0; t < DIFF_TIER_COUNT; t++)
      CkPrintf("CharmLB>   %-14s alpha=%.6es beta=%.6es/byte\n",
               tierName((DiffusionTier)t), tier[t].alpha, tier[t].beta);
  }
  return true;
}
