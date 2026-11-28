#include "json.hpp"
#include <fstream>
#include <iostream>

#include "BaseLB.h"

using json = nlohmann::json;

int DiffusionLB::writeStatsMsgs(BaseLB::LDStats* statsData) {
#if CMK_LBDB_ON
  const char* filename = "lbdump.dat";
  FILE *f = fopen(filename, "w");
  int stats_msg_count ;
  if (f==NULL) {
    CkAbort("Fatal Error> writeStatsMsgs failed to open the output file %s!\n", filename);
  }

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();
  PUP::toDisk p(f);
  p((char *)&machInfo, sizeof(machInfo));	// machine info

  p|_lb_args.lbversion();		// write version number
  p|stats_msg_count;

  statsData->n_nodes = numNodes;
  statsData->pup(p);

  fclose(f);

  CmiPrintf("WriteStatsMsgs to %s succeed (with %d nodes)!\n", filename, statsData->n_nodes);
#endif
}

// write lbstats to a json file
int DiffusionLB::writeStatsMsgsJSON(BaseLB::LDStats* statsData)
{
  json jsonData;

  jsonData["n_migratable"] = statsData->n_migrateobjs;

  // processor stats: n_objs, pe_speed, total_walltime, idletime, bg_walltime, pe,
  // available

  json objpe = json::object();

  for (int obj = 0; obj < statsData->objData.size(); obj++)
  {
    int from = statsData->from_proc[obj];
    int to = statsData->to_proc[obj];

    if (from >= numPes || from < 0)
    {
      CkAbort("<writeStatsMsgs> from_proc is out of bounds (%d not in [0,%d))", from, numPes);
    }

    if (to >= numPes || to < -1)
    {
      CkAbort("<writeStatsMsgs> to_proc is out of bounds (%d not in [0,%d))", to, numPes);
    }

    if (to != -1 && (statsData->objData[obj].migratable == false))
    {
      CkAbort("<writeStatsMsgs> object should not be migrating");
    }

    LDObjData odata = statsData->objData[obj];
    objpe[std::to_string(obj)] = {{"migratable", odata.migratable},
                                  {"position", odata.position},
                                  {"wallTime", odata.wallTime},
                                  {"oldpe", from},
                                  {"newpe", (to == -1) ? from : to},
                                  {"omHandle", odata.omID().id.idx},
                                  {"id", odata.objID()}};

    // from_proc: old pe for object
    // to_proc: pe object is migrating to NOT USING
  }

  jsonData["n_procs"] = statsData->procs.size();
  jsonData["n_nodes"] = CkNumNodes();
  jsonData["objData"] = objpe; // objdata: objID, omID, migratable, position, cpuTime, wallTime


  json commdata = json::object();
  for (int comm = 0; comm < statsData->commData.size(); comm++)
  {
    LDCommData cdata = statsData->commData[comm];
    commdata[std::to_string(comm)] = {
        {"src_proc", cdata.from_proc()},
        {"sender_obj", {{"omID", cdata.sender.omID().id.idx},
                        {"objID", cdata.sender.objID()}}},
        {"receiver_obj", {{"omID", cdata.receiver.get_destObj().omID().id.idx},
                          {"objID", cdata.receiver.get_destObj().objID()}}},
        {"recv_type", cdata.recv_type()},
        {"msg_size", cdata.bytes},
        {"msg_count", cdata.messages}};
  }
  jsonData["commData"] = commdata; // commData: list of (src_proc, sender, receiver, recv_type, msg_size, msg_count)

  std::ofstream outputFile("lbdump.json");
  if (outputFile.is_open())
  {
    outputFile << jsonData.dump(4) << std::endl;
    outputFile.close();
    std::cout << "JSON data successfully written to lbdump.json" << std::endl;
  }
  else
  {
    std::cerr << "Unable to open file for writing!" << std::endl;
    return 1;
  }

  return 0;
}
