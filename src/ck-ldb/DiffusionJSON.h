#include "json.hpp"
#include <fstream>
#include <iostream>

#include "BaseLB.h"

using json = nlohmann::json;

// write lbstats to a json file
int LBwriteStatsMsgs(BaseLB::LDStats* statsData)
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

    if (from >= CkNumPes() || from < 0)
    {
      CkAbort("<LBwriteStatsMsgs> from_proc is out of bounds");
    }

    if (to >= CkNumPes() || to < -1)
    {
      CkAbort("<LBwriteStatsMsgs> to_proc is out of bounds");
    }

    if (to != -1 && (statsData->objData[obj].migratable == false))
    {
      CkAbort("<LBwriteStatsMsgs> object shoult not be migrating");
    }

    LDObjData odata = statsData->objData[obj];
    objpe[std::to_string(obj)] = {{"migratable", odata.migratable},
                                  {"position", odata.position},
                                  {"wallTime", odata.wallTime},
                                  {"oldpe", from},
                                  {"newpe", (to == -1) ? from : to}};
  }

  jsonData["n_procs"] = statsData->procs.size();
  jsonData["objData"] = objpe;
  // objdata: objID, omID, migratable, position, cpuTime, wallTime

  // from_proc: old pe for object
  // to_proc: pe object is migrating to NOT USING

  // commData: list of (src_proc, sender, receiver, recv_type, msg_size, msg_count)
  // can get edges between objects

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
