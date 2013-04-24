#ifndef COMLIB_STATS_H
#define COMLIB_STATS_H

/**
   @addtogroup CharmComlib
   @{
   @file
   
   @brief Classes for storing simple statistics about messages send and received.
*/

#include "charm++.h"
#include "convcomlibmanager.h"

class ComlibLocalStats;

/**
   Old class that is no longer used.
*/
class ComlibComRec {
    int nmessages_sent;
    int totalbytes_sent;
    int nmessages_received;
    int totalbytes_received;
    
    unsigned char *procMap; // Map of which processors have communicated
    int npes;               // Total number of processors participating 
                            // in the communication operation
    int degree;             // Number of processors messages are sent 
                            // to or received from
    bool recorded;

    friend class ComlibLocalStats;
 public:
    ComlibComRec() {
        npes = CkNumPes();
        procMap = 0;
        totalbytes_sent = 0;
        totalbytes_received = 0;
        nmessages_sent = 0;
        nmessages_received = 0;
        degree = 0;
        recorded = false;
    }

    ComlibComRec(int _npes) {
        npes = _npes;
        procMap = 0;
        totalbytes_sent = 0;
        totalbytes_received = 0;
        nmessages_sent = 0;
        nmessages_received = 0;
        degree = 0;
        recorded = false;
    }

    ~ComlibComRec() {
        if(recorded && procMap)
            CmiFree(procMap);

        procMap = 0;
    }

    void setNpes(int _npes) {npes = _npes;}
    bool isRecorded() { return recorded;}

    int getTotalBytes() { return  totalbytes_sent + totalbytes_received; }
    int getTotalMessages() { return nmessages_sent + nmessages_received;}
    int getDegree() { return degree;}

    inline void recordSend(int size, int dest) {
        if(!recorded) {
            recorded = true;
            int mapsize = (npes / (sizeof(char)*8) + 1) * sizeof(char); 
            procMap = (unsigned char*) CmiAlloc(mapsize);
            memset(procMap, 0, mapsize);
        }

        nmessages_sent ++;
        totalbytes_sent += size;
        int pos = dest / (sizeof(char)*8);
        int off = dest % (sizeof(char)*8);

        if((procMap[pos] & (1 << off)) == 0) {
            degree ++;
            procMap[pos] |= 1 << off;    //mark a processor as being sent to
        }
    }

    inline void recordSendM(int size, int *dest_m, int ndest) {
        
        if(!recorded) {
            recorded = true;
            int mapsize = (npes / (sizeof(char)*8) + 1) * sizeof(char); 
            procMap = (unsigned char*) CmiAlloc(mapsize);
            memset(procMap, 0, mapsize);
        }
        
        nmessages_sent += ndest;
        totalbytes_sent += size * ndest;
        
        for(int count = 0; count < ndest; count ++) {
            int pos = dest_m[count] / (sizeof(char)*8);
            int off = dest_m[count] % (sizeof(char)*8);
            
            if((procMap[pos] & (1 << off)) == 0) {
                degree ++;
                //mark a processor as being sent to
                procMap[pos] |= 1 << off;    
            }
        }
    }

    inline void recordRecv(int size, int src) {
        if(!recorded) {
            recorded = true;
            int mapsize = (npes / (sizeof(char)*8) + 1) * sizeof(char); 
            procMap = (unsigned char*) CmiAlloc(mapsize);
            memset(procMap, 0, mapsize);
        }

        nmessages_received ++;
        totalbytes_received += size;
        int pos = src / (sizeof(char) * 8);
        int off = src % (sizeof(char) * 8);

        if((procMap[pos] & (1 << off)) == 0) {
            degree ++;
            procMap[pos] |= 1 << off;    //mark a processor as being sent to
        }
    }
    
    inline void recordRecvM(int size, int *src_m, int nsrc) {
        if(!recorded) {
            recorded = true;
            int mapsize = (npes / (sizeof(char)*8) + 1) * sizeof(char); 
            procMap = (unsigned char*) CmiAlloc(mapsize);
            memset(procMap, 0, mapsize);
        }

        nmessages_received += nsrc;
        totalbytes_received += size * nsrc;

        for(int count = 0; count < nsrc; count++) {
            int pos = src_m[count] / (sizeof(char) * 8);
            int off = src_m[count] % (sizeof(char) * 8);
            
            if((procMap[pos] & (1 << off)) == 0) {
                degree ++;
                //mark a processor as being sent to
                procMap[pos] |= 1 << off;    
            }
        }
    }
    
    void reset () {
        if(procMap)
            CmiFree(procMap);
        procMap = 0;
        totalbytes_sent = 0;
        totalbytes_received = 0;
        nmessages_sent = 0;
        nmessages_received = 0;
        degree = 0;
        recorded = false;
    }

    void pup(PUP::er &p) {
        p | nmessages_sent;
        p | totalbytes_sent;
        p | nmessages_received;
        p | totalbytes_received;
        p | npes;
        p | degree;
        p | recorded;


        int mapsize = (npes / (sizeof(char)*8) + 1) * sizeof(char); 
        if(p.isUnpacking()) {
            if(recorded) 
                procMap = (unsigned char*) CmiAlloc(mapsize);
        }
        
        if(recorded)
            p(procMap, mapsize);
    }
};


/**
   Old class that is no longer used.
*/
class ComlibLocalStats {
 public:
    CkVec<ComlibComRec> cdata;
    int nstrats;

    ComlibLocalStats(int _strats) : cdata(_strats) {
      nstrats = _strats;
    }
    
    ComlibLocalStats() : cdata(1) {
      nstrats = 1;
    }

    void setNstrats(int nst) {
        nstrats = nst;
        cdata.resize(nstrats);
    }

    inline void recordSend(int sid, int size, int dest) {
      if(sid >= nstrats) {
	nstrats = sid + 1;
	cdata.resize(nstrats);
      }

      cdata[sid].recordSend(size, dest);

    }

    inline void recordRecv(int sid, int size, int src) {
      if(sid >= nstrats) {
	nstrats = sid + 1;
	cdata.resize(nstrats);
      }

      cdata[sid].recordRecv(size, src);
    }

    inline void recordSendM(int sid, int size, int *dest_m, int ndest) {
      if(sid >= nstrats) {
	nstrats = sid + 1;
	cdata.resize(nstrats);
      }

      cdata[sid].recordSendM(size, dest_m, ndest);
    }

    inline void recordRecvM(int sid, int size, int *src_m, int nsrc) {
      if(sid >= nstrats) {
	nstrats = sid + 1;
	cdata.resize(nstrats);
      }

      cdata[sid].recordRecvM(size, src_m, nsrc);
    }
    
    inline void reset() {
      for(int count = 0; count < nstrats; count++)
	cdata[count].reset();
    }

    void pup(PUP::er &p) {
      p | nstrats;
      p | cdata;
    }

    ComlibLocalStats & operator=(ComlibLocalStats &in) {
      nstrats = in.nstrats;

      cdata.resize(in.cdata.size());
      for(int count = 0; count < in.nstrats; count++) {
	if(in.cdata[count].isRecorded()) {
	  memcpy(&cdata[count],&in.cdata[count], sizeof(ComlibComRec));
	  
	  int npes = in.cdata[count].npes;
	  int mapsize = (npes / (sizeof(char)*8) + 1) * sizeof(char); 
	  cdata[count].procMap = (unsigned char*) CmiAlloc(mapsize);
	  memcpy(cdata[count].procMap, in.cdata[count].procMap, mapsize);
	}
	else
	  cdata[count].reset();
      }
      
      return *this;
    }
};

class ComlibGlobalStats {
 
  ComlibLocalStats *statsArr;
  
 public:
  
  ComlibGlobalStats();
  ~ComlibGlobalStats() {}
  
  void updateStats(ComlibLocalStats &stats, int pe); 
  
  //The average amount of data communicated
  void getAverageStats(int sid, double &, double &, double &, double &);
};

/*@}*/
#endif
