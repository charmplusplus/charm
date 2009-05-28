/**
   @addtogroup CharmComlib
   @{
   @file
*/

#include "ComlibStats.h"
#include "ComlibManager.h"

ComlibGlobalStats::ComlibGlobalStats() {
    statsArr = new ComlibLocalStats[CkNumPes()];
}

void ComlibGlobalStats::updateStats(ComlibLocalStats &stats, int p) {
    statsArr[p] = stats;
}

void ComlibGlobalStats::getAverageStats(int sid, double &avMsgSize, 
                                        double &avNumMessages, 
                                        double &avDegree,
                                        double &npes) {

    double bytes = 0, messages = 0, degree = 0;
    npes = avNumMessages = avMsgSize = avDegree = 0.0;

    for(int count = 0; count < CkNumPes(); count ++) {
        if(statsArr[count].cdata[sid].isRecorded()) {
            npes ++;
            //count send and received
            bytes += statsArr[count].cdata[sid].getTotalBytes();
            //count send and received
            messages += statsArr[count].cdata[sid].getTotalMessages();
            degree += statsArr[count].cdata[sid].getDegree();
        }
    }

    if(npes > 0.0 && messages > 0.0) {
        avNumMessages = messages/npes;
        avMsgSize = bytes/ (npes * avNumMessages);
        avDegree =  degree/npes;
    }
}


/*@}*/
