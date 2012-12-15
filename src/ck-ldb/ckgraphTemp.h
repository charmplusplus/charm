/** \file ckgraph.h
 *  Author: Osman Sarood
 *  Date Created: December 14th, 2012
 *  E-mail: sarood1@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _CKGRAPHTEMP_H_
#define _CKGRAPHTEMP_H_

#include <vector>
#include "BaseLB.h"
#include "ckgraph.h"

class ProcArrayTemp : public ProcArray {
  public:
		int *procFreq, *procFreqNew;
		void convertToInsts(BaseLB::LDStats *);
    ProcArrayTemp(BaseLB::LDStats *stats, int *p,int *pn) : ProcArray(stats){
			procFreqNew = pn; procFreq = p;
		}
    ~ProcArrayTemp() { }
};

class ObjGraphTemp : public ObjGraph {
	public:
		int *procFreq, *procFreqNew;
		ObjGraphTemp(BaseLB::LDStats *stats,int *p, int *pn):ObjGraph(stats){
			procFreqNew = pn; procFreq = p;
		}
		void convertToInsts(BaseLB::LDStats *);
};

#endif // _CKGRAPHTEMP_H_

/*@}*/

