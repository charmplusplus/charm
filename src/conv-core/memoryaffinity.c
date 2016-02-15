/**
  * Considering the effect of memory affinity on the performance
  * for the NUMA nodes.
  *
  * Author: Christiane Pousa Ribeiro
  * LIG - INRIA - MESCAL
  * April, 2010
  *
  * Modified and integrated into Charm++ by Chao Mei
  * May, 2010
  */

#include "converse.h"
#include "sockRoutines.h"
#define DEBUGP(x)  /* CmiPrintf x;  */
CpvExtern(int, myCPUAffToCore);
#if CMK_HAS_NUMACTRL
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <linux/mempolicy.h>
#include <numaif.h>
#include <numa.h>
#include <string.h>
#include <sched.h>
#include <math.h>
#include <dirent.h>
#include <sys/types.h>
typedef unsigned long mem_aff_mask;
static void MEM_MASK_ZERO(mem_aff_mask *mem_mask) {
    memset(mem_mask, 0, sizeof(mem_aff_mask));
}
static void MEM_MASK_SET(int nid, mem_aff_mask *mem_mask) {
    *mem_mask = *mem_mask | (1<<nid);
}
static void MEM_MASK_CLEAR(int nid, mem_aff_mask *mem_mask) {
    *mem_mask = *mem_mask & (~(1<<nid));
}
int print_mem_affinity() {
    mem_aff_mask mask;
    unsigned int len = 8*sizeof(mask);
    char spol[16];
    int policy;
    spol[0]='\0';
    /* Memory policy to the current process */
    if ((get_mempolicy(&policy,&mask,len,0,0)) < 0) {
        perror("mem_getaffinity");
        return -1;
    }
    if (policy == MPOL_INTERLEAVE)
        strcpy(spol,"INTERLEAVE");
    else if (policy == MPOL_BIND)
        strcpy(spol,"BIND");
    else
        strcpy(spol,"PREFERRED");
    CmiPrintf("%d: Mem affinity mask is: %08lx with policy %s\n", CmiMyPe(),mask,spol);
    return 0;
}
static int CmiNumNUMANodes(void) {
    return numa_max_node()+1;
}
static int getNUMANidByRank(int coreid) {
    int i;
    /*int totalCores = CmiNumCores();*/
    int totalNUMANodes = CmiNumNUMANodes();
    /*The core id array is viewed as 2D array but in form of 1D array*/
    /*int *coreids=(int *)malloc(sizeof(int)*totalCores);*/
    /*Assume each NUMA node has the same number of cores*/
    /*int nCoresPerNode = totalCores/totalNUMANodes;*/
    /*CmiAssert(totalCores%totalNUMANodes==0);*/
    char nodeStr[256];
    DIR* nodeDir;
    struct dirent* nodeDirEnt;
    int cpuid = -1;
    for (i=0; i<totalNUMANodes; i++) {
        snprintf(nodeStr, 256, "/sys/devices/system/node/node%d", i);
        nodeDir = opendir(nodeStr);
        if (nodeDir) {
            while ((nodeDirEnt = readdir(nodeDir))) {
                if (sscanf(nodeDirEnt->d_name, "cpu%d", &cpuid) == 1) {
                    if(cpuid == coreid) {
                        closedir(nodeDir);
	                return i;
                    }
                }
            }
            closedir(nodeDir);
        }
    }
    /*free(coreids);*/
    CmiPrintf("%d: the corresponding NUMA node for cpu id %d is not found!\n", CmiMyPe(), coreid);
    CmiAssert(0);
    return -1;
}

/**
 * policy: indicates the memory policy
 * nids: indicates the NUMA nodes to be applied
 * len: indicates how many NUMA nodes nids has
 */
int CmiSetMemAffinity(int policy, int *nids, int len) {
    int i;
    mem_aff_mask myMask;
    unsigned int masksize = 8*sizeof(mem_aff_mask);
    MEM_MASK_ZERO(&myMask);
    for (i=0; i<len; i++) MEM_MASK_SET(nids[i], &myMask);
    if (set_mempolicy(policy, &myMask, masksize)<0) {
        CmiPrintf("Error> setting memory policy (%d) error with mask %X\n", policy, myMask);
        return -1;
    } else
        return 0;
}
void CmiInitMemAffinity(char **argv) {

    int i;
    int policy=-1;
    /*step1: parsing args maffinity, mempol and nodemap (nodemap is optional)*/
    int maffinity_flag = CmiGetArgFlagDesc(argv, "+maffinity", "memory affinity");
    /*the node here refers to the nodes that are seen by libnuma on a phy node*/
    /*nodemap is a string of ints separated by ","*/
    char *nodemap = NULL;

    char *mpol = NULL;
    CmiGetArgStringDesc(argv, "+memnodemap", &nodemap, "define memory node mapping");
    CmiGetArgStringDesc(argv, "+mempol", &mpol, "define memory policy {bind, preferred or interleave} ");


    if (!maffinity_flag) return;

    /*Currently skip the communication thread*/
    /**
      * Note: the cpu affinity of comm thread may not be set
      * if "commap" is not specified. This is why the following
      * code regarding the comm thd needs to be put before
      * the codes that checks whether cpu affinity is set
      * or not
      */
    if (CmiMyPe() >= CmiNumPes()) {
        CmiNodeAllBarrier();
        return;
    }

    /*step2: checking whether the required cpu affinity has been set*/
    if (CpvInitialized(myCPUAffToCore) && CpvAccess(myCPUAffToCore)==-1) {
        if (CmiMyPe()==0)
            CmiPrintf("Charm++> memory affinity disabled because cpu affinity is not enabled!\n");
        CmiNodeAllBarrier();
        return;
    }

    if (CmiMyPe()==0) {
        CmiPrintf("Charm++> memory affinity enabled! \n");
    }

    /*Select memory policy*/
    if (mpol==NULL) {
        CmiAbort("Memory policy must be specified!\n");
    }
    if (strcmp(mpol, "interleave")==0) policy = MPOL_INTERLEAVE;
    else if (strcmp(mpol, "preferred")==0) policy = MPOL_PREFERRED;
    else if (strcmp(mpol, "bind")==0) policy = MPOL_BIND;
    else {
        CmiPrintf("Error> Invalid memory policy :%s\n", mpol);
        CmiAbort("Invalid memory policy!");
    }

    /**
     * step3: check whether nodemap is NULL or not
     * step 3a): nodemap is not NULL
     * step 3b): nodemap is NULL, set memory policy according to the result
     * of cpu affinity settings.
     */
    if (nodemap!=NULL) {
        int *nodemapArr = NULL;
        int nodemapArrSize = 1;
        int prevIntStart,j;
        int curnid;
        int myPhyRank = CpvAccess(myCPUAffToCore);
        int myMemNid;
        int retval = -1;
        for (i=0; i<strlen((const char *)nodemap); i++) {
            if (nodemap[i]==',') nodemapArrSize++;
        }
        nodemapArr = malloc(nodemapArrSize*sizeof(int));
        prevIntStart=j=0;
        for (i=0; i<strlen((const char *)nodemap); i++) {
            if (nodemap[i]==',') {
                curnid = atoi(nodemap+prevIntStart);
                if (curnid >= CmiNumNUMANodes()) {
                    CmiPrintf("Error> Invalid node number %d, only have %d nodes (0-%d) on the machine. \n", curnid, CmiNumNUMANodes(), CmiNumNUMANodes()-1);
                    CmiAbort("Invalid node number!");
                }
                nodemapArr[j++] = curnid;
                prevIntStart=i+1;
            }
        }
        /*record the last nid after the last comma*/
        curnid = atoi(nodemap+prevIntStart);
        if (curnid >= CmiNumNUMANodes()) {
            CmiPrintf("Error> Invalid node number %d, only have %d nodes (0-%d) on the machine. \n", curnid, CmiNumNUMANodes(), CmiNumNUMANodes()-1);
            CmiAbort("Invalid node number!");
        }
        nodemapArr[j] = curnid;

        myMemNid = nodemapArr[myPhyRank%nodemapArrSize];
        if (policy==MPOL_INTERLEAVE) {
            retval = CmiSetMemAffinity(policy, nodemapArr, nodemapArrSize);
        } else {
            retval = CmiSetMemAffinity(policy, &myMemNid, 1);
        }
        if (retval<0) {
            CmiAbort("set_mempolicy error w/ mem nodemap");
        }
        free(nodemapArr);
    } else {
        /*use the affinity map set by the cpu affinity*/
        int myPhyRank = CpvAccess(myCPUAffToCore);
        /*get the NUMA node id from myPhyRank (a core id)*/
        int myMemNid = getNUMANidByRank(myPhyRank);

        int retval=-1;
        if (policy==MPOL_INTERLEAVE) {
            int totalNUMANodes = CmiNumNUMANodes();
            int *nids = (int *)malloc(totalNUMANodes*sizeof(int));
            for (i=0; i<totalNUMANodes; i++) nids[i] = i;
            retval = CmiSetMemAffinity(policy, nids, totalNUMANodes);
            free(nids);
        } else {
            retval = CmiSetMemAffinity(policy, &myMemNid, 1);
        }
        if (retval<0) {
            CmiAbort("set_mempolicy error w/o mem nodemap");
        }
    }

    /*print_mem_affinity();*/
    CmiNodeAllBarrier();
}
#else
void CmiInitMemAffinity(char **argv) {
    char *tmpstr = NULL;
    int maffinity_flag = CmiGetArgFlagDesc(argv,"+maffinity",
                                           "memory affinity");
    if (maffinity_flag && CmiMyPe()==0)
        CmiPrintf("memory affinity is not supported, +maffinity flag disabled.\n");

    /* consume the remaining possible arguments */
    CmiGetArgStringDesc(argv, "+memnodemap", &tmpstr, "define memory node mapping");
    CmiGetArgStringDesc(argv, "+mempol", &tmpstr, "define memory policy {bind, preferred or interleave} ");
}
#endif

