//#define NO_TEMP_LB
//#define ORG_VERSION
//#define MAX_MIN
#define MAX_TEMP 49
//#define tolerance 0.03
/** \file TempAwareRefineLB.C
 *
 *  Written by Osman Sarood
 *  Temperature aware load balancer. Needs frequency control access to work.
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "TempAwareRefineLB.h"
#include "ckgraph.h"
#include <algorithm>

CreateLBFunc_Def(TempAwareRefineLB, "always assign the heaviest obj onto lightest loaded processor.")

#ifdef TEMP_LDB


static int cpufreq_sysfs_write (
                     const char *setting,int proc
                     )
{
char path[100];
sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",proc);
                FILE *fd = fopen (path, "w");

                if (!fd) {
                        printf("PROC#%d ooooooo666 FILE OPEN ERROR file=%s\n",CkMyPe(),path);
                        return -1;
                }
//                else CkPrintf("PROC#%d opened freq file=%s\n",proc,path);

        fseek ( fd , 0 , SEEK_SET );
        int numw=fprintf (fd, setting);
        if (numw <= 0) {

                fclose (fd);
                printf("FILE WRITING ERROR\n");
                return 0;
        }
//        else CkPrintf("Freq for Proc#%d set to %s numw=%d\n",proc,setting,numw);
        fclose(fd);
        return 1;
}

float TempAwareRefineLB::getTemp(int cpu)
{
        char val[10];
        FILE *f;
                char path[100];
                sprintf(path,"/sys/devices/platform/coretemp.%d/temp1_input",cpu);
                f=fopen(path,"r");
                if (!f) {
                        printf("777 FILE OPEN ERROR file=%s\n",path);
                        exit(0);
                }

        if(f==NULL) {printf("ddddddddddddddddddddddddddd\n");exit(0);}
        fgets(val,10,f);
        fclose(f);
        return atof(val)/1000;
}

static int cpufreq_sysfs_read (int proc)
{
        FILE *fd;
        char path[100];
        int i=proc;
        sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",i);

        fd = fopen (path, "r");

        if (!fd) {
                printf("22 FILE OPEN ERROR file=%s\n",path);
                return 0;
        }
        char val[10];
        fgets(val,10,fd);
        int ff=atoi(val);
        fclose (fd);

        return ff;
}

void printCurrentTemperature(void *LB, double curWallTime)
{
  TempAwareRefineLB *taalb = static_cast<TempAwareRefineLB *>(LB);
  int pe = CkMyPe();
  float temp = taalb->getTemp(pe % taalb->physicalCoresPerNode);
  int freq = cpufreq_sysfs_read (pe % taalb->logicalCoresPerNode);
  fprintf(taalb->logFD, "%f, %d, %f, %d\n", curWallTime, pe, temp, freq);
}

int getProcFreqPtr(int *freqs,int numAvail,int freq)
{
	for(int i=0;i<numAvail;i++) if(freqs[i]==freq) return i;
}
#endif
FILE *migFile;
double starting;
TempAwareRefineLB::TempAwareRefineLB(const CkLBOptions &opt): CBase_TempAwareRefineLB(opt)
{
#ifdef TEMP_LDB
starting=CmiWallTimer();
//  procsPerNode=4;
migFile=fopen("migInfo","w");
  numAvailFreqs = 11;
//numAvailFreqs = 14;
//numAvailFreqs = 7;  
freqs=new int[numAvailFreqs];
freqsEffect=new int[numAvailFreqs];
// for might (lab machine)
/*
  freqs[0] = 2262000;
  freqs[1] = 2261000;
  freqs[2] = 2128000;
  freqs[3] = 1995000;
  freqs[4] = 1862000;
  freqs[5] = 1729000;
  freqs[6] = 1596000;
*/

// for tarekc cluster
  freqs[0] = 2395000;
  freqs[1] = 2394000;
  freqs[2] = 2261000;
  freqs[3] = 2128000;
  freqs[4] = 1995000;
  freqs[5] = 1862000;
  freqs[6] = 1729000;
  freqs[7] = 1596000;
  freqs[8] = 1463000;
  freqs[9] = 1330000;
  freqs[10] = 1197000;

	freqsEffect[0] = 1979886;
  freqsEffect[1] = 1943017;
  freqsEffect[2] = 1910989;
  freqsEffect[3] = 1876619;
  freqsEffect[4] = 1824126;
  freqsEffect[5] = 1763990;
  freqsEffect[6] = 1666773;
  freqsEffect[7] = 1560224;
  freqsEffect[8] = 1443154;
  freqsEffect[9] = 1317009;
  freqsEffect[10] = 1200000;


/*
// for grace, humility etc (lab i7 machines)
  freqs[0] = 2801000;
  freqs[1] = 2800000;
  freqs[2] = 2667000;
  freqs[3] = 2533000;
  freqs[4] = 2400000;
  freqs[5] = 2267000;
freqs[6] = 2133000;
freqs[7] = 2000000;
freqs[8] = 1867000;
freqs[9] = 1733000;
freqs[10] = 1600000;
freqs[11] = 1467000;
freqs[12] = 1333000;
  freqs[13] = 1200000;
*/

  procFreqPtr = new int[CkNumPes()];

  for(int i=0;i<CkNumPes();i++)
  {
        char newfreq[10];
        sprintf(newfreq,"%d",freqs[0]);
	cpufreq_sysfs_write(newfreq,i%physicalCoresPerNode);	
	procFreqPtr[i]=0;
  }
//  logicalCoresPerChip=4;
  procFreq=NULL;
  procTemp=NULL;
	procFreqNew=NULL;
	procFreqNewEffect = NULL;
	avgChipTemp=NULL;
  lbname = "TempAwareRefineLB";
  if (CkMyPe()==0)
    CkPrintf("[%d] TempAwareRefineLB created\n",CkMyPe());

  char logFile[100];
  snprintf(logFile, sizeof(logFile), "temp_freq.log.%d", CkMyPe());
  if ((logFD = fopen(logFile, "a"))) {
    fprintf(logFD, "Time, PE, Temperature, Frequency\n");
  } else {
    CkAbort("Couldn't open temperature/frequency log file");
  }


  CcdCallOnConditionKeep(CcdPERIODIC_1second, &printCurrentTemperature, this);
#else
	CmiAbort("TEMPLB ERROR: not supported without TEMP_LDB flag.\n");
#endif

}

void TempAwareRefineLB::populateEffectiveFreq(int numProcs)
{
#ifdef TEMP_LDB
	for(int i=0;i<numProcs;i++)
	{
		for(int j=0;j<numAvailFreqs;j++)
		{
			if(freqs[j] == procFreqNew[i]) // same freq . copy effective freq
			{
				procFreqNewEffect[i] = freqsEffect[j];
//				CkPrintf("** Proc%d j:%d NEWFreq:%d\n",i,j,procFreqNewEffect[i]);
			}
			if(freqs[j] == procFreq[i]) 
			{
				procFreqEffect[i] = freqsEffect[j];
//				CkPrintf("-- Proc%d j:%d OLDFreq:%d procFreq:%d \n",i,j,procFreqEffect[i],procFreq[i]);
			}
		}
	}
#endif
}

bool TempAwareRefineLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

class ProcLoadGreater {
  public:
    bool operator()(ProcInfo p1, ProcInfo p2) {
      return (p1.getTotalLoad() > p2.getTotalLoad());
    }
};

class ProcLoadLesser {
  public:
    bool operator()(ProcInfo p1, ProcInfo p2) {
      return (p1.getTotalLoad() < p2.getTotalLoad());
    }
};

class ObjLoadGreater {
  public:
    bool operator()(Vertex v1, Vertex v2) {
      return (v1.getVertexLoad() > v2.getVertexLoad());
    }
};

void TempAwareRefineLB::changeFreq(int nFreq)
{
#ifdef TEMP_LDB
        //CkPrintf("PROC#%d in changeFreq numProcs=%d\n",CkMyPe(),nFreq);
//  for(int i=0;i<numProcs;i++)
  {
//        if(procFreq[i]!=procFreqNew[i])
        {
              char newfreq[10];
              sprintf(newfreq,"%d",nFreq);
              cpufreq_sysfs_write(newfreq,CkMyPe()%physicalCoresPerNode);//i%physicalCoresPerNode);
//            CkPrintf("PROC#%d freq changing from %d to %d temp=%f\n",i,procFreq[i],procFreqNew[i],procTemp[i]);
        }
  }
#endif
}

#ifdef TEMP_LDB
int getTaskIdForMigration(ObjGraph *ogr,int pe,int start)
{
	for(int vert = start; vert < ogr->vertices.size(); vert++)
	{
		if(ogr->vertices[vert].getCurrentPe()==pe && ogr->vertices[vert].getNewPe()==-1) return vert;
	}
        CkPrintf("THERE IS A PROBLEM IN TEMPREFINELB 222 start=%d pe=%d objArraySize=%d!!!!!\n",start,pe,ogr->vertices.size());
        CkExit();
}

int getNumTasks(ObjGraph *ogr,int pe)
{
	int c=0;
        for(int vert = 0; vert < ogr->vertices.size(); vert++)
        {
                if(ogr->vertices[vert].getCurrentPe()==pe && ogr->vertices[vert].getNewPe()==-1) c++;
        }
	return c;
}

int getTaskIdForMigration(ObjGraph *ogr,int pe,std::vector<int> assTasks)
{
        for(int vert = 0; vert < ogr->vertices.size(); vert++)
        {
                if(ogr->vertices[vert].getCurrentPe()==pe  && ogr->vertices[vert].getNewPe()==-1)
		{
/*
   CkPrintf("======================= pe=%d vert=%d ========================\n",pe,vert);

			bool hasIt=false;
			for(int i=0;i<assTasks.size();i++)
			{
				if(vert==assTasks[i]) 
				{
					hasIt=true;
					break;
				}
			}
			if(hasIt==false) return vert;
*/
			return vert;
		}
        }
return -1;
//	CkPrintf("THERE IS A PROBLEM IN TEMPREFINELB 111  pe=%d objArraySize=%d assTasks.size()=%d !!!!!\n",pe,ogr->vertices.size(),assTasks.size());
//        CmiPrintStackTrace(0);
//	CkExit();
}

bool saneFreqNormLds(double *loads, int numProcs)
{
	double tot=0.0;
	for(int i=0;i<numProcs;i++)
	{
		tot+=loads[i];
	}
	double r=numProcs-tot;
	if(r>0.01 || r<-0.01)
	{
		CkPrintf("THere is a problem with LOADs!!! r=%f procs=%d loadSum=%f\n",r,numProcs,tot);
		return false;
	}
	else return true;
}
#endif
void TempAwareRefineLB::work(LDStats* stats)
{
#ifdef TEMP_LDB
////////////////////////////////////////////////////
  numProcs=stats->nprocs();
  numChips=numProcs/logicalCoresPerChip;
  avgChipTemp=new float[numChips];
  if(procFreq!=NULL) delete [] procFreq;
	if(procFreqEffect!=NULL) delete [] procFreqEffect;
//  if(procFreqPtr!=NULL) delete [] procFreqPtr;
  if(procTemp!=NULL) delete [] procTemp;
  if(procFreqNew!=NULL) delete [] procFreqNew;
	if(procFreqNewEffect!=NULL) delete [] procFreqNewEffect;
  if(avgChipTemp!=NULL) delete [] avgChipTemp;

  procFreq = new int[numProcs];
	procFreqEffect = new int[numProcs];
//  procFreqPtr = new int[numProcs];
  procTemp = new float[numProcs];
  procFreqNew = new int[numProcs];
	procFreqNewEffect = new int[numProcs];
  avgChipTemp = new float[numChips];

  for(int i=0;i<numChips;i++) avgChipTemp[i]=0;

  for(int i=0;i<numProcs;i++)
  {
        procFreq[i] = stats->procs[i].pe_speed;
        procTemp[i] = stats->procs[i].pe_temp;
//      procFreqPtr[i] = getProcFreqPtr(freqs,numAvailFreqs,procFreq[i]);
        avgChipTemp[i/logicalCoresPerChip] += procTemp[i];
  }

  for(int i=0;i<numChips;i++) 
  {
        avgChipTemp[i]/=logicalCoresPerChip;
//CkPrintf("---- CHIP#%d has temp=%f ----------\n",i,avgChipTemp[i]);
  }
  for(int i=0;i<numChips;i++)
  {
	int over=0,under=0;
        if(avgChipTemp[i] > MAX_TEMP)
        {
		over=1;
                if(procFreqPtr[i*logicalCoresPerChip]==numAvailFreqs-1)
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[procFreqPtr[j]];
                        CkPrintf("CHIP#%d RUNNING HOT EVEN WITH MIN FREQUENCY!!\n",i);
                }
                else
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++)
                        {
                                if(procFreqPtr[j]<numAvailFreqs-1) procFreqPtr[j]++;
#ifdef MAX_MIN
/// PLEASE COMMENT OUT .. TESTING ONLY
if(i==0) {procFreqPtr[j] = numAvailFreqs-1;/*CkPrintf("C for i:%d\n",j);*/}
//if(i<numChips-1) procFreqPtr[j]=0;
else  procFreqPtr[j]=0;
/////////////////////////
#endif
                                procFreqNew[j] = freqs[procFreqPtr[j]];
                        }
#ifndef ORG_VERSION
                        CkPrintf("!!!!! Chip#%d running HOT shifting from %d to %d temp=%f\n",i,procFreq[i*logicalCoresPerChip],procFreqNew[i*logicalCoresPerChip],avgChipTemp[i]);
#endif
                }
        }
        else
//	if(avgChipTemp[i] < MAX_TEMP-1)
        {
		under=1;
                if(procFreqPtr[i*logicalCoresPerChip]>0)
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++)
                        {
                                if(procFreqPtr[j]>0)
                                        procFreqPtr[j]--;
#ifdef MAX_MIN
/// PLEASE COMMENT OUT .. TESTING ONLY
if(i==0) procFreqPtr[j] = numAvailFreqs-1;
//if(i<numChips-1) procFreqPtr[j]=0;
else  procFreqPtr[j]=0;
/////////////////////////
#endif
                                procFreqNew[j] = freqs[procFreqPtr[j]];
                        }
#ifndef ORG_VERSION
                        CkPrintf("!!!!! Chip#%d running COLD shifting from %d to %d temp=%f\n",i,procFreq[i*logicalCoresPerChip],procFreqNew[i*logicalCoresPerChip],avgChipTemp[i]);
#endif
                }
                else
                {
                        for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[procFreqPtr[j]];
                }
        }
/*
	if(under==0 && over==0) 
	{
		for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[procFreqPtr[j]];
	}
*/
//if(i==5) for(int j=i*c(resPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[numAvailFreqs-1];
//else 
#ifdef ORG_VERSION
for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[0];
#endif
//for(int j=i*logicalCoresPerChip;j<i*logicalCoresPerChip+logicalCoresPerChip;j++) procFreqNew[j] = freqs[0];
  }
//for(int x=0;x<numProcs;x+=logicalCoresPerChip) if(procFreq[x]!=procFreqNew[x]) thisProxy[x].changeFreq(procFreqNew[x]);
//for(int x=0;x<numProcs;x++) CkPrintf("Procs#%d freq %d\n",x,procFreqNew[x]);
////////////////////////////////////////////////////

#ifndef NO_TEMP_LB
  int obj;
  int n_pes = stats->nprocs();

  //  CkPrintf("[%d] RefineLB strategy\n",CkMyPe());

  // RemoveNonMigratable(stats, n_pes);

  // get original object mapping
  int* from_procs = RefinerTemp::AllocProcs(n_pes, stats);
  for(obj=0;obj<stats->n_objs;obj++)  {
    int pe = stats->from_proc[obj];
    from_procs[obj] = pe;
  }
  // Get a new buffer to refine into
	populateEffectiveFreq(numProcs);
  int* to_procs = RefinerTemp::AllocProcs(n_pes, stats);
//  RefinerTemp refiner(1.03,procFreqEffect,procFreqNewEffect,n_pes);  // overload tolerance=1.05
	RefinerTemp refiner(1.03,procFreq,procFreqNew,n_pes);
  refiner.Refine(n_pes, stats, from_procs, to_procs);
  // Save output
	int migs=0;
	int *numMigs = new int[numProcs];
	int totE = 0;
	for(int mm=0;mm<numProcs;mm++) numMigs[mm] = 0;
  for(obj=0;obj<stats->n_objs;obj++) {
      int pe = stats->from_proc[obj];
			numMigs[to_procs[obj]]++;
//stats->objData[obj].objID();
  LDObjData &odata = stats->objData[obj];
	computeInfo *c1 = new computeInfo();
	c1->id = odata.objID();
//if(to_procs[obj]==3) CkPrintf("[%d,%d] going to 3 totE:%d\n",c1->id.getID()[0],c1->id.getID()[1],totE++);//,(stats->objData[obj].objID().getID())[1],totE++);
      if (to_procs[obj] != pe) {
	migs++;
        //if (_lb_args.debug()>=2)  
				{
//          CkPrintf("[%d,%d] Obj %d migrating from %d to %d\n",
//                 c1->id.getID()[0],c1->id.getID()[1],obj,pe,to_procs[obj]);
        }
        stats->to_proc[obj] = to_procs[obj];
      }
  }

	for(int mm=0;mm<numProcs;mm++)
	{
		//CkPrintf("PROC#%d freq:%d objs:%d ----------\n",mm,procFreqNew[mm],numMigs[mm]);
	}
  CkPrintf("TEMPLB INFO: Total Objs:%d migrations:%d time:%f \n",stats->n_objs,migs,CmiWallTimer()-starting);
  fprintf(migFile,"%f %d\n",CmiWallTimer()-starting,migs);
  // Free the refine buffers
  RefinerTemp::FreeProcs(from_procs);
  RefinerTemp::FreeProcs(to_procs);

#endif
//for(int x=0;x<numProcs;x++) CkPrintf("Procs#%d ------- freq %d\n",x,procFreqNew[x]);
/*
for(int x=0;x<numProcs;x+=logicalCoresPerChip) 
{
	if(procFreq[x]!=procFreqNew[x]) 
	{
		CkPrintf("Chaning the freq for PROC#%d\n",x);
		thisProxy[x].changeFreq(procFreqNew[x]);
	}
}
*/
for(int x=0;x<numProcs;x++)
  {
//CkPrintf("--------- Proc#%d %d numProcs=%d\n",x,procFreqNew[x],numProcs);
if(procFreq[x]!=procFreqNew[x]) thisProxy[x].changeFreq(procFreqNew[x]);
}
#endif // TEMP_LDB endif
}
#include "TempAwareRefineLB.def.h"

/*@}*/

