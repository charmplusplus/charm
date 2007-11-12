#include "reduction.h"

/*These aren't readonlies, because they're only used on PE 0*/
static CProxy_reductionArray redArr;
static CProxy_reductionGroup redGrp;
static int nFinished;

class reductionInfo {
public:
	int properValue;
	reductionInfo(int v) :properValue(v) {}
	void check(void *data,int size) {
		int *v=(int *)data;
		if (size!=2*sizeof(int))
			CkAbort("Unexpected-size reduction result!");
		if (v[0]!=properValue)
			CkAbort("Corrupted first field!");
		if (v[1]!=0 && v[1]!=(2*properValue))
			CkAbort("Corrupted second field!");
	}
};

static void reductionClient(void *redInfo,int size,void *data) {
	reductionInfo *info=(reductionInfo *)redInfo;
	info->check(data,size);
	nFinished++;
	if (nFinished%4 == 0) megatest_finish();
}

void reduction_moduleinit(void) {
	nFinished=0;
	const int numElements = 5;
	redArr=CProxy_reductionArray::ckNew(numElements);
	redArr.setReductionClient(reductionClient,new reductionInfo(numElements));
	redGrp=CProxy_reductionGroup::ckNew();
	redGrp.setReductionClient(reductionClient,new reductionInfo(CkNumPes()));
}

void reduction_init(void)
{
	redArr.start();
	redGrp.start();
}

void reductionArray::start(void) {
	int i[2];
	i[0]=1; //Sum=numElements
	i[1]=0; //Sum=0
	contribute(sizeof(i),&i,CkReduction::sum_int);
	i[1]=2; //Sum=2*numElements
	contribute(sizeof(i),&i,CkReduction::sum_int);
	if (0) //Migrate to the next processor
		ckMigrate((CkMyPe()+1)%CkNumPes());
}
void reductionGroup::start(void) {
	int i[2];
	i[0]=1;
	i[1]=0;
	contribute(sizeof(i),&i,CkReduction::sum_int);	
	i[1]=2;	
	contribute(sizeof(i),&i,CkReduction::sum_int);
}

MEGATEST_REGISTER_TEST(reduction,"olawlor",1)
#include "reduction.def.h"
