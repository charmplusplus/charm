/**
   @addtogroup ConvComlibRouter
   @{
   @file 
   @brief Grid (3d grid) based router
   @Author Sameer Kumar
*/





#include "3dgridrouter.h"
//#define NULL 0

#define gmap(pe) {if (gpes) pe=gpes[pe];}

/**The only communication op used. Modify this to use
 ** vector send */
#if CMK_COMLIB_USE_VECTORIZE
#define GRIDSENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe)  \
  	{int len;\
	PTvectorlist newmsg;\
        newmsg=PeGrid->ExtractAndVectorize(kid, u1, knpe, kpelist);\
	if (newmsg) {\
	  CmiSetHandler(newmsg->msgs[0], khndl);\
          CmiSyncVectorSendAndFree(knextpe, -newmsg->count, newmsg->sizes, newmsg->msgs);\
        }\
	else {\
	  SendDummyMsg(kid, knextpe, u2);\
	}\
}
#else
#define GRIDSENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe)  \
  	{int len;\
	char * newmsg;\
        newmsg=PeGrid->ExtractAndPack(kid, u1, knpe, kpelist, &len);\
	if (newmsg) {\
	  CmiSetHandler(newmsg, khndl);\
          CmiSyncSendAndFree(knextpe, len, (char *)newmsg);\
        }\
	else {\
	  SendDummyMsg(kid, knextpe, u2);\
	}\
}
#endif

#define ROWLEN COLLEN

#define RowLen(pe) ColLen3D(pe)
#define PELISTSIZE ((ROWLEN-1)/sizeof(int)+1)

inline int ColLen3D(int npes)
{
    int len= (int)cubeRoot((double)npes);
    //    ComlibPrintf("%d:collen len = %d\n", CkMyPe(), len);
    if (npes > (len * len * len)) len++;
    return(len);
}

inline int Expect1(int gpe, int gnpes)
{
    int i, len=ColLen3D(gnpes);
    int pe = gpe % (len * len);
    
    int npes = len * len;
    if((gnpes - 1)/(len * len) == gpe / (len * len))
        npes = ((gnpes - 1) % (len*len)) + 1;
    
    for (i=len-1;i>=0;i--) {
        int myrow=pe/len;
        int toprep=i*len;
        int offset=pe-myrow*len;
        if ((toprep+offset) <= (npes-1)) return(i+1);
    }
    return 0;
    //return(len);
}

inline int Expect2(int gpe, int gnpes) {
    int len=RowLen(gnpes);
    int myplane = gpe / (len * len);
    int lastplane = (gnpes - 1)/(len * len);
    
    if(myplane < lastplane)
        return len;

    int pe = gpe % (len * len);
    int myrow = pe / len;
    int lastrow = ((gnpes - 1) % (len * len)) / len;
    
    if (myrow < lastrow)
        return len;
    
    int ret = ((gnpes - 1) % (len * len)) - myrow * len + 1;
    if(ret < 0)
        ret = 0;
    return ret;
}
    

inline int LPMsgExpect(int gpe, int gnpes)
{
    int i;
    int row = RowLen(gnpes);
    int col = ColLen3D(gnpes);
    int len = (int)ceil(((double)gnpes) / (row * col));

    for (i=len-1;i>=0;i--) {
        int toprep=i*(row * col);
        
        if ((toprep + (gpe % (row * col))) <= (gnpes-1)) return(i+1);
    }
    return(len);
}

/****************************************************
 * Preallocated memory=P ints + MAXNUMMSGS msgstructs
 *****************************************************/
D3GridRouter::D3GridRouter(int n, int me, Strategy *parent) : Router(parent)
{
    ComlibPrintf("PE=%d me=%d NUMPES=%d\n", CkMyPe(), me, n);
    
    NumPes=n;
    MyPe=me;
    gpes=NULL;

    COLLEN=ColLen3D(NumPes);

    recvExpected[0] = 0;
    recvExpected[1] = 0;
    routerStage = 0;
    
    int myrow = (MyPe % (ROWLEN * COLLEN)) / COLLEN;
    int myrep = myrow * COLLEN + MyPe - (MyPe % (ROWLEN * COLLEN));
    int numunmappedpes=myrep+ROWLEN-NumPes;
    int nummappedpes=ROWLEN;
    
    if (numunmappedpes >0) {
	nummappedpes=NumPes-myrep;
	int i=NumPes+MyPe-myrep;
	while (i<myrep+ROWLEN) {
            recvExpected[0] += Expect1(i, NumPes);
            i+=nummappedpes;
	}
    }
    
    if((NumPes % (COLLEN * ROWLEN) != 0) && ((NumPes - 1)/(ROWLEN*COLLEN) - MyPe/(ROWLEN*COLLEN) == 1)){
        if(myrep + ROWLEN * COLLEN >= NumPes) 
            recvExpected[0] += Expect1(MyPe + ROWLEN*COLLEN, NumPes);
        
        if(MyPe + ROWLEN * COLLEN >= NumPes) 
            recvExpected[1] += Expect2(MyPe + ROWLEN*COLLEN, NumPes);
        ComlibPrintf("%d: here\n");
    }

    recvExpected[0] += Expect1(MyPe, NumPes);
    recvExpected[1] += Expect2(MyPe, NumPes);

    LPMsgExpected = LPMsgExpect(MyPe, NumPes);
    //ComlibPrintf("%d LPMsgExpected=%d\n", MyPe, LPMsgExpected);
    
    PeGrid = new PeTable(/*CkNumPes()*/NumPes);
    
    nplanes = (int)ceil(((double)NumPes) / (ROWLEN * COLLEN));

    oneplane = new int*[COLLEN];
    psize = new int[COLLEN];
    for(int count = 0; count < COLLEN; count ++) {
        oneplane[count] = new int[nplanes * ROWLEN];

        int idx = 0;
        
        for (int j=0;j< ROWLEN;j++) 
            for(int k = 0; k < nplanes; k++) {
                int dest = count * ROWLEN + j + ROWLEN * COLLEN * k;
                if(dest < NumPes) {
                    oneplane[count][idx++] = dest;
                }
                else break;
            }
        psize[count] = idx;
    }

    zline = new int[nplanes];
    
    InitVars();
    ComlibPrintf("%d:%d:COLLEN=%d, ROWLEN=%d, recvexpected=%d,%d\n", CkMyPe(), MyPe, COLLEN, ROWLEN, recvExpected[0], recvExpected[1]);
}

D3GridRouter::~D3GridRouter()
{
    delete PeGrid;
    delete[] zline;
    for(int count = 0; count < COLLEN; count ++)
        delete[] oneplane[count];
    
    delete[] oneplane;
}

void D3GridRouter :: InitVars()
{
    recvCount[0]=0;
    recvCount[1]=0;
    
    LPMsgCount=0;
}

void D3GridRouter::NumDeposits(comID, int num)
{
}

void D3GridRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
    int npe=NumPes;
    int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
    for (int i=0;i<npe;i++) destpes[i]=i;
    EachToManyMulticast(id, size, msg, npe, destpes, more);
}

void D3GridRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{
    int i;
    
    //Buffer the message
    if (size) {
  	PeGrid->InsertMsgs(numpes, destpes, size, msg);
    }
    
    if (more) return;

    routerStage = 0;
    ComlibPrintf("All messages received %d %d\n", CkMyPe(), COLLEN);
    
    //Send the messages
    int firstproc = MyPe - (MyPe % (ROWLEN * COLLEN));
    for (i=0;i<COLLEN;i++) {
        
        ComlibPrintf("ROWLEN = %d, COLLEN =%d first proc = %d\n", 
                     ROWLEN, COLLEN, firstproc);
        
        //int MYROW = (MyPe % (ROWLEN * COLLEN))/ROWLEN;
        int nextrowrep = firstproc + i*ROWLEN;
        int nextpe = (MyPe % (ROWLEN * COLLEN)) % ROWLEN + nextrowrep;
        int nummappedpes=NumPes-nextrowrep;
        
        if (nummappedpes <= 0) { // looks for nextpe in the previous plane
            nextpe -= ROWLEN * COLLEN;
            if(nextpe < 0)
                continue;
        }

        if (nextpe >= NumPes) {
            int mm=(nextpe-NumPes) % nummappedpes;
            nextpe=nextrowrep+mm;
        }
        
        if (nextpe == MyPe) {
            ComlibPrintf("%d calling recv directly\n", MyPe);
            recvCount[0]++;
            RecvManyMsg(id, NULL);
            continue;
        }
        
        //ComlibPrintf("nummappedpes = %d, NumPes = %d, nextrowrep = %d, nextpe = %d, mype = %d\n", nummappedpes, NumPes, nextrowrep,  nextpe, MyPe);
        
        gmap(nextpe);
        ComlibPrintf("sending to column %d and dest %d in %d\n", i, nextpe, CkMyPe());
        GRIDSENDFN(id, 0, 0, psize[i], oneplane[i],CkpvAccess(RouterRecvHandle), nextpe); 
    }
}

void D3GridRouter::RecvManyMsg(comID id, char *msg)
{
    ComlibPrintf("%d recvcount=%d,%d recvexpected = %d,%d\n", MyPe, recvCount[0],  recvCount[1], recvExpected[0],recvExpected[1]);
    int stage = 0;
    if (msg) {
        stage = PeGrid->UnpackAndInsert(msg);
        recvCount[stage]++;
    }
    
    if ((recvCount[0] == recvExpected[0]) && (routerStage == 0)){
        routerStage = 1;
        int myrow = (MyPe % (ROWLEN * COLLEN)) / COLLEN;
        int myrep = myrow*ROWLEN + MyPe - (MyPe % (ROWLEN * COLLEN));
        for (int i=0;i<ROWLEN;i++) {
            int nextpe = myrep + i;
            
            //if (nextpe >= NumPes || nextpe==MyPe) continue;
            
            if(nextpe == MyPe) {
                recvCount[1]++;
                RecvManyMsg(id, NULL);
                continue;
            }
            
            if(nextpe >= NumPes) {
                nextpe -= ROWLEN * COLLEN;
                if(nextpe < 0)
                    continue;
            }

            int *pelist = zline;
            int k = 0;
            
            //ComlibPrintf("recv:myrow = %d, nplanes = %d\n", myrow, nplanes);
            //ComlibPrintf("recv:%d->%d:", MyPe, nextpe);
            for(k = 0; k < nplanes; k++) {
                int dest = myrow * ROWLEN + i + ROWLEN * COLLEN * k;
                //ComlibPrintf("%d,", dest);
                if(dest >= NumPes)
                    break;
                zline[k] = dest;
            }
            //ComlibPrintf(")\n");

            ComlibPrintf("Before gmap %d\n", nextpe);
            
            gmap(nextpe);
            
            //ComlibPrintf("After gmap %d\n", nextpe);
            ComlibPrintf("%d:sending recv message %d %d\n", MyPe, nextpe, myrep);
            GRIDSENDFN(id, 1, 1, k, pelist, CkpvAccess(RouterRecvHandle), nextpe);
        }
    }
    
    if((recvCount[1] == recvExpected[1]) && (routerStage == 1)){
        routerStage = 2;
        for (int k=0; k < nplanes; k++) {
            int nextpe = (MyPe % (ROWLEN * COLLEN)) + k * ROWLEN * COLLEN;

            if (nextpe >= NumPes || nextpe==MyPe) continue;
            
            int gnextpe = nextpe;
            int *pelist = &gnextpe;
            
            ComlibPrintf("Before gmap %d\n", nextpe);
            
            gmap(nextpe);
            
            //ComlibPrintf("After gmap %d\n", nextpe);
            
            ComlibPrintf("%d:sending proc message %d %d\n", MyPe, nextpe, nplanes);
            GRIDSENDFN(id, 2, 2, 1, pelist, CkpvAccess(RouterProcHandle), nextpe);
        }
        LocalProcMsg(id);
    }
}

void D3GridRouter::DummyEP(comID id, int stage)
{
    if (stage == 2) {
        ComlibPrintf("%d dummy calling lp\n", MyPe);
	LocalProcMsg(id);
    }
    else {
        recvCount[stage]++;
	ComlibPrintf("%d dummy calling recv\n", MyPe);
  	RecvManyMsg(id, NULL);
    }
}

void D3GridRouter:: ProcManyMsg(comID id, char *m)
{
    PeGrid->UnpackAndInsert(m);
    ComlibPrintf("%d proc calling lp\n", MyPe);
    LocalProcMsg(id);
}

void D3GridRouter:: LocalProcMsg(comID id)
{
    ComlibPrintf("%d local procmsg called\n", MyPe);
    
    LPMsgCount++;
    PeGrid->ExtractAndDeliverLocalMsgs(MyPe, container);
    
    if (LPMsgCount==LPMsgExpected) {
	PeGrid->Purge();
	InitVars();
        routerStage = 0;
        ComlibPrintf("%d:Round Done\n", CkMyPe());
	Done(id);
    }
}

void D3GridRouter :: SetMap(int *pes)
{
    gpes=pes;
}

/*@}*/
