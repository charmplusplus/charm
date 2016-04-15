/**
 * AMPI Test program
 *  Orion Sky Lawlor, olawlor@acm.org, 2003/4/4
 */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "charm.h" /* For CkAbort */

int getRank(void) {
	int rank; 
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	return rank;
}

void testFailed(const char *where) {
	fprintf(stderr,"[%d] MPI TEST FAILED> %s\n",getRank(),where);
	CkAbort("MPI TEST FAILED");
	MPI_Finalize();
	exit(1);
}

void testEqual(int curValue,int goodValue,const char *where) {
	if (goodValue!=curValue) {
		fprintf(stderr,"[%d] %s> expected %d, got %d!\n",
			getRank(),where,goodValue,curValue);
		testFailed(where);
	}
}

void testRange(int curValue,int loBound,int upBound,const char *where) {
	if (curValue<loBound) {
		fprintf(stderr,"[%d] %s> expected at least %d, got %d!\n",
			getRank(),where,loBound,curValue);
		testFailed(where);
	}
	if (curValue>upBound) {
		fprintf(stderr,"[%d] %s> expected at most %d, got %d!\n",
			getRank(),where,upBound,curValue);
		testFailed(where);
	}
}

/// Change this verbosity level to get more printouts:
int verboseLevel=1;
void beginTest(int testLevel, const char *testName) {
	if (testLevel<10) // Block between important tests:
		MPI_Barrier(MPI_COMM_WORLD);
	if (testLevel<=verboseLevel) {
		int rank=getRank();
		if (rank==0 || verboseLevel>10)
			printf("[%d] Testing: %s\n",rank,testName);
	}
}

#define TEST_MPI(routine,args) \
	beginTest(12,"     calling routine " #routine); \
	testEqual(MPI_SUCCESS,routine args, #routine " return code")


/**
 * Test out a bunch of MPI routines for this communicator.
 */
class MPI_Tester {
public:
	int rank,size; //Our rank in, and true size of the communicator
	
	MPI_Tester(MPI_Comm comm,int trueSize=-1);
	void test(void);
	void testMigrate(void);

private: 
	MPI_Comm comm; //Communicator to test

//Little utility routines:
	/// Broadcast this value from this master:
	int bcast(int value,int master) {
		int mValue=-1; if (rank==master) mValue=value;
		TEST_MPI(MPI_Bcast,(&mValue,1,MPI_INT,master,comm));
		return mValue;
	}
	/// Reduce this value to this master:
	int reduce(int value,int master,MPI_Op op) {
		int destValue=-1;
		TEST_MPI(MPI_Reduce,(&value,&destValue,1,MPI_INT,op,master,comm));
		return destValue;
	}
	/// Receive this integer from this source:
	int recv(int source,int tag) {
		int recvVal=-1;
		MPI_Status sts;
		TEST_MPI(MPI_Recv,(&recvVal,1,MPI_INT,source,tag,comm,&sts));
		testEqual(sts.MPI_TAG,tag,"Recv status tag");
		if (source!=MPI_ANY_SOURCE)
			testEqual(sts.MPI_SOURCE,source,"Recv status source");
		testEqual(sts.MPI_COMM,comm,"Recv status comm");
		/* not in standard: testEqual(1,sts.MPI_LENGTH,"Recv status length");*/
		return recvVal;
	}
	
	/// Check if anything is available on the network--
	///  if so, it's a leftover message and is very bad.
	void drain(void);
};

MPI_Tester::MPI_Tester(MPI_Comm comm_,int trueSize)
	:comm(comm_)
{ /// Communicator size:
	beginTest(5,"Comm_size/rank");
	TEST_MPI(MPI_Comm_size,(comm,&size));
	if (trueSize!=-1) testEqual(size,trueSize,"MPI_Comm_size value");
	TEST_MPI(MPI_Comm_rank,(comm,&rank));
	testRange(rank,0,size,"MPI_Comm_rank value");
}

// "Hash" function of three integers.
//  We want fn(a,b,c) != fn(b,a,c) != fn(a,c,b) etc.
inline int fn(int a,int b,int c) {
	return (a+1000*b+1000000*c);
}

void MPI_Tester::test(void)
{ 
	MPI_Request req;
	MPI_Status  status;
	int i;
/// Send and receive:
	beginTest(3,"Send/recv");
	// Send to the next guy in a ring:
	int next=(rank+1)%size;
	int prev=(rank-1+size)%size;
	int tag=12387, recvVal=-1;
	
	// Forward around ring:
	TEST_MPI(MPI_Send,(&rank,1,MPI_INT,next,tag,comm));
	testEqual(recv(prev,tag),prev,"Received rank (using prev as source)");
	
	// Forward around ring:
        if (size >= 2) {
	  if (rank == 0) {
	    TEST_MPI(MPI_Issend,(&rank,1,MPI_INT,next,tag,comm, &req));
	    TEST_MPI(MPI_Wait, (&req, &status));
	  }
	  else if (rank == 1)
	    testEqual(recv(prev,tag),prev,"Received rank (using prev as source)");
	}

	// Simultanious forward and backward:
	TEST_MPI(MPI_Send,(&rank,1,MPI_INT,next,tag,comm));
	TEST_MPI(MPI_Send,(&rank,1,MPI_INT,prev,tag,comm));
	testEqual(recv(next,tag),next,"Received rank (specifying next as source)");
	testEqual(recv(prev,tag),prev,"Received rank (specifying prev as source)");

/// Collective operations:
	beginTest(4,"Barrier");
	TEST_MPI(MPI_Barrier,(comm));
	
	int master=7%size;
	beginTest(3,"Broadcast");
	const int bcastValue=123;
	testEqual(bcast(bcastValue,0),bcastValue,"Broadcast integer from 0");
	testEqual(bcast(bcastValue,master),bcastValue,"Broadcast integer from master");
	testEqual(bcast(size,0),size,"Broadcast comm size");
	testEqual(bcast(master,master),master,"Broadcast master rank from master");
	
	beginTest(3,"Reduction");
	int sumSize=reduce(1,master,MPI_SUM);
	if (rank==master) testEqual(sumSize,size,"Reduce sum integer 1");
	int prodSize=reduce(1,master,MPI_PROD);
	if (rank==master) testEqual(prodSize,1,"Reduce product integer 1");
	
	beginTest(3,"Scatter");
	const int scatter_key=17;
	int *sendV=new int[size], *recvV=new int[size];
	for (i=0;i<size;i++) 
		sendV[i]=(rank==master)?fn(i,0,scatter_key):-1;
	int recvI=-1, sendI=-1;
	TEST_MPI(MPI_Scatter, (sendV,1,MPI_INT, &recvI,1,MPI_INT, master, comm));
	testEqual(recvI,fn(rank,0,scatter_key),"Scatter results");
	
	beginTest(3,"Scatterv");
	const int scatterv_key=18;
	int *sizeV=new int[size], *offsetV=new int[size];
	int offset=0;
	if (rank==master)
		for (i=0;i<size;i++) {
			sendV[i]=fn(i,0,scatterv_key);
			sizeV[i]=1; offsetV[i]=offset; offset+=sizeV[i]; 
		}
	recvI=-1;
	TEST_MPI(MPI_Scatterv, (sendV,sizeV,offsetV,MPI_INT, &recvI,1,MPI_INT, master, comm));
	testEqual(recvI,fn(rank,0,scatterv_key),"Scatterv results");
	
	beginTest(3,"Gather");
	const int gather_key=19;
	for (i=0;i<size;i++) recvV[i]=-1;
	sendI=fn(rank,0,gather_key);
	TEST_MPI(MPI_Gather, (&sendI,1,MPI_INT, recvV,1,MPI_INT, master, comm));
	if (rank==master)
		for (i=0;i<size;i++) {
			sendV[i] = fn(i,0,gather_key);
			testEqual(recvV[i],sendV[i],"Gather results");
		}
	
	// FIXME: add gatherv
	
	beginTest(3,"Alltoall");
	const int alltoall_key=21;
	for (i=0;i<size;i++) {
		sendV[i]=fn(rank,i,alltoall_key); //Everybody sends out their own rank
		recvV[i]=-1;
	}
	TEST_MPI(MPI_Alltoall, (sendV,1,MPI_INT, recvV,1,MPI_INT, comm));
	for (i=0;i<size;i++) testEqual(recvV[i],fn(i,rank,alltoall_key),"Alltoall results");
	
	drain();
	
	// FIXME: add alltoallv
	
	delete[] sendV; delete[] recvV;
	delete[] sizeV; delete[] offsetV;
}

void MPI_Tester::drain(void) {
	MPI_Status sts;
	int flag=0, flagSet=0;
	do {
		TEST_MPI(MPI_Iprobe,(MPI_ANY_SOURCE,MPI_ANY_TAG,comm,&flag,&sts));
		if (flag) {
			int len; MPI_Get_count(&sts,MPI_BYTE,&len);
			CkError("FATAL ERROR> Leftover AMPI message: src=%d, dest=%d, tag=%d, comm=%d, length=%d\n",
				sts.MPI_SOURCE, rank, sts.MPI_TAG, sts.MPI_COMM, len);
			char *msg=new char[len];
			MPI_Recv(msg,len,MPI_BYTE, sts.MPI_SOURCE, sts.MPI_TAG, sts.MPI_COMM, &sts);
			flagSet=1;
		}
	} while (flag==1);
	MPI_Barrier(comm);
	if (flagSet) CkAbort("Leftover messages in AMPI queues!\n");
}

void MPI_Tester::testMigrate(void) {
	beginTest(2,"Migration");
	int srcPe=CkMyPe();
    MPI_Info hints;

    MPI_Info_create(&hints);
    MPI_Info_set(hints, "ampi_load_balance", "sync");
	
	// Before migrating, send a message to the next guy:
	//    this tests out migration with pending messages
	int dest=(rank+1)%size;
	int tag=8383210;
	TEST_MPI(MPI_Send,(&dest,1,MPI_INT, dest,tag,comm));
	TEST_MPI(MPI_Barrier,(comm));
	
	AMPI_Migrate(hints);
	
	TEST_MPI(MPI_Barrier,(comm));
	int recv=-1; MPI_Status sts;
	TEST_MPI(MPI_Recv,(&recv,1,MPI_INT, MPI_ANY_SOURCE,tag,comm, &sts));
	if (recv!=rank) CkAbort("Message corrupted during migration!\n");
	
	int destPe=CkMyPe();
	if (srcPe!=destPe) CkPrintf("[%d] migrated from %d to %d\n",
				rank,srcPe,destPe);
}

int main(int argc,char **argv)
{
	int nLoop=4;
	MPI_Init(&argc,&argv);
	if (argc>1) verboseLevel=atoi(argv[1]);
	if (argc>2) nLoop=atoi(argv[2]);
	MPI_Comm comm=MPI_COMM_WORLD;
	MPI_Tester masterTester(comm);
	
for (int loop=0;loop<nLoop;loop++) {
	beginTest(1,"Testing...");
	if (1) 
	{//Try out MPI_COMM_WORLD:
		beginTest(2,"Communicator = MPI_COMM_WORLD");
		masterTester.test();
	}
	
	if (1) 
	{//Try out MPI_COMM_SELF:
		beginTest(2,"Communicator = MPI_COMM_SELF");
		MPI_Tester selfTester(MPI_COMM_SELF);
		selfTester.test();
	}
	
	if (1) 
	{ //Split the world in two and retest:
		beginTest(2,"Communicator Split");
		MPI_Comm split;
		int nPieces=2+loop;
		int myColor=(masterTester.rank*nPieces)/masterTester.size;
		TEST_MPI(MPI_Comm_split,(comm,myColor,0,&split));
		MPI_Tester splitTester(split);
		splitTester.test();
	}

	if(1)
	  { // Test construction of group and new communicator 
	    int rank;
	    int np;
	    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	    MPI_Comm_size( MPI_COMM_WORLD, &np );

	    if(np >= 4){
	      int groupranks[] = {0,2,3};
	      MPI_Comm newcomm;
	      MPI_Group mpi_base_group, new_grp;
	      MPI_Comm_group( MPI_COMM_WORLD, &mpi_base_group );
	      MPI_Group_incl( mpi_base_group, 3, groupranks, &new_grp );
	      MPI_Comm_create(MPI_COMM_WORLD, new_grp, &newcomm );
	      
	      // Verify the correct ranks are in the new communicator
	      if ( newcomm == MPI_COMM_NULL ){
		if(rank== 0 || rank== 2 || rank== 3 )
		  testFailed("Testing construction of group and new communicator");
	      } else {
		if(rank == 1 || rank > 3)
		  testFailed("Testing construction of group and new communicator");		
	      }
	    }
	  }
	
	
	if (1 && loop!=nLoop-1) 
	  masterTester.testMigrate();
 }
 
	if (getRank()==0) CkPrintf("All tests passed\n");
	MPI_Finalize();
	return 0;
}
