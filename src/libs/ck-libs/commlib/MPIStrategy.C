#include "MPIStrategy.h"
#include "mpi.h"

#if CHARM_MPI
MPI_Comm groupComm;
MPI_Group group, groupWorld;
#endif

MPIStrategy::MPIStrategy(){
    messageBuf = NULL;
    messageCount = 0;
    npes = CkNumPes();
    pelist = NULL;
}

MPIStrategy::MPIStrategy(int npes, int *pelist){
    messageBuf = NULL;
    messageCount = 0;

    this->npes = npes;
    this->pelist = pelist;
}

void MPIStrategy::insertMessage(CharmMessageHolder *cmsg){
    cmsg->next = messageBuf;
    messageBuf = cmsg;    
}

void MPIStrategy::doneInserting(){
#if CHARM_MPI
    ComlibPrintf("[%d] In MPI strategy\n", CkMyPe());
    
    CharmMessageHolder *cmsg = messageBuf;
    char *buf_ptr = mpi_sndbuf;
    
    //if(npes == 0)
    //  npes = CkNumPes();
    
    for(count = 0; count < npes; count ++) {
        ((int *)buf_ptr)[0] = 0;
        buf_ptr += MPI_MAX_MSG_SIZE;
    }
    
    buf_ptr = mpi_sndbuf;
    for(count = 0; count < messageCount; count ++) {
        if(npes < CkNumPes()) {
            ComlibPrintf("[%d] Copying data to %d and rank %d\n", 
                         cmsg->dest_proc, procMap[cmsg->dest_proc]);
            buf_ptr = mpi_sndbuf + MPI_MAX_MSG_SIZE * procMap[cmsg->dest_proc];  
        }
        else
            buf_ptr = mpi_sndbuf + MPI_MAX_MSG_SIZE * cmsg->dest_proc; 
        
        char * msg = cmsg->getCharmMessage();
        envelope * env = UsrToEnv(msg);
        
        ((int *)buf_ptr)[0] = env->getTotalsize();
        
        ComlibPrintf("[%d] Copying message\n", CkMyPe());
        memcpy(buf_ptr + sizeof(int), (char *)env, env->getTotalsize());
        
        ComlibPrintf("[%d] Deleting message\n", CkMyPe());
        CmiFree((char *) env);
        CharmMessageHolder *prev = cmsg;
        cmsg = cmsg->next;
        delete prev;
    }
    
    //ComlibPrintf("[%d] Calling Barrier\n", CkMyPe());
    //PMPI_Barrier(groupComm);
    
    ComlibPrintf("[%d] Calling All to all\n", CkMyPe());
    PMPI_Alltoall(mpi_sndbuf, MPI_MAX_MSG_SIZE, MPI_CHAR, mpi_recvbuf, 
                  MPI_MAX_MSG_SIZE, MPI_CHAR, groupComm);
    
    ComlibPrintf("[%d] All to all finished\n", CkMyPe());
    buf_ptr = mpi_recvbuf;
    for(count = 0; count < npes; count ++) {
        int recv_msg_size = ((int *)buf_ptr)[0];
        char * recv_msg = buf_ptr + sizeof(int);
        
        if((recv_msg_size > 0) && recv_msg_size < MPI_MAX_MSG_SIZE) {
            ComlibPrintf("[%d] Receiving message of size %d\n", CkMyPe(), 
                         recv_msg_size);
            CmiSyncSend(CmiMyPe(), recv_msg_size, recv_msg);
        }
        buf_ptr += MPI_MAX_MSG_SIZE;
    }
#endif
}

void MPIStrategy::pup(PUP::er &p) {
    p | messageCount;
    p | npes; 
       
    if(p.isUnpacking())
        pelist = new int[npes];
    p(pelist , npes);

    messageBuf = NULL;
    
    if(p.isUnpacking()){
#if CHARM_MPI
        if(npes < CkNumPes()){
            PMPI_Comm_group(MPI_COMM_WORLD, &groupWorld);
            PMPI_Group_incl(groupWorld, npes, pelist, &group);
            PMPI_Comm_create(MPI_COMM_WORLD, group, &groupComm);
        }
        else groupComm = MPI_COMM_WORLD;
#endif
    }
}

PUPable_def(MPIStrategy);
