#ifndef MPI_STRATEGY
#define MPI_STRATEGY

#include "ComlibManager.h"

#if CHARM_MPI
#include "mpi.h"
#define MPI_MAX_MSG_SIZE 1000
#define MPI_BUF_SIZE 2000000
char mpi_sndbuf[MPI_BUF_SIZE];
char mpi_recvbuf[MPI_BUF_SIZE];
#endif

class MPIStrategy : public Strategy {
    CharmMessageHolder *messageBuf;
    int messageCount;
    int npes, *pelist;

 public:
    MPIStrategy();
    MPIStrategy(CkMigrateMessage *){}
    MPIStrategy(int npes, int *pelist);

    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    PUPable_decl(MPIStrategy);
};
#endif
