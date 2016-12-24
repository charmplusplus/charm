#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"

void intercomm_bcast_test(MPI_Comm myFirstComm, int global_rank, int root) {
    int data;
    if (global_rank == 0) {
        data = 42;
        MPI_Bcast(&data, 1, MPI_INT, MPI_ROOT, myFirstComm);
    }
    else if (global_rank%2 == 0) {
        MPI_Bcast(&data, 1, MPI_INT, MPI_PROC_NULL, myFirstComm);
    }
    else {
        MPI_Bcast(&data, 1, MPI_INT, root, myFirstComm);
    }

    printf("[%d] data : %d\n", global_rank, data);
}

void intercomm_ibcast_test(MPI_Comm myFirstComm, int global_rank, int root) {
    char msg[20];
    MPI_Status sts;
    MPI_Request req = MPI_REQUEST_NULL;

    if (global_rank == 0) {//root
        strcpy(msg, "Hello world");
        MPI_Ibcast(msg, 12, MPI_CHAR, MPI_ROOT, myFirstComm, &req);
        MPI_Wait(&req, &sts);
        strcpy(msg, "What will happen?");
    }
    else if (global_rank % 2 == 0) {//local group
        MPI_Ibcast(msg, 12, MPI_CHAR, MPI_PROC_NULL, myFirstComm, &req);
        MPI_Wait(&req, &sts);
    }
    else {//remote group
        MPI_Ibcast(msg, 12, MPI_CHAR, root, myFirstComm, &req);
        MPI_Wait(&req, &sts);
    }

    printf("[%d] Message : %s\n", global_rank, msg);

}

void intercomm_barrier_test(MPI_Comm myFirstComm, int global_rank) {
    if (global_rank % 2 == 0) {
        MPI_Barrier(myFirstComm);
        printf("[%d]Local group resumed\n", global_rank);
    }
    else {
        printf("[%d]Remote group ranks doing work before barrier\n", global_rank);
        MPI_Barrier(myFirstComm);
    }
}

void intercomm_ibarrier_test(MPI_Comm myFirstComm, int global_rank) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Status sts;

    if (global_rank % 2 == 0) {
        MPI_Ibarrier(myFirstComm, &req);
        MPI_Wait(&req, &sts);
        printf("[%d]Local group resumed\n", global_rank);
    }
    else {
        printf("[%d]Remote group ranks doing work before barrier\n", global_rank);
        MPI_Ibarrier(myFirstComm, &req);
        MPI_Wait(&req, &sts);
    }
}

void intercomm_gather_test(MPI_Comm myFirstComm, int global_rank, int root) {
    int sendarray[5];
    int *rbuf, remoteSize;
    MPI_Comm_remote_size(myFirstComm, &remoteSize);
    rbuf = (int*) malloc(remoteSize*5*sizeof(int));

    if (global_rank == 0) {//root
        MPI_Gather(sendarray, 5, MPI_INT, rbuf, 5, MPI_INT, MPI_ROOT, myFirstComm);
        for (int i = 0; i < remoteSize*5; i++) {
            printf(" %d", rbuf[i]);
        }
        printf("\n");
    }
    else if (global_rank%2 == 0) {//local group
        MPI_Gather(sendarray, 5, MPI_INT, rbuf, 5, MPI_INT, MPI_PROC_NULL, myFirstComm);
    }
    else if (global_rank%2 == 1) {//remote group
        for (int i = 0; i < 5; i++)
            sendarray[i] = 42 + global_rank;

        MPI_Gather(sendarray, 5, MPI_INT, rbuf, 5, MPI_INT, root, myFirstComm);
    }
}

void intercomm_igather_test(MPI_Comm myFirstComm, int global_rank, int root) {
    int sendarray[5];
    int *rbuf, remoteSize;
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Status sts;
    MPI_Comm_remote_size(myFirstComm, &remoteSize);
    rbuf = (int*) malloc(remoteSize*5*sizeof(int));

    if (global_rank == 0) {//root
        MPI_Igather(sendarray, 5, MPI_INT, rbuf, 5, MPI_INT, MPI_ROOT, myFirstComm, &req);
        MPI_Wait(&req, &sts);
        for (int i = 0; i < remoteSize*5; i++) {
            printf(" %d", rbuf[i]);
        }
        printf("\n");
    }
    else if (global_rank%2 == 0) {//local group
        MPI_Igather(sendarray, 5, MPI_INT, rbuf, 5, MPI_INT, MPI_PROC_NULL, myFirstComm, &req);
    }
    else if (global_rank%2 == 1) {//remote group
        for (int i = 0; i < 5; i++)
            sendarray[i] = 42 + global_rank;

        MPI_Igather(sendarray, 5, MPI_INT, rbuf, 5, MPI_INT, root, myFirstComm, &req);
    }
}

void intercomm_gatherv_test(MPI_Comm myFirstComm, int global_rank, int root, int non_blocking) {
    int local_rank, remote_size;
    MPI_Comm_rank(myFirstComm, &local_rank);
    // each remote rank sends its local_rank number of elements
    int *sendbuf = (int*)malloc(local_rank*sizeof(int));
    MPI_Comm_remote_size(myFirstComm, &remote_size);
    MPI_Status sts;
    MPI_Request req = MPI_REQUEST_NULL;

    int recvbuf_size = 0;
    int *recv_counts = (int*)malloc(remote_size*sizeof(int));
    int *displacements = (int*)malloc(remote_size*sizeof(int));
    for (int i = 0; i < remote_size; i++) {
        recvbuf_size += i;
        recv_counts[i] = i;
        if (i == 0) {
            displacements[i] = 0;
        }
        else {
            displacements[i] = displacements[i-1] + recv_counts[i-1];
        }
    }
    int *recvbuf = (int*)malloc(recvbuf_size*sizeof(int));

    if (global_rank == 0) {//root for gatherv
        if (non_blocking) {
            MPI_Igatherv(sendbuf, local_rank, MPI_INT, recvbuf, recv_counts, displacements, MPI_INT, MPI_ROOT, myFirstComm, &req);
            MPI_Wait(&req, &sts);
            printf("[root] Testing intercomm igatherv:\n");
        }
        else {
            MPI_Gatherv(sendbuf, local_rank, MPI_INT, recvbuf, recv_counts, displacements, MPI_INT, MPI_ROOT, myFirstComm);
            printf("[root] Testing intercomm gatherv:\n");
        }
        // print recvbuf here
        for (int i = 0; i < recvbuf_size; i++) {
            printf(" %d", recvbuf[i]);
        }
        printf("\n");
    }
    else if (global_rank%2 == 0) {//local group
        if (non_blocking) {
            MPI_Igatherv(sendbuf, local_rank, MPI_INT, recvbuf, recv_counts, displacements, MPI_INT, MPI_PROC_NULL, myFirstComm, &req);
        }
        else {
            MPI_Gatherv(sendbuf, local_rank, MPI_INT, recvbuf, recv_counts, displacements, MPI_INT, MPI_PROC_NULL, myFirstComm);
        }
    }
    else if (global_rank%2 == 1) {//remote group
        for (int i = 0; i < local_rank; i++) {
            sendbuf[i] = local_rank + 42;
        }
        if (non_blocking) {
            MPI_Igatherv(sendbuf, local_rank, MPI_INT, recvbuf, recv_counts, displacements, MPI_INT, root, myFirstComm, &req);
        }
        else {
            MPI_Gatherv(sendbuf, local_rank, MPI_INT, recvbuf, recv_counts, displacements, MPI_INT, root, myFirstComm);
        }
    }

}

void intercomm_scatter_test(MPI_Comm myFirstComm, int global_rank, int root, int non_blocking) {
    int remote_size;
    MPI_Comm_remote_size(myFirstComm, &remote_size);
    int sendcount = 5;
    int *sendbuf = (int*)malloc(sendcount*remote_size*sizeof(int));
    int *recvbuf = (int*)malloc(sendcount*sizeof(int));
    MPI_Status sts;
    MPI_Request req = MPI_REQUEST_NULL;

    if (global_rank == 0) {
        int idx = 0;
        for (int i = 0; i < remote_size; i++) {
            for (int j = 0; j < sendcount; j++) {
                sendbuf[idx] = i;
                idx++;
            }
        }

        if (non_blocking)
            MPI_Iscatter(sendbuf, sendcount, MPI_INT, recvbuf, sendcount, MPI_INT, MPI_ROOT, myFirstComm, &req);
        else
            MPI_Scatter(sendbuf, sendcount, MPI_INT, recvbuf, sendcount, MPI_INT, MPI_ROOT, myFirstComm);
    }
    else if (global_rank%2 == 0) {
        if (non_blocking)
            MPI_Iscatter(sendbuf, sendcount, MPI_INT, recvbuf, sendcount, MPI_INT, MPI_PROC_NULL, myFirstComm, &req);
        else
            MPI_Scatter(sendbuf, sendcount, MPI_INT, recvbuf, sendcount, MPI_INT, MPI_PROC_NULL, myFirstComm);
    }
    else {
        if (non_blocking) {
            MPI_Iscatter(sendbuf, sendcount, MPI_INT, recvbuf, sendcount, MPI_INT, root, myFirstComm, &req);
            MPI_Wait(&req, &sts);
        }
        else {
            MPI_Scatter(sendbuf, sendcount, MPI_INT, recvbuf, sendcount, MPI_INT, root, myFirstComm);
        }

        int local_rank;
        MPI_Comm_rank(myFirstComm, &local_rank);
        printf ("[%d]", local_rank);
        for (int i = 0; i < sendcount; i++) {
            printf(" %d", recvbuf[i]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int size, rank;
    MPI_Comm myComm;
    MPI_Comm myFirstComm;
    MPI_Status sts;
    int root = 0; // rank of root in local group

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int color = rank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &myComm);
    MPI_Intercomm_create(myComm, 0, MPI_COMM_WORLD, (color+1)%2, 1, &myFirstComm);

    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

   /* Intercommunicator broadcast collective tests */
    // test intercomm_bcast
   //intercomm_bcast_test(myFirstComm, global_rank, root);
   // test intercomm_ibcast
   // intercomm_ibcast_test(myFirstComm, global_rank, root);

   /* Intercommunicator barrier collective tests */
   // test intercomm_barrier
   //intercomm_barrier_test(myFirstComm, global_rank);
   // test intercomm_ibarrier
   //intercomm_ibarrier_test(myFirstComm, global_rank);

    /* Intercommunicator gather collective tests */
   // test intercomm_gather
   //intercomm_gather_test(myFirstComm, global_rank, root);
   // test intercomm_igather
   // intercomm_igather_test(myFirstComm, global_rank, root);

    /* Intercommunicator gatherv collective tests */
    //intercomm_gatherv_test(myFirstComm, global_rank, root, 0); /* Intercomm gatherv test */
    //intercomm_gatherv_test(myFirstComm, global_rank, root, 1); /* Intercomm igatherv test */

    /* Intercommunicator scatter collective tests */
    intercomm_scatter_test(myFirstComm, global_rank, root, 0); /* Intercomm scatter test */
    //intercomm_scatter_test(myFirstComm, global_rank, root, 1); /* Intercomm iscatter test */

    MPI_Comm_free(&myComm);
    MPI_Comm_free(&myFirstComm);

    MPI_Finalize();
    return 0;

}
