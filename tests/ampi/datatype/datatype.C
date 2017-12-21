#include <stdio.h>
#include <mpi.h>

#define VEC_NELM 128
#define VEC_STRIDE 8

//  Test type reference counting for SendReq
void typefree_isend_test(int rank, int size) {
    int errs = 0;
    int source, dest, i;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    int *buf = 0;
    comm = MPI_COMM_WORLD;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    source = 0;
    dest = size - 1;

    /*
     * The idea here is to create a simple but non-contig datatype,
     * perform an isend with it, free it, and then create
     * many new datatypes.  While not a complete test, if the datatype
     * was freed and the space was reused, this test may detect
     * that error.
     */
    MPI_Type_vector(VEC_NELM, 1, VEC_STRIDE, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);

    if (rank == source) {
        buf = (int *) malloc(VEC_NELM * VEC_STRIDE * sizeof(int));
        for (i = 0; i < VEC_NELM * VEC_STRIDE; i++)
            buf[i] = i;
        MPI_Isend(buf, 1, strideType, dest, 0, comm, &req);
        MPI_Type_free(&strideType);

        /*
         * If strideType was incorrectly freed and the following types reuse its memory, then
         * we should see that elements in the destination's receive buffer are from the initial
         * VEC_NELEM / VEC_STRIDE "rows" of the send buffer instead of the first "column".
         */
        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        /* Synchronize with the receiver */
        MPI_Sendrecv(NULL, 0, MPI_INT, dest, 1, NULL, 0, MPI_INT, dest, 1, comm, &status);

        MPI_Wait(&req, &status);
        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
        free(buf);
    }
    else if (rank == dest) {
        buf = (int *) malloc(VEC_NELM * sizeof(int));
        for (i = 0; i < VEC_NELM; i++)
            buf[i] = -i;
        /* Synchronize with the sender */
        MPI_Sendrecv(NULL, 0, MPI_INT, source, 1, NULL, 0, MPI_INT, source, 1, comm, &status);
        MPI_Recv(buf, VEC_NELM, MPI_INT, source, 0, comm, &status);
        for (i = 0; i < VEC_NELM; i++) {
            if (buf[i] != i * VEC_STRIDE) {
                errs++;
                if (errs < 10) {
                    printf("buf[%d] = %d, expected %d\n", VEC_STRIDE * i, buf[VEC_STRIDE * i], i);
                }
            }
        }
        free(buf);
    }

    /* Clean up the strideType */
    if (rank != source) {
        MPI_Type_free(&strideType);
    }
}

//  Test type reference counting for SsendReq
void typefree_issend_test(int rank, int size) {
    int errs = 0;
    int source, dest, i;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    int *buf = 0;
    comm = MPI_COMM_WORLD;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    source = 0;
    dest = size - 1;

    /*
     * The idea here is to create a simple but non-contig datatype,
     * perform an isend with it, free it, and then create
     * many new datatypes.  While not a complete test, if the datatype
     * was freed and the space was reused, this test may detect
     * that error.
     */
    MPI_Type_vector(VEC_NELM, 1, VEC_STRIDE, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);

    if (rank == source) {
        buf = (int *) malloc(VEC_NELM * VEC_STRIDE * sizeof(int));
        for (i = 0; i < VEC_NELM * VEC_STRIDE; i++)
            buf[i] = i;
        MPI_Issend(buf, 1, strideType, dest, 0, comm, &req);
        MPI_Type_free(&strideType);

        /*
         * If strideType was incorrectly freed and the following types reuse its memory, then
         * we should see that elements in the destination's receive buffer are from the initial
         * VEC_NELEM / VEC_STRIDE "rows" of the send buffer instead of the first "column".
         */
        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        /* Synchronize with the receiver */
        MPI_Sendrecv(NULL, 0, MPI_INT, dest, 1, NULL, 0, MPI_INT, dest, 1, comm, &status);

        MPI_Wait(&req, &status);
        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
        free(buf);
    }
    else if (rank == dest) {
        buf = (int *) malloc(VEC_NELM * sizeof(int));
        for (i = 0; i < VEC_NELM; i++)
            buf[i] = -i;
        /* Synchronize with the sender */
        MPI_Sendrecv(NULL, 0, MPI_INT, source, 1, NULL, 0, MPI_INT, source, 1, comm, &status);
        MPI_Recv(buf, VEC_NELM, MPI_INT, source, 0, comm, &status);
        for (i = 0; i < VEC_NELM; i++) {
            if (buf[i] != i * VEC_STRIDE) {
                errs++;
                if (errs < 10) {
                    printf("buf[%d] = %d, expected %d\n", VEC_STRIDE * i, buf[VEC_STRIDE * i], i);
                }
            }
        }
        free(buf);
    }

    /* Clean up the strideType */
    if (rank != source) {
        MPI_Type_free(&strideType);
    }
}

void vector_add(int *invec, int *inoutvec, int *len, MPI_Datatype *dtype) {
    int count = VEC_NELM;
    // Needs to be the same as the number of processes in the communicator used for typefree_ireduce_test
    int stride = 2;

    for ( int i=0; i<count; i++ ) {
        inoutvec[i*stride] += invec[i*stride];
    }
}

// Test type reference counting for RednReq
void typefree_ireduce_test(MPI_Comm comm) {
    int errs = 0;
    int rank, size, i;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    MPI_Op op;
    int *sendbuf = 0;
    int *recvbuf = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int root = 0;

    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    MPI_Op_create((MPI_User_function *)vector_add, 1, &op);

    sendbuf = (int *) malloc(VEC_NELM * size * size * sizeof(int));
    recvbuf = (int *) malloc(VEC_NELM * size * size * sizeof(int));
    for (i = 0; i < VEC_NELM * size * size; i++) {
        recvbuf[i] = 0;
        sendbuf[i] = rank+1;
    }
    if (rank == root) {
        MPI_Ireduce(sendbuf, recvbuf, 1, strideType, op, root, comm, &req);
        MPI_Type_free(&strideType);
        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes (comm limits these to 1 process)
        for (int r = root + 1; r < size; r++) {
          MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, &status);
        }
        MPI_Wait(&req, &status);

        int expected = (size * (size+1)) / 2;
        // Check results
        for (i = 0; i < VEC_NELM; i++) {
            if (recvbuf[i*size] != expected) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i*size], expected);
                }
            }
        }

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, &status);
        MPI_Ireduce(sendbuf, recvbuf, 1, strideType, op, root, comm, &req);
        MPI_Wait(&req, &status);
        MPI_Type_free(&strideType);
    }
    free(sendbuf);
    free(recvbuf);
}

//  Test type reference counting for GatherReq
void typefree_igather_test(int rank, int size) {
    int errs = 0;
    int i;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype contigType;
    MPI_Datatype tmpType[1024];
    MPI_Op op;
    int *sendbuf = 0;
    int *recvbuf = 0;

    comm = MPI_COMM_WORLD;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int root = 0;

    MPI_Type_contiguous(VEC_NELM, MPI_INT, &contigType);
    MPI_Type_commit(&contigType);

    sendbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
    for (i = 0; i < VEC_NELM * size; i++)
        sendbuf[i] = i;
    if (rank == root) {
        /*
         * MPI_gather places a newly received vector after the extent of the previous vector in recvbuf.
         * If contigType is incorrectly freed and its memory is reused, allocating this extra memory ensures
         * that the program does not segfault.
         */
        recvbuf = (int *) malloc(VEC_NELM * size * size * sizeof(int));
        for (i = 0; i < VEC_NELM * size * size; i++)
            recvbuf[i] = -i;

        MPI_Igather(sendbuf+rank, 1, contigType, recvbuf, 1, contigType, root, comm, &req);
        MPI_Type_free(&contigType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes
        for (int r = root + 1; r < size; r++) {
            MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, &status);
        }
        MPI_Wait(&req, &status);

        // Check result
        for (i = 0; i < VEC_NELM * size; i++) {
            if (recvbuf[i] != i) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i);
                }
            }
        }

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
        free(recvbuf);
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, &status);

        MPI_Igather(sendbuf+(VEC_NELM*rank), VEC_NELM, MPI_INT, NULL, 0, 0, root, comm, &req);
        MPI_Wait(&req, &status);

        MPI_Type_free(&contigType);
    }

    free(sendbuf);
}

//  Test type reference counting for GathervReq
void typefree_igatherv_test(int rank, int size) {
    int errs = 0;
    int i, j;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType, resizedType;
    MPI_Datatype tmpType[1024];
    int *sendbuf = 0;
    int *recvbuf = 0;
    int recvcounts[size], displs[size];

    comm = MPI_COMM_WORLD;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int root = 0;

    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    // This enables the root to receive vectors in column-major order.
    MPI_Type_create_resized(strideType, 0, 4, &resizedType);

    for (i = 0; i < size; i++) {
        recvcounts[i] = 1;
        displs[i] = i;
    }

    sendbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
    for (i = 0; i < VEC_NELM * size; i++)
        sendbuf[i] = i;

    if (rank == root) {
        recvbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
        for (i = 0; i < VEC_NELM * size; i++)
          recvbuf[i] = -i;

        MPI_Igatherv(sendbuf, VEC_NELM, MPI_INT, recvbuf, recvcounts, displs, resizedType, root, comm, &req);
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes
        for (int r = root + 1; r < size; r++) {
            MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, &status);
        }
        MPI_Wait(&req, &status);

        // We expect to have received a "transposed" sendbuf.
        for (i = 0; i < size; i++) {
            for (j = 0; j < VEC_NELM; j++) {
                if (recvbuf[j*size + i] != i*VEC_NELM+j) {
                    errs++;
                    if (errs < 10) {
                        printf("recvbuf[%d] = %d, expected %d\n", j*size+i, recvbuf[j*size+i], i*VEC_NELM+j);
                    }
                }
            }
        }

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
        free(recvbuf);
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, &status);

        MPI_Igatherv(sendbuf+(rank*VEC_NELM), VEC_NELM, MPI_INT, NULL, recvcounts, displs, strideType, root, comm, &req);
        MPI_Wait(&req, &status);

        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);
    }

    free(sendbuf);
}

//  Test type reference counting for ATAReq
void typefree_ialltoallv_test(MPI_Comm comm) {
    int errs = 0;
    int i, rank, size;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType, resizedType;
    MPI_Datatype tmpType[1024];

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    MPI_Type_create_resized(strideType, 0, 4, &resizedType);

    int *sendbuf = 0;
    int *recvbuf = 0;
    int sendcounts[size], recvcounts[size], sdispls[size], rdispls[size];
    for (i = 0; i < size; i++) {
        sendcounts[i] = 1;
        sdispls[i] = (rank + i) % size;
        recvcounts[i] = 1;
        rdispls[i] = (rank + i) % size;
    }

    sendbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
    recvbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
    for (i = 0; i < VEC_NELM * size; i++) {
        sendbuf[i] = i;
        recvbuf[i] = -i;
    }

    if (rank == 0) {
        MPI_Ialltoallv(sendbuf, sendcounts, sdispls, resizedType,
                       recvbuf, recvcounts, rdispls, resizedType,
                       comm, &req);
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }

        MPI_Sendrecv(NULL, 0, MPI_INT, 1, 1, NULL, 0, MPI_INT, 1, 1, comm, &status);
        MPI_Wait(&req, &status);

        // Check results
        for (i = 0; i < VEC_NELM * size; i++) {
            if (recvbuf[i] != i) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i);
                }
            }
        }
    } else if (rank == 1) {
        MPI_Sendrecv(NULL, 0, MPI_INT, 0, 1, NULL, 0, MPI_INT, 0, 1, comm, &status);
        MPI_Ialltoallv(sendbuf, sendcounts, sdispls, resizedType,
                       recvbuf, recvcounts, rdispls, resizedType,
                       comm, &req);
        MPI_Wait(&req, &status);
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);

        // Check results
        for (i = 0; i < VEC_NELM * size; i++) {
            if (recvbuf[i] != i) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i);
                }
            }
        }
    } else {
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);
    }
    free(sendbuf);
    free(recvbuf);
}

//  Test type reference counting for ATAReq
void typefree_ialltoall_test(MPI_Comm comm) {
    int errs = 0;
    int i, rank, size;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype contigType;
    MPI_Datatype tmpType[1024];

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Type_contiguous(VEC_NELM, MPI_INT, &contigType);
    MPI_Type_commit(&contigType);

    int *sendbuf = 0;
    int *recvbuf = 0;
    int sendcounts[size], recvcounts[size], sdispls[size], rdispls[size];
    for (i = 0; i < size; i++) {
        sendcounts[i] = 1;
        sdispls[i] = (rank + i) % size;
        recvcounts[i] = 1;
        rdispls[i] = (rank + i) % size;
    }

    sendbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
    recvbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
    for (i = 0; i < VEC_NELM * size; i++) {
        sendbuf[i] = i;
        recvbuf[i] = -i;
    }

    if (rank == 0) {
        MPI_Ialltoallv(sendbuf, sendcounts, sdispls, contigType,
                       recvbuf, recvcounts, rdispls, contigType,
                       comm, &req);
        MPI_Type_free(&contigType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }

        MPI_Sendrecv(NULL, 0, MPI_INT, 1, 1, NULL, 0, MPI_INT, 1, 1, comm, &status);
        MPI_Wait(&req, &status);

        // Check results
        for (i = 0; i < VEC_NELM * size; i++) {
            if (recvbuf[i] != i) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i);
                }
            }
        }
    } else if (rank == 1) {
        MPI_Sendrecv(NULL, 0, MPI_INT, 0, 1, NULL, 0, MPI_INT, 0, 1, comm, &status);
        MPI_Ialltoallv(sendbuf, sendcounts, sdispls, contigType,
                       recvbuf, recvcounts, rdispls, contigType,
                       comm, &req);
        MPI_Wait(&req, &status);
        MPI_Type_free(&contigType);

        // Check results
        for (i = 0; i < VEC_NELM * size; i++) {
            if (recvbuf[i] != i) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i);
                }
            }
        }
    } else {
        MPI_Type_free(&contigType);
    }
    free(sendbuf);
    free(recvbuf);
}

//  Test type reference counting for ATAReq
void typefree_iscatterv_test(int rank, int size) {
    int errs = 0;
    int i;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType, resizedType;
    MPI_Datatype tmpType[1024];
    int *sendbuf = 0;
    int *recvbuf = 0;
    int sendcounts[size], displs[size];

    comm = MPI_COMM_WORLD;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int root = 0;

    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    // This enables the root to send vectors from sendbuf in column-major order.
    MPI_Type_create_resized(strideType, 0, 4, &resizedType);

    for (i = 0; i < size; i++) {
        sendcounts[i] = 1;
        displs[i] = i;
    }

    recvbuf = (int *) malloc(VEC_NELM * sizeof(int));
    for (i = 0; i < VEC_NELM; i++)
        recvbuf[i] = -i;

    if (rank == root) {
        sendbuf = (int *) malloc(VEC_NELM * size * sizeof(int));
        for (i = 0; i < VEC_NELM * size; i++)
            sendbuf[i] = i;
        MPI_Iscatterv(sendbuf, sendcounts, displs, resizedType, recvbuf, VEC_NELM, MPI_INT, root, comm, &req);
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes
        for (int r = root + 1; r < size; r++) {
            MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, &status);
        }
        MPI_Wait(&req, &status);

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
        free(sendbuf);
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, &status);

        MPI_Iscatterv(NULL, NULL, NULL, 0, recvbuf, VEC_NELM, MPI_INT, root, comm, &req);
        MPI_Wait(&req, &status);

        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);
    }

    // Check results
    if (rank != root)
      for (i = 0; i < VEC_NELM; i++) {
          if (recvbuf[i] != i*size+rank) {
              errs++;
              if (errs < 10) {
                  printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i*size+rank);
              }
          }
      }

    free(recvbuf);
}

void typefree_persistent_req_test(int rank, int size) {
    int errs = 0;
    int source, dest, i;
    MPI_Comm comm;
    MPI_Status status;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    int *buf = 0;
    comm = MPI_COMM_WORLD;

    source = 0;
    dest = size - 1;

    if (rank == dest) {
        MPI_Type_vector(VEC_NELM, 1, VEC_STRIDE, MPI_INT, &strideType);
        MPI_Type_commit(&strideType);

        buf = (int *) malloc(VEC_NELM * VEC_STRIDE * sizeof(int));
        for (i = 0; i < VEC_NELM * VEC_STRIDE; i++) {
            buf[i] = -i;
        }

        MPI_Recv_init(buf, 1, strideType, source, 0, comm, &req);

        MPI_Start(&req);
        MPI_Wait(&req, &status);

        MPI_Type_free(&strideType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        /* Synchronize with the sender */
        MPI_Sendrecv(NULL, 0, MPI_INT, source, 1, NULL, 0, MPI_INT, source, 1, comm, &status);

        MPI_Start(&req);
        MPI_Wait(&req, &status);

        MPI_Request_free(&req);

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }

        // Check results
        for (i = 0; i < VEC_NELM; i++) {
            if (buf[i*VEC_STRIDE] != i) {
                errs++;
                if (errs < 10) {
                    printf("buf[%d] = %d, expected %d\n", VEC_STRIDE * i, buf[VEC_STRIDE * i], i);
                }
            }
        }
    } else if (rank == source) {
        buf = (int *) malloc(VEC_NELM * sizeof(int));
        for (i = 0; i < VEC_NELM; i++) {
            buf[i] = 0;
        }
        MPI_Isend(buf, VEC_NELM, MPI_INT, dest, 0, comm, &req);
        MPI_Wait(&req, &status);

        /* Synchronize with the receiver */
        MPI_Sendrecv(NULL, 0, MPI_INT, dest, 1, NULL, 0, MPI_INT, dest, 1, comm, &status);

        for (i = 0; i < VEC_NELM; i++) {
            buf[i] = i;
        }
        MPI_Isend(buf, VEC_NELM, MPI_INT, dest, 0, comm, &req);
        MPI_Wait(&req, &status);
    }

    free(buf);
}

int main(int argc, char **argv) {
    int size, global_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (global_rank == 0) printf("Testing datatype free isend\n");
    typefree_isend_test(global_rank, size);

    if (global_rank == 0) printf("Testing datatype free issend\n");
    typefree_issend_test(global_rank, size);

    if (global_rank == 0) printf("Testing datatype free igatherv\n");
    typefree_igatherv_test(global_rank, size);

    if (global_rank == 0) printf("Testing datatype free igather\n");
    typefree_igather_test(global_rank, size);

    if (global_rank == 0) printf("Testing datatype free persistent request\n");
    typefree_persistent_req_test(global_rank, size);

    int color = (global_rank == 0 || global_rank == 1) ? 0 : 1;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, global_rank, &comm);

    /*
     * TODO: This test yields multiple errors when compiled with AMPI, but runs as
     * expected when compiled with MPICH. This is likely the result of a bug in
     * the AMPI alltoallv/ialltoallv implementation.
     */
//    if (global_rank == 0) printf("Testing datatype free ialltoallv\n");
//    if (global_rank == 0 || global_rank == 1)
//        typefree_ialltoallv_test(comm);

    if (global_rank == 0) printf("Testing datatype free ialltoall\n");
    if (global_rank == 0 || global_rank == 1)
        typefree_ialltoall_test(comm);

    /*
     * TODO: This test yields multiple errors/segfaults when compiled with AMPI, but
     * runs as expected when compiled with MPICH.
     */
//    if (global_rank == 0) printf("Testing datatype free ireduce\n");
//    if (global_rank == 0 || global_rank == 1)
//        typefree_ireduce_test(comm);
    MPI_Comm_free(&comm);

    /*
     * TODO: This test causes a segfault when compiled with AMPI, but runs as
     * expected when compiled with MPICH. This is likely the result of a bug in
     * the AMPI scatterv/iscatterv implementation.
     */
//    if (global_rank == 0) printf("Testing datatype free iscatterv\n");
//    typefree_iscatterv_test(global_rank, size);

    MPI_Finalize();

    return 0;
}
