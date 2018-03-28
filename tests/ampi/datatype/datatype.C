#include <stdio.h>
#include <vector>
#include <numeric>
#include <mpi.h>

#define VEC_NELM 128
#define VEC_STRIDE 8

//  Test type reference counting for SendReq
void typefree_isend_test(int rank, int size) {
    int i, errs = 0;
    int source = 0, dest = size - 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    std::vector<int> buf;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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
        buf.resize(VEC_NELM * VEC_STRIDE);
        std::iota(buf.begin(), buf.end(), 0);

        MPI_Isend(buf.data(), 1, strideType, dest, 0, comm, &req);
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
        MPI_Sendrecv(NULL, 0, MPI_INT, dest, 1, NULL, 0, MPI_INT, dest, 1, comm, MPI_STATUS_IGNORE);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
    }
    else if (rank == dest) {
        buf.resize(VEC_NELM);
        for (i = 0; i < VEC_NELM; i++)
            buf[i] = -i;

        /* Synchronize with the sender */
        MPI_Sendrecv(NULL, 0, MPI_INT, source, 1, NULL, 0, MPI_INT, source, 1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(buf.data(), VEC_NELM, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);

        for (i = 0; i < VEC_NELM; i++) {
            if (buf[i] != i * VEC_STRIDE) {
                errs++;
                if (errs < 10) {
                    printf("buf[%d] = %d, expected %d\n", VEC_STRIDE * i, buf[VEC_STRIDE * i], i);
                }
            }
        }
    }

    /* Clean up the strideType */
    if (rank != source) {
        MPI_Type_free(&strideType);
    }
}

//  Test type reference counting for SsendReq
void typefree_issend_test(int rank, int size) {
    int i, errs = 0;
    int source = 0, dest = size - 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    std::vector<int> buf;

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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
        buf.resize(VEC_NELM * VEC_STRIDE);
        std::iota(buf.begin(), buf.end(), 0);

        MPI_Issend(buf.data(), 1, strideType, dest, 0, comm, &req);
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
        MPI_Sendrecv(NULL, 0, MPI_INT, dest, 1, NULL, 0, MPI_INT, dest, 1, comm, MPI_STATUS_IGNORE);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
    }
    else if (rank == dest) {
        buf.resize(VEC_NELM);
        for (i = 0; i < VEC_NELM; i++)
            buf[i] = -i;

        /* Synchronize with the sender */
        MPI_Sendrecv(NULL, 0, MPI_INT, source, 1, NULL, 0, MPI_INT, source, 1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(buf.data(), VEC_NELM, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);

        for (i = 0; i < VEC_NELM; i++) {
            if (buf[i] != i * VEC_STRIDE) {
                errs++;
                if (errs < 10) {
                    printf("buf[%d] = %d, expected %d\n", VEC_STRIDE * i, buf[VEC_STRIDE * i], i);
                }
            }
        }
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
    int root = 0;
    int rank, size, i;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    MPI_Op op;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    MPI_Op_create((MPI_User_function *)vector_add, 1, &op);

    std::vector<int> sendbuf(VEC_NELM * size * size, rank+1);
    std::vector<int> recvbuf(VEC_NELM * size * size, 1);

    if (rank == root) {
        MPI_Ireduce(sendbuf.data(), recvbuf.data(), 1, strideType, op, root, comm, &req);
        MPI_Type_free(&strideType);
        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes (comm limits these to 1 process)
        for (int r = root + 1; r < size; r++) {
          MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);

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
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, MPI_STATUS_IGNORE);
        MPI_Ireduce(sendbuf.data(), recvbuf.data(), 1, strideType, op, root, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Type_free(&strideType);
    }
}

//  Test type reference counting for GatherReq
void typefree_igather_test(int rank, int size) {
    int i, errs = 0;
    int root = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req;
    MPI_Datatype contigType;
    MPI_Datatype tmpType[1024];
    MPI_Op op;
    std::vector<int> sendbuf(VEC_NELM * size);
    std::iota(sendbuf.begin(), sendbuf.end(), 0);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_contiguous(VEC_NELM, MPI_INT, &contigType);
    MPI_Type_commit(&contigType);

    if (rank == root) {
        /*
         * MPI_gather places a newly received vector after the extent of the previous vector in recvbuf.
         * If contigType is incorrectly freed and its memory is reused, allocating this extra memory ensures
         * that the program does not segfault.
         */
        std::vector<int> recvbuf(VEC_NELM * size * size);
        for (i = 0; i < VEC_NELM * size * size; i++)
            recvbuf[i] = -i;

        MPI_Igather(&sendbuf[rank], 1, contigType, recvbuf.data(), 1, contigType, root, comm, &req);
        MPI_Type_free(&contigType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes
        for (int r = root + 1; r < size; r++) {
            MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);

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
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, MPI_STATUS_IGNORE);

        MPI_Igather(&sendbuf[VEC_NELM*rank], VEC_NELM, MPI_INT, NULL, 0, 0, root, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        MPI_Type_free(&contigType);
    }
}

//  Test type reference counting for GathervReq
void typefree_igatherv_test(int rank, int size) {
    int errs = 0;
    int root = 0;
    int i, j;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req;
    MPI_Datatype strideType, resizedType;
    MPI_Datatype tmpType[1024];
    std::vector<int> sendbuf(VEC_NELM * size);
    std::vector<int> recvcounts(size, 1), displs(size);

    std::iota(sendbuf.begin(), sendbuf.end(), 0);
    std::iota(displs.begin(), displs.end(), 0);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    // This enables the root to receive vectors in column-major order.
    MPI_Type_create_resized(strideType, 0, 4, &resizedType);

    if (rank == root) {
        std::vector<int> recvbuf(VEC_NELM * size);
        for (i = 0; i < VEC_NELM * size; i++)
          recvbuf[i] = -i;

        MPI_Igatherv(sendbuf.data(), VEC_NELM, MPI_INT, recvbuf.data(), recvcounts.data(), displs.data(), resizedType, root, comm, &req);
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes
        for (int r = root + 1; r < size; r++) {
            MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);

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
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, MPI_STATUS_IGNORE);

        MPI_Igatherv(&sendbuf[rank*VEC_NELM], VEC_NELM, MPI_INT, NULL, recvcounts.data(), displs.data(), strideType, root, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);
    }
}

//  Test type reference counting for ATAReq
void typefree_ialltoallv_test(MPI_Comm comm) {
    int errs = 0;
    int i, rank, size;
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

    std::vector<int> sendbuf(VEC_NELM * size), recvbuf(VEC_NELM * size);
    std::vector<int> sendcounts(size, 1), recvcounts(size, 1);
    std::vector<int> sdispls(size), rdispls(size);
    for (i = 0; i < size; i++) {
        sdispls[i] = (rank + i) % size;
        rdispls[i] = (rank + i) % size;
    }

    std::iota(sendbuf.begin(), sendbuf.end(), 0);
    for (i = 0; i < VEC_NELM * size; i++) {
        recvbuf[i] = -i;
    }

    if (rank == 0) {
        MPI_Ialltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), resizedType,
                       recvbuf.data(), recvcounts.data(), rdispls.data(), resizedType,
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

        MPI_Sendrecv(NULL, 0, MPI_INT, 1, 1, NULL, 0, MPI_INT, 1, 1, comm, MPI_STATUS_IGNORE);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

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
        MPI_Sendrecv(NULL, 0, MPI_INT, 0, 1, NULL, 0, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);
        MPI_Ialltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), resizedType,
                       recvbuf.data(), recvcounts.data(), rdispls.data(), resizedType,
                       comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
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
}

//  Test type reference counting for ATAReq
void typefree_ialltoall_test(MPI_Comm comm) {
    int errs = 0;
    int i, rank, size;
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

    std::vector<int> sendbuf(VEC_NELM * size), recvbuf(VEC_NELM * size);
    std::vector<int> sendcounts(size, 1), recvcounts(size, 1);
    std::vector<int> sdispls(size), rdispls(size);
    for (i = 0; i < size; i++) {
        sdispls[i] = (rank + i) % size;
        rdispls[i] = (rank + i) % size;
    }

    std::iota(sendbuf.begin(), sendbuf.end(), 0);
    for (i = 0; i < VEC_NELM * size; i++) {
        recvbuf[i] = -i;
    }

    if (rank == 0) {
        MPI_Ialltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), contigType,
                       recvbuf.data(), recvcounts.data(), rdispls.data(), contigType,
                       comm, &req);
        MPI_Type_free(&contigType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }

        MPI_Sendrecv(NULL, 0, MPI_INT, 1, 1, NULL, 0, MPI_INT, 1, 1, comm, MPI_STATUS_IGNORE);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

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
        MPI_Sendrecv(NULL, 0, MPI_INT, 0, 1, NULL, 0, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);
        MPI_Ialltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), contigType,
                       recvbuf.data(), recvcounts.data(), rdispls.data(), contigType,
                       comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
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
}

//  Test type reference counting for ATAReq
void typefree_iscatterv_test(int rank, int size) {
    int errs = 0;
    int root = 0;
    int i;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req;
    MPI_Datatype strideType, resizedType;
    MPI_Datatype tmpType[1024];
    std::vector<int> recvbuf(VEC_NELM * size);
    std::vector<int> sendcounts(size, 1), displs(size);

    if (size < 2) {
        fprintf(stderr, "This test requires at least two processes.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Type_vector(VEC_NELM, 1, size, MPI_INT, &strideType);
    MPI_Type_commit(&strideType);
    // This enables the root to send vectors from sendbuf in column-major order.
    MPI_Type_create_resized(strideType, 0, 4, &resizedType);

    std::iota(displs.begin(), displs.end(), 0);
    for (i = 0; i < VEC_NELM; i++)
        recvbuf[i] = -i;

    if (rank == root) {
        std::vector<int> sendbuf(VEC_NELM * size);
        std::iota(sendbuf.begin(), sendbuf.end(), 0);

        MPI_Iscatterv(sendbuf.data(), sendcounts.data(), displs.data(), resizedType, recvbuf.data(), VEC_NELM, MPI_INT, root, comm, &req);
        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        // Sync with non-root processes
        for (int r = root + 1; r < size; r++) {
            MPI_Sendrecv(NULL, 0, MPI_INT, r, 1, NULL, 0, MPI_INT, r, 1, comm, MPI_STATUS_IGNORE);
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        for (i = 0; i < 1024; i++) {
            MPI_Type_free(&tmpType[i]);
        }
    } else {
        // Sync with root
        MPI_Sendrecv(NULL, 0, MPI_INT, root, 1, NULL, 0, MPI_INT, root, 1, comm, MPI_STATUS_IGNORE);

        MPI_Iscatterv(NULL, NULL, NULL, 0, recvbuf.data(), VEC_NELM, MPI_INT, root, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        MPI_Type_free(&strideType);
        MPI_Type_free(&resizedType);
    }

    // Check results
    if (rank != root) {
        for (i = 0; i < VEC_NELM; i++) {
            if (recvbuf[i] != i*size+rank) {
                errs++;
                if (errs < 10) {
                    printf("recvbuf[%d] = %d, expected %d\n", i, recvbuf[i], i*size+rank);
                }
            }
        }
    }
}

void typefree_persistent_req_test(int rank, int size) {
    int i, errs = 0;
    int source = 0;
    int dest = size - 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request req;
    MPI_Datatype strideType;
    MPI_Datatype tmpType[1024];
    std::vector<int> buf;

    if (rank == dest) {
        MPI_Type_vector(VEC_NELM, 1, VEC_STRIDE, MPI_INT, &strideType);
        MPI_Type_commit(&strideType);

        buf.resize(VEC_NELM * VEC_STRIDE);
        for (i = 0; i < VEC_NELM * VEC_STRIDE; i++) {
            buf[i] = -i;
        }

        MPI_Recv_init(buf.data(), 1, strideType, source, 0, comm, &req);

        MPI_Start(&req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        MPI_Type_free(&strideType);

        for (i = 0; i < 1024; i++) {
            MPI_Type_vector(VEC_NELM, 1, 1, MPI_INT, &tmpType[i]);
            MPI_Type_commit(&tmpType[i]);
        }

        /* Synchronize with the sender */
        MPI_Sendrecv(NULL, 0, MPI_INT, source, 1, NULL, 0, MPI_INT, source, 1, comm, MPI_STATUS_IGNORE);

        MPI_Start(&req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

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
        buf.resize(VEC_NELM, 0);

        MPI_Isend(buf.data(), VEC_NELM, MPI_INT, dest, 0, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        /* Synchronize with the receiver */
        MPI_Sendrecv(NULL, 0, MPI_INT, dest, 1, NULL, 0, MPI_INT, dest, 1, comm, MPI_STATUS_IGNORE);

        std::iota(buf.begin(), buf.end(), 0);
        MPI_Isend(buf.data(), VEC_NELM, MPI_INT, dest, 0, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
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
