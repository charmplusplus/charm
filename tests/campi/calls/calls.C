// campi test: the blocking calls that are awaitables beside the common ones:
// MPI_Probe, MPI_Mprobe/MPI_Mrecv, MPI_Waitany, MPI_Waitsome (also over a
// collective request), MPI_Bsend, MPI_Rsend, MPI_Sendrecv_replace (also over
// a non-contiguous datatype), MPI_Reduce_scatter_block, MPI_Reduce_scatter
// and MPI_Alltoallw (also with MPI_IN_PLACE), and MPI_Comm_split woken by an
// unrelated wake-up. Where a call should block, the
// matching send is held back until the receiver has signalled, so the
// receiver suspends in the call.
#include <cstdio>
#include <vector>
#include "mpi.h"
#include "campi.h"

CAMPI_MAIN(argc, argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int right = (rank + 1) % size, left = (rank + size - 1) % size;
  long myerr = 0;
#define C(c)                                                                  \
  do {                                                                        \
    if (!(c)) { myerr++; printf("rank %d: check failed at line %d: %s\n",     \
                                rank, __LINE__, #c); }                        \
  } while (0)
  int go = 1;

  // 1. MPI_Probe: tell the left neighbour to send, then probe for it
  {
    MPI_Send(&go, 1, MPI_INT, left, 10, MPI_COMM_WORLD);
    MPI_Recv(&go, 1, MPI_INT, right, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int out[2] = {rank, rank + 100};
    MPI_Request r;
    MPI_Isend(out, 2, MPI_INT, right, 11, MPI_COMM_WORLD, &r);
    MPI_Status st;
    MPI_Probe(MPI_ANY_SOURCE, 11, MPI_COMM_WORLD, &st);
    int cnt = -1, in[2] = {-1, -1};
    MPI_Get_count(&st, MPI_INT, &cnt);
    C(st.MPI_SOURCE == left && st.MPI_TAG == 11 && cnt == 2);
    MPI_Recv(in, 2, MPI_INT, st.MPI_SOURCE, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    C(in[0] == left && in[1] == left + 100);
    MPI_Wait(&r, MPI_STATUS_IGNORE);
    MPI_Probe(MPI_PROC_NULL, 11, MPI_COMM_WORLD, &st);
    C(st.MPI_SOURCE == MPI_PROC_NULL);
  }

  // 2. MPI_Mprobe and MPI_Mrecv, and both with MPI_PROC_NULL
  {
    MPI_Send(&go, 1, MPI_INT, left, 20, MPI_COMM_WORLD);
    MPI_Recv(&go, 1, MPI_INT, right, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int out = rank * 3;
    MPI_Request r;
    MPI_Isend(&out, 1, MPI_INT, right, 21, MPI_COMM_WORLD, &r);
    MPI_Message m;
    MPI_Status st;
    MPI_Mprobe(left, 21, MPI_COMM_WORLD, &m, &st);
    C(st.MPI_SOURCE == left && st.MPI_TAG == 21);
    int in = -1;
    MPI_Mrecv(&in, 1, MPI_INT, &m, &st);
    C(in == left * 3 && m == MPI_MESSAGE_NULL && st.MPI_SOURCE == left);
    MPI_Wait(&r, MPI_STATUS_IGNORE);
    MPI_Mprobe(MPI_PROC_NULL, 21, MPI_COMM_WORLD, &m, &st);
    C(m == MPI_MESSAGE_NO_PROC);
    MPI_Mrecv(&in, 1, MPI_INT, &m, &st);
    C(m == MPI_MESSAGE_NULL && st.MPI_SOURCE == MPI_PROC_NULL);
  }

  // 3. MPI_Waitany over receives from every rank, posted before a barrier
  //    after which the others send; then over only null requests
  {
    std::vector<int> in(size, -1);
    std::vector<MPI_Request> reqs(size);
    for (int src = 0; src < size; src++)
      MPI_Irecv(&in[src], 1, MPI_INT, src, 30, MPI_COMM_WORLD, &reqs[src]);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int k = 0; k < size; k++) {
      int dst = (rank + k) % size, out = rank * 10 + dst;
      MPI_Send(&out, 1, MPI_INT, dst, 30, MPI_COMM_WORLD);
    }
    std::vector<int> seen(size, 0);
    for (int k = 0; k < size; k++) {
      int idx = -1;
      MPI_Status st;
      MPI_Waitany(size, reqs.data(), &idx, &st);
      C(idx >= 0 && idx < size && reqs[idx] == MPI_REQUEST_NULL);
      if (idx >= 0 && idx < size) {
        seen[idx]++;
        C(st.MPI_SOURCE == idx && in[idx] == idx * 10 + rank);
      }
    }
    for (int s = 0; s < size; s++) C(seen[s] == 1);
    int idx = 0;
    MPI_Waitany(size, reqs.data(), &idx, MPI_STATUS_IGNORE);
    C(idx == MPI_UNDEFINED);
  }

  // 4. MPI_Waitany over a receive whose sender waits for us, and an
  //    Ibarrier: both must complete, in either order
  {
    MPI_Request reqs[2];
    int in = -1;
    MPI_Irecv(&in, 1, MPI_INT, left, 40, MPI_COMM_WORLD, &reqs[0]);
    MPI_Ibarrier(MPI_COMM_WORLD, &reqs[1]);
    MPI_Send(&go, 1, MPI_INT, left, 41, MPI_COMM_WORLD);
    MPI_Recv(&go, 1, MPI_INT, right, 41, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int out = rank + 4000;
    MPI_Send(&out, 1, MPI_INT, right, 40, MPI_COMM_WORLD);
    int done = 0, idx;
    for (int k = 0; k < 2; k++) {
      MPI_Waitany(2, reqs, &idx, MPI_STATUS_IGNORE);
      if (idx == 0 || idx == 1) done |= 1 << idx;
    }
    C(done == 3 && in == left + 4000);
  }

  // 5. MPI_Waitsome until every receive is in, then over null requests
  {
    std::vector<int> in(size, -1);
    std::vector<MPI_Request> reqs(size);
    for (int src = 0; src < size; src++)
      MPI_Irecv(&in[src], 1, MPI_INT, src, 50, MPI_COMM_WORLD, &reqs[src]);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int k = size - 1; k >= 0; k--) {
      int dst = (rank + k) % size, out = rank * 7 + dst;
      MPI_Send(&out, 1, MPI_INT, dst, 50, MPI_COMM_WORLD);
    }
    std::vector<int> idx(size);
    std::vector<MPI_Status> sts(size);
    int got = 0, rounds = 0;
    while (got < size) {
      int n = 0;
      MPI_Waitsome(size, reqs.data(), &n, idx.data(), sts.data());
      C(n >= 1 && n <= size - got);
      if (n < 1 || n > size - got) break;
      for (int i = 0; i < n; i++)
        C(sts[i].MPI_SOURCE == idx[i] && in[idx[i]] == idx[i] * 7 + rank);
      got += n;
      rounds++;
    }
    int n = 0;
    MPI_Waitsome(size, reqs.data(), &n, idx.data(), sts.data());
    C(got == size && n == MPI_UNDEFINED);
  }

  // 6. MPI_Bsend around the ring, and MPI_Rsend to a posted receive
  {
    int bufsize = 4 * (MPI_BSEND_OVERHEAD + (int)sizeof(int));
    std::vector<char> attach(bufsize);
    MPI_Buffer_attach(attach.data(), bufsize);
    int out = rank + 600, in = -1;
    MPI_Bsend(&out, 1, MPI_INT, right, 60, MPI_COMM_WORLD);
    MPI_Recv(&in, 1, MPI_INT, left, 60, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    C(in == left + 600);
    void *b;
    int bs;
    MPI_Buffer_detach(&b, &bs);
    MPI_Request r;
    in = -1;
    MPI_Irecv(&in, 1, MPI_INT, left, 61, MPI_COMM_WORLD, &r);
    MPI_Barrier(MPI_COMM_WORLD);
    out = rank + 610;
    MPI_Rsend(&out, 1, MPI_INT, right, 61, MPI_COMM_WORLD);
    MPI_Wait(&r, MPI_STATUS_IGNORE);
    C(in == left + 610);
  }

  // 7. MPI_Sendrecv_replace: contiguous, then a vector type whose gaps must
  //    be left alone
  {
    int v[3] = {rank, rank + 1, rank + 2};
    MPI_Status st;
    MPI_Sendrecv_replace(v, 3, MPI_INT, right, 70, left, 70, MPI_COMM_WORLD, &st);
    C(v[0] == left && v[1] == left + 1 && v[2] == left + 2 && st.MPI_SOURCE == left);
    MPI_Datatype vec;
    MPI_Type_vector(3, 1, 2, MPI_INT, &vec);
    MPI_Type_commit(&vec);
    int w[5] = {rank, -1, rank + 10, -2, rank + 20};
    MPI_Sendrecv_replace(w, 1, vec, right, 71, left, 71, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    C(w[0] == left && w[2] == left + 10 && w[4] == left + 20 && w[1] == -1 && w[3] == -2);
    MPI_Type_free(&vec);
  }

  // 8. MPI_Reduce_scatter_block, then in place
  {
    const int K = 2;
    std::vector<int> s(size * K), r(K, -1);
    for (int i = 0; i < size * K; i++) s[i] = rank + i;
    MPI_Reduce_scatter_block(s.data(), r.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // element rank*K + k summed over ranks: sum(q) + size*(rank*K + k)
    int sumq = size * (size - 1) / 2;
    for (int k = 0; k < K; k++) C(r[k] == sumq + size * (rank * K + k));
    MPI_Reduce_scatter_block(MPI_IN_PLACE, s.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (int k = 0; k < K; k++) C(s[k] == sumq + size * (rank * K + k));
  }

  // 9. MPI_Reduce_scatter with rank i receiving i+1 elements
  {
    std::vector<int> counts(size);
    int total = 0;
    for (int i = 0; i < size; i++) total += counts[i] = i + 1;
    std::vector<int> s(total), r(rank + 1, -1);
    for (int i = 0; i < total; i++) s[i] = i * (rank + 1);
    MPI_Reduce_scatter(s.data(), r.data(), counts.data(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int first = rank * (rank + 1) / 2, f = size * (size + 1) / 2;
    for (int k = 0; k <= rank; k++) C(r[k] == (first + k) * f);
  }

  // 10. MPI_Alltoallw: rank sends dst+1 ints to dst; displacements in bytes.
  //     Then in place, with one int per peer.
  {
    std::vector<int> scounts(size), sdispls(size), rcounts(size), rdispls(size);
    std::vector<MPI_Datatype> types(size, MPI_INT);
    int sn = 0, rn = 0;
    for (int i = 0; i < size; i++) {
      scounts[i] = i + 1;
      sdispls[i] = sn * (int)sizeof(int);
      sn += i + 1;
      rcounts[i] = rank + 1;
      rdispls[i] = rn * (int)sizeof(int);
      rn += rank + 1;
    }
    std::vector<int> s(sn), r(rn, -1);
    for (int i = 0; i < size; i++)
      for (int k = 0; k <= i; k++) s[sdispls[i] / sizeof(int) + k] = rank * 1000 + i * 10 + k;
    MPI_Alltoallw(s.data(), scounts.data(), sdispls.data(), types.data(), r.data(),
                  rcounts.data(), rdispls.data(), types.data(), MPI_COMM_WORLD);
    for (int src = 0; src < size; src++)
      for (int k = 0; k <= rank; k++)
        C(r[rdispls[src] / sizeof(int) + k] == src * 1000 + rank * 10 + k);

    std::vector<int> ones(size, 1), bytes(size), ip(size);
    for (int i = 0; i < size; i++) {
      bytes[i] = i * (int)sizeof(int);
      ip[i] = rank * 100 + i;
    }
    MPI_Alltoallw(MPI_IN_PLACE, nullptr, nullptr, nullptr, ip.data(), ones.data(),
                  bytes.data(), types.data(), MPI_COMM_WORLD);
    for (int src = 0; src < size; src++) C(ip[src] == src * 100 + rank);
  }

  // 11. MPI_Comm_split ends on its own wake-up only: rank 1 is suspended in
  // the split when rank 0 sends it an unrelated wake-up (AMPI_Resume)
  if (size >= 2) {
    if (rank == 1) MPI_Send(&go, 1, MPI_INT, 0, 20, MPI_COMM_WORLD);
    if (rank == 0) {
      MPI_Recv(&go, 1, MPI_INT, 1, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      AMPI_Resume(1, MPI_COMM_WORLD);
    }
    MPI_Comm half;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, -rank, &half);
    int hrank = -1, hsize = -1, sum = -1;
    MPI_Comm_rank(half, &hrank);
    MPI_Comm_size(half, &hsize);
    C(hsize == (size + 1 - rank % 2) / 2 && hrank == hsize - 1 - rank / 2);
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, half);
    int expect = 0;
    for (int i = rank % 2; i < size; i += 2) expect += i;
    C(sum == expect);
    MPI_Comm_free(&half);
  }

  long total = 0;
  MPI_Allreduce(&myerr, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    printf("campi calls: %d ranks, errors=%ld: %s\n", size, total, total == 0 ? "PASSED" : "FAILED");
  MPI_Finalize();
  co_return total == 0 ? 0 : 1;
}
