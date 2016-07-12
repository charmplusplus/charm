// Particle Simulation - Edward Hutter

// Simulates particles randomly moving throughout a 3D domain, using a 3D decomposition.
// Communication is restricted to 8 neighbors per rank.
// Uses a cartesian comm, MPI_Isend, MPI_Iprobe, MPI_Recv.

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <utility>
#include <unistd.h>
#include <map>
#include <time.h>

using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::map;

#define CUBE_BOUNDS 100. // Length in each dimension of the cubic domain.

// Return a random floating point number: used to move the particle.
double fRand(int n)
{
  if (n == 0) {
    double f = ((double)rand())/RAND_MAX;
    f *= .1;
    return (rand()%2 > 0) ? f : (f*(-1.));
  }
  else if (n == 1) {
    return ((double)rand())/RAND_MAX;
  }
  else {
    return -1.;
  }
}

// 3-dimensional location of a particle.
struct particle
{
  double x;
  double y;
  double z;
};

// Each rank owns an environment (subdomain) where its particles are free to move around.
// When a particles moves outside of a rank's environment, it must be communicated to
// whichever other rank's environment it crosses into.
class Environment
{
public:
  Environment(MPI_Comm comm,int ndims, vector<int> &Cartdims, vector<int> &Cartcoords,
              int rank, int world_size, int numParticles)
  {
    this->CartCOMM = comm;
    this->rankX = Cartcoords[0];
    this->rankY = Cartcoords[1];
    this->rankZ = Cartcoords[2];
    this->messageNumberSize = 0;

    if (this->rankX+1 < Cartdims[0]) {
      int destCoordArrayX_right[3];
      destCoordArrayX_right[0] = this->rankX+1;
      destCoordArrayX_right[1] = this->rankY;
      destCoordArrayX_right[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayX_right,&this->destRankX_right);
      this->messageNumberSize++;
    } else {
      this->destRankX_right = MPI_PROC_NULL;
    }

    if (this->rankY+1 < Cartdims[1]) {
      int destCoordArrayY_up[3];
      destCoordArrayY_up[0] = this->rankX;
      destCoordArrayY_up[1] = this->rankY+1;
      destCoordArrayY_up[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayY_up,&this->destRankY_up);
      this->messageNumberSize++;
    } else {
      this->destRankY_up = MPI_PROC_NULL;
    }

    if (this->rankX-1 >= 0) {
      int destCoordArrayX_left[3];
      destCoordArrayX_left[0] = this->rankX-1;
      destCoordArrayX_left[1] = this->rankY;
      destCoordArrayX_left[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayX_left,&this->destRankX_left);
      this->messageNumberSize++;
    } else {
      this->destRankX_left = MPI_PROC_NULL;
    }

    if (this->rankY-1 >= 0) {
      int destCoordArrayY_down[3];
      destCoordArrayY_down[0] = this->rankX;
      destCoordArrayY_down[1] = this->rankY-1;
      destCoordArrayY_down[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayY_down,&this->destRankY_down);
      this->messageNumberSize++;
    } else {
      this->destRankY_down = MPI_PROC_NULL;
    }

    if ((this->rankX+1 < Cartdims[0]) && (this->rankY+1 < Cartdims[1])) {
      int destCoordArrayDiag_rightUp[3];
      destCoordArrayDiag_rightUp[0] = this->rankX+1;
      destCoordArrayDiag_rightUp[1] = this->rankY+1;
      destCoordArrayDiag_rightUp[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayDiag_rightUp,&this->destRankDiag_rightUp);
      this->messageNumberSize++;
    } else {
      this->destRankDiag_rightUp = MPI_PROC_NULL;
    }

    if ((this->rankX+1 < Cartdims[0]) && (this->rankY-1 >= 0)) {
      int destCoordArrayDiag_rightDown[3];
      destCoordArrayDiag_rightDown[0] = this->rankX+1;
      destCoordArrayDiag_rightDown[1] = this->rankY-1;
      destCoordArrayDiag_rightDown[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayDiag_rightDown,&this->destRankDiag_rightDown);
      this->messageNumberSize++;
    } else {
      this->destRankDiag_rightDown = MPI_PROC_NULL;
    }

    if ((this->rankX-1 >= 0) && (this->rankY+1 < Cartdims[1])) {
      int destCoordArrayDiag_leftUp[3];
      destCoordArrayDiag_leftUp[0] = this->rankX-1;
      destCoordArrayDiag_leftUp[1] = this->rankY+1;
      destCoordArrayDiag_leftUp[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayDiag_leftUp,&this->destRankDiag_leftUp);
      this->messageNumberSize++;
    } else {
      this->destRankDiag_leftUp = MPI_PROC_NULL;
    }

    if ((this->rankX-1 >= 0) && (this->rankY-1 >= 0)) {
      int destCoordArrayDiag_leftDown[3];
      destCoordArrayDiag_leftDown[0] = this->rankX-1;
      destCoordArrayDiag_leftDown[1] = this->rankY-1;
      destCoordArrayDiag_leftDown[2] = this->rankZ;
      MPI_Cart_rank(this->CartCOMM,destCoordArrayDiag_leftDown,&this->destRankDiag_leftDown);
      this->messageNumberSize++;
    } else {
      this->destRankDiag_leftDown = MPI_PROC_NULL;
    }

    this->factor = CUBE_BOUNDS*1./world_size;
    this->Xbounds = Cartcoords[0]*factor;
    this->Ybounds = Cartcoords[1]*factor;
    this->Zbounds = Cartcoords[2]*factor;
    this->numSend = 0;
    this->numRecv = 0;
    this->startParticleSize = numParticles;
    this->numParticles = numParticles;
    
    particle temp;
    for (int i=0; i<this->numParticles; i++) {
      temp.x = this->Xbounds + factor*fRand(1);
      temp.y = this->Ybounds + factor*fRand(1);
      temp.z = this->Zbounds + factor*fRand(1);
      this->particleVec.push_back(temp);
    }

    for (int i=0; i<ndims; i++) {
      this->dimensionVec.push_back(factor*Cartdims[i]);
    }

    this->rankNumber = rank;
    this->ndims = ndims;
    this->worldSize = world_size;

    this->pBufferX_right.clear();
    this->pBufferX_left.clear();
    this->pBufferY_up.clear();
    this->pBufferY_down.clear();
    this->pBufferDiag_rightUp.clear();
    this->pBufferDiag_rightDown.clear();
    this->pBufferDiag_leftUp.clear();
    this->pBufferDiag_leftDown.clear();

    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    int bLengths[3] = {1, 1, 1};
    MPI_Aint disp[3];
    particle p;
    disp[0] = (MPI_Aint)((long)&p.x - (long)&p);
    disp[1] = (MPI_Aint)((long)&p.y - (long)&p);
    disp[2] = (MPI_Aint)((long)&p.z - (long)&p);

    MPI_Type_create_struct(3, bLengths, disp, types, &this->myStruct);
    MPI_Type_commit(&this->myStruct);
  }

  void Free()
  {
    MPI_Type_free(&myStruct);
  }

  // Find the real 3D coordinates of a particle.
  pair<pair<double, double>, double> revealCoords(int particleNum)
  {
    if (this->numParticles == 0) {
      pair<pair<int, int>, int> p;
      p.first.first = -1.;
      p.first.second = -1.;
      p.second = -1.;
      return p;
    }
    pair<pair<double, double>, double> p;
    p.first.first  = this->particleVec[particleNum%this->numParticles].x;
    p.first.second = this->particleVec[particleNum%this->numParticles].y;
    p.second       = this->particleVec[particleNum%this->numParticles].z;
    return p;
  }

  // How many particles does a rank own?
  int revealNumParticles(void)
  {
    return this->numParticles;
  }

  // Add a certain number of particles to be received into our environment.
  void Add(int count, vector<particle> &transferArray)
  {
    if (count == 0) {
      return;
    }

    particle temp;
    for (int i=0; i<count; i++) {
      temp = transferArray[i];
      this->particleVec.push_back(temp);
    }
    this->numParticles += count;
    this->numRecv      += count;
  }

  // Remove particles from the environment after they have moved away.
  void Remove(int particleNum)
  {
    if (this->numParticles <= 0) {
      return;
    }

    particle temp = this->particleVec[particleNum%this->numParticles];
    this->particleVec[particleNum%this->numParticles] = this->particleVec[this->particleVec.size()-1];
    this->particleVec[this->particleVec.size()-1]     = temp;

    this->particleVec.pop_back();
    this->numParticles--;
    this->numSend++;
  }

  // Only move particles in 2 directions, to restrict us to 8 neighbors instead of 26.
  void move(void)
  {
    if (this->numParticles == 0) {
      return;
    }

    int pLength = this->numParticles;
    for (int particleNum=0; particleNum<pLength; particleNum++) {
      this->particleVec[particleNum%this->numParticles].x += (this->factor*fRand(0));
      this->particleVec[particleNum%this->numParticles].y += (this->factor*fRand(0));

      double &myX = this->particleVec[particleNum%this->numParticles].x;
      double &myY = this->particleVec[particleNum%this->numParticles].y;

      // If a particle escapes off edge of the cubic domain, put it back in.
      if (myX < 0.) {
        myX = this->factor * fRand(1);
      }
      else if (myX >= dimensionVec[0]) {
        myX = dimensionVec[0] - this->factor * fRand(1);
      }

      if (myY < 0) {
        myY = this->factor * fRand(1);
      }
      else if (myY >= dimensionVec[1]) {
        myY = dimensionVec[1] - this->factor * fRand(1);
      }

      // Check whether or not there particles moved away into a valid but different rank. Then:
      //   1) push particle to correct buffer
      //   2) remove from this rank's particleVec

      if ((myX < this->Xbounds) && (myY < this->Ybounds)) {
        pBufferDiag_leftDown.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if ((myX < this->Xbounds) && (myY > this->Ybounds+this->factor)) {
        pBufferDiag_leftUp.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if ((myX > this->Xbounds+this->factor) && (myY > this->Ybounds+this->factor)) {
        pBufferDiag_rightUp.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if ((myX > this->Xbounds+this->factor) && (myY < this->Ybounds)) {
        pBufferDiag_rightDown.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if (((myX >= this->Xbounds) && (myX <this->Xbounds+this->factor)) && (myY < this->Ybounds)) {
        pBufferY_down.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if (((myX >= this->Xbounds) && (myX < this->Xbounds+this->factor)) && (myY > this->Ybounds+this->factor)) {
        pBufferY_up.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if ((myX > this->Xbounds+this->factor) && ((myY >= this->Ybounds) && (myY < this->Ybounds+this->factor))) {
        pBufferX_right.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
      else if ((myX < this->Xbounds) && ((myY >= this->Ybounds) && (myY < this->Ybounds+this->factor))) {
        pBufferX_left.push_back(this->particleVec[particleNum%this->numParticles]);
        Remove(particleNum);
      }
    }
  }

  // Communicate the updated particles to their new ranks.
  // One send for each of the buffers that we have been saving.
  void transfer(void)
  {
    int count;

    vector<particle> sendBufferX_right(this->pBufferX_right.size());
    vector<particle> sendBufferX_left(this->pBufferX_left.size());
    vector<particle> sendBufferDiag_leftDown(this->pBufferDiag_leftDown.size());
    vector<particle> sendBufferY_down(this->pBufferY_down.size());
    vector<particle> sendBufferDiag_rightDown(this->pBufferDiag_rightDown.size());
    vector<particle> sendBufferY_up(this->pBufferY_up.size());
    vector<particle> sendBufferDiag_rightUp(this->pBufferDiag_rightUp.size());
    vector<particle> sendBufferDiag_leftUp(this->pBufferDiag_leftUp.size());

    MPI_Request sendRequest[8];
    MPI_Status statArray[8];

    for (unsigned int i=0; i<this->pBufferX_right.size(); i++) {
      sendBufferX_right[i] = this->pBufferX_right[i];
    }
    MPI_Isend(&sendBufferX_right[0], this->pBufferX_right.size(), this->myStruct,
              this->destRankX_right, 0, this->CartCOMM, &sendRequest[0]);

    for (unsigned int i=0; i<this->pBufferX_left.size(); i++) {
      sendBufferX_left[i] = this->pBufferX_left[i];
    }
    MPI_Isend(&sendBufferX_left[0], this->pBufferX_left.size(), this->myStruct,
              this->destRankX_left, 0, this->CartCOMM, &sendRequest[1]);

    for (unsigned int i=0; i<this->pBufferDiag_leftDown.size(); i++) {
      sendBufferDiag_leftDown[i] = this->pBufferDiag_leftDown[i];
    }
    MPI_Isend(&sendBufferDiag_leftDown[0], this->pBufferDiag_leftDown.size(), this->myStruct,
              this->destRankDiag_leftDown, 0, this->CartCOMM, &sendRequest[2]);

    for (unsigned int i=0; i<this->pBufferY_down.size(); i++) {
      sendBufferY_down[i] = this->pBufferY_down[i];
    }
    MPI_Isend(&sendBufferY_down[0], this->pBufferY_down.size(), this->myStruct,
              this->destRankY_down, 0, this->CartCOMM, &sendRequest[3]);

    for (unsigned int i=0; i<this->pBufferDiag_rightDown.size(); i++) {
      sendBufferDiag_rightDown[i] = this->pBufferDiag_rightDown[i];
    }
    MPI_Isend(&sendBufferDiag_rightDown[0], this->pBufferDiag_rightDown.size(), this->myStruct,
              this->destRankDiag_rightDown, 0, this->CartCOMM, &sendRequest[4]);

    for (unsigned int i=0; i<this->pBufferY_up.size(); i++) {
      sendBufferY_up[i] = this->pBufferY_up[i];
    }
    MPI_Isend(&sendBufferY_up[0], this->pBufferY_up.size(), this->myStruct,
              this->destRankY_up, 0, this->CartCOMM, &sendRequest[5]);

    for (unsigned int i=0; i<this->pBufferDiag_rightUp.size(); i++) {
      sendBufferDiag_rightUp[i] = this->pBufferDiag_rightUp[i];
    }
    MPI_Isend(&sendBufferDiag_rightUp[0], this->pBufferDiag_rightUp.size(), this->myStruct,
              this->destRankDiag_rightUp, 0, this->CartCOMM, &sendRequest[6]);

    for (unsigned int i=0; i<this->pBufferDiag_leftUp.size(); i++) {
      sendBufferDiag_leftUp[i] = this->pBufferDiag_leftUp[i];
    }
    MPI_Isend(&sendBufferDiag_leftUp[0], this->pBufferDiag_leftUp.size(), this->myStruct,
              this->destRankDiag_leftUp, 0, this->CartCOMM, &sendRequest[7]);

    MPI_Waitall(8,&sendRequest[0], &statArray[0]); 
    for (int i=0; i<this->messageNumberSize; i++) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, this->CartCOMM, &statArray[i]);
      MPI_Get_count(&statArray[i], this->myStruct, &count);
      vector<particle> recvBuffer(count);
      MPI_Recv(&recvBuffer[0], count, this->myStruct, MPI_ANY_SOURCE, MPI_ANY_TAG, this->CartCOMM, &statArray[i]);
      Add(count, recvBuffer);
    }

    this->pBufferX_right.clear();
    this->pBufferX_left.clear();
    this->pBufferDiag_leftDown.clear();
    this->pBufferY_down.clear();
    this->pBufferDiag_rightDown.clear();
    this->pBufferY_up.clear();
    this->pBufferDiag_rightUp.clear();
    this->pBufferDiag_leftUp.clear();
  }

  int getNumSend(void)
  {
    return this->numSend;
  }

  int getNumRecv(void)
  {
    return this->numRecv;
  }

private:
  MPI_Datatype myStruct;
  MPI_Comm CartCOMM;
  int numParticles, startParticleSize;
  int rankNumber, worldSize;
  int numSend, numRecv;
  int rankX, rankY, rankZ;
  double Xbounds, Ybounds, Zbounds;
  double factor;
  int ndims;
  int messageNumberSize;

  vector<int> dimensionVec;
  vector<particle> particleVec;

  // Particle buffers per neighboring rank
  vector<particle> pBufferX_left;
  vector<particle> pBufferX_right;
  vector<particle> pBufferY_up;
  vector<particle> pBufferY_down;
  vector<particle> pBufferDiag_leftUp;
  vector<particle> pBufferDiag_leftDown;
  vector<particle> pBufferDiag_rightUp;
  vector<particle> pBufferDiag_rightDown;

  // Neighboring ranks
  int destRankX_right, destRankY_up, destRankX_left, destRankY_down;
  int destRankDiag_rightUp, destRankDiag_rightDown, destRankDiag_leftUp, destRankDiag_leftDown;
};


int main(int argc, char **argv)
{
  int rank, size;
  MPI_Comm CubeCOMM;
  vector<int> ndims(3), periodArray(3), coordsArray(3);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int i=0; i<3; i++) {
    ndims[i]       = 0;
    periodArray[i] = 0;
  }
  MPI_Dims_create(size, 3, &ndims[0]);

  MPI_Cart_create(MPI_COMM_WORLD, 3, &ndims[0], &periodArray[0], 1, &CubeCOMM);
  MPI_Comm_rank(CubeCOMM, &rank);
  MPI_Cart_coords(CubeCOMM, rank, 3, &coordsArray[0]);
  srand(time(NULL) + rank); // Unique random seed for each rank

  Environment myEnvironment(CubeCOMM, 3, ndims, coordsArray, rank, size, 500);

  for (int i=0; i<10; i++) {
    myEnvironment.move();
    myEnvironment.transfer();

    cout << "Iter " << i+1 << ", rank " << rank << " sent " << myEnvironment.getNumSend();
    cout << " and received " << myEnvironment.getNumRecv() << " for a net difference of ";
    cout << myEnvironment.getNumSend() - myEnvironment.getNumRecv() << endl;

    MPI_Barrier(CubeCOMM);
  }

  myEnvironment.Free();
  MPI_Finalize();
  return 0;
}
