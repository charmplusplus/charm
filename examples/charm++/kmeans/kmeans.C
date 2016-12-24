#include "kmeans.decl.h"

#include "rand48_replacement.h"
#include <vector>

#define XOFFSET 0
#define YOFFSET 1
#define SIZEOFFSET 2
#define SIZEPERPOINT 3

/* readonly */ int n;
/* readonly */ int k;
/* readonly */ int numCharesX;
/* readonly */ int numCharesY;


class Main : public CBase_Main {
private:
  Point* means;
  Point* oldMeans;

public:
  CProxy_Domain domain;

  Main(CkArgMsg* m) {
    if (m->argc < 4) {
      CkPrintf("%s [number of total points] [number of clusters] [number of chares in each dimension]\n", m->argv[0]);
      CkAbort("Insufficient number of arguments");
    }

    n = atoi(m->argv[1]);
    k = atoi(m->argv[2]);

    if (n < k) {
      CkAbort("The number of total points must be at least as large as k\n");
    }

    numCharesX = atoi(m->argv[3]);
    numCharesY = atoi(m->argv[3]);

    domain = CProxy_Domain::ckNew(numCharesX, numCharesY);

    // Generate some random points as initial means
    means = new Point[k];
    srand48((int)(10000*CkWallTimer()));
    for (int i = 0; i < k; ++i) {
      means[i].x = drand48();
      means[i].y = drand48();
    }

    oldMeans = new Point[k];
    memcpy(oldMeans, means, k*sizeof(Point));

    thisProxy.compute();
  }

  void compute() {
    CkReductionMsg* msg;
    int iteration = 0;

    while (true) {
      CkPrintf("Iteration %d\n", iteration++);

      // Suspend until each chare finds the closest mean for its points
      domain.findClusters(k, means, CkCallbackResumeThread((void*&)msg));

      // Now the new data is in msg, so use it to calculate the new candidate means
      double* data = (double*)msg->getData();
      for (int i = 0; i < k; ++i) {
        int dataIndex = SIZEPERPOINT*i;
        if (data[dataIndex + SIZEOFFSET] > 0) {
          means[i].x = data[dataIndex + XOFFSET] / data[dataIndex + SIZEOFFSET];
          means[i].y = data[dataIndex + YOFFSET] / data[dataIndex + SIZEOFFSET];
        }
      }

      // Now that we've found the new means, check if they're the same as before
      bool match = true;
      for (int i = 0; i < k; ++i) {
        if (means[i].x != oldMeans[i].x || means[i].y != oldMeans[i].y) {
          match = false;
          break;
        }
      }

      // If we've found them, exit
      if (match) {
        thisProxy.done();
        return;
      }

      // Otherwise, set the new means as the old and repeat
      memcpy(oldMeans, means, k*sizeof(Point));
    }
  }

  void done() {
    // Report the means we've found
    for (int i = 0; i < k; ++i)
    {
      CkPrintf("%d, (%f, %f)\n", i, means[i].x, means[i].y);
    }

    CkExit();
  }
};

class Domain : public CBase_Domain {
private:
  Point *points;
  int numPoints;

  int findClosest(Point& datum, Point* candidates) {
    double minDistance = datum.distance2(candidates[0]);
    int closest = 0;

    for (int i = 1; i < k; ++i) {
      double currentDistance = datum.distance2(candidates[i]);
      if (currentDistance < minDistance) {
        minDistance = currentDistance;
        closest = i;
      }
    }

    return closest;
  }

public:
  Domain() {
    srand48((int)(10000*CkWallTimer()) + thisIndex.x * numCharesX + thisIndex.y);

    numPoints = n / (numCharesX * numCharesY);
    numPoints += (n % (numCharesX * numCharesY) < (thisIndex.x * numCharesX + thisIndex.y)) ? 1 : 0;

    points = new Point[numPoints];

    // Generate random points
    for (int i = 0; i < numPoints; ++i) {
      points[i].x = (drand48() + thisIndex.x) / numCharesX;
      points[i].y = (drand48() + thisIndex.y) / numCharesY;
    }
  }

  Domain(CkMigrateMessage* m) { }

  void findClusters(int k, Point means[], CkCallback &cb) {
    std::vector<double> result(SIZEPERPOINT*k, 0); // For each candidate mean, create entry for x, y, and count

    for (int i = 0; i < numPoints; ++i) {
      int closest = findClosest(points[i], means);
      closest *= SIZEPERPOINT; // Go to corresponding index in result
      result[closest + XOFFSET] += points[i].x;
      result[closest + YOFFSET] += points[i].y;
      result[closest + SIZEOFFSET]++;
    }

    contribute(SIZEPERPOINT*k*sizeof(double), &result[0], CkReduction::sum_double, cb);
  }
};

#include "kmeans.def.h"
