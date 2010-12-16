// -*- mode: c++; tab-width: 4 -*-

// When running 1D, make NEPP = COL1
// When running 2D, same
// When running 3D, make NEPP = subset of COL1

#include "nepp.h"
#include "msa/msa.h"

class XYZ { // coords, forces
public:
    double x, y, z;
    XYZ() { x = y = z = 0.0; }
    XYZ(const int rhs) { x = y = z = (double)rhs; } // identity value
    XYZ& operator+= (const XYZ& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this;}
    XYZ& negate() { x = -x; y=-y; z=-z; return *this;}
};
PUPbytes(XYZ);

class AtomInfo {
public:
    double mass, charge;
  AtomInfo() : mass(0.0), charge(0.0) { }
    AtomInfo(const int rhs) { mass = charge = (double)rhs; } // identity value
    AtomInfo& operator+= (const AtomInfo& rhs) { } // we're not calling accumulate on this
};
PUPbytes(AtomInfo);

typedef MSA::MSA1D<XYZ, DefaultEntry<XYZ,false>, NEPP> XyzMSA;
typedef MSA::MSA1D<AtomInfo, DefaultEntry<AtomInfo,false>, NEPP> AtomInfoMSA;
typedef MSA::MSA2D<bool, DefaultEntry<bool,false>, NEPP, MSA_ROW_MAJOR> NeighborMSA;

#include "moldyn.decl.h"

#include <assert.h>
#include <math.h>
#include "params.h"

const double epsilon = 0.00000001;
inline int notequal(double v1, double v2)
{
    return (fabs(v1 - v2) > epsilon);
}

class moldyn : public CBase_moldyn
{
protected:
    double start_time;
    CProxy_WorkerArray workers;
    int reallyDone;

public:
    moldyn(CkArgMsg* m)
    {
        // Usage: a.out [number_of_worker_threads [max_bytes]]
        if(m->argc >1 ) NUM_WORKERS=atoi(m->argv[1]);
        if(m->argc >2 ) NUM_ATOMS=atoi(m->argv[2]);
        if(m->argc >3 ) CACHE_SIZE_BYTES=atoi(m->argv[3]);
        if(m->argc >4 ) detailedTimings= ((atoi(m->argv[4])!=0)?true:false) ; // 1D, 2D, 3D
        delete m;
        reallyDone = 0;

        XyzMSA coords(NUM_ATOMS, NUM_WORKERS, CACHE_SIZE_BYTES);
        XyzMSA forces(NUM_ATOMS, NUM_WORKERS, CACHE_SIZE_BYTES);
        AtomInfoMSA atominfo(NUM_ATOMS, NUM_WORKERS, CACHE_SIZE_BYTES);
        NeighborMSA nbrList(NUM_ATOMS, NUM_ATOMS, NUM_WORKERS, CACHE_SIZE_BYTES);

        workers = CProxy_WorkerArray::ckNew(coords, forces, atominfo, nbrList, NUM_WORKERS, NUM_WORKERS);
        workers.ckSetReductionClient(new CkCallback(CkIndex_moldyn::done(NULL), thisProxy));

        start_time = CkWallTimer();
        workers.Start();
    }

    // This method gets called twice, and should only terminate the
    // second time.
    void done(CkReductionMsg* m)
    {
        int *ip = (int*)m->getData();
        bool prefetchWorked = (*ip==0);
        delete m;

        if (reallyDone == 0) {
            workers.Kontinue();
            reallyDone++;

            double end_time = CkWallTimer();

            const char TAB = '\t';

            char hostname[100];
            gethostname(hostname, 100);

            ckout << CkNumPes() << TAB
                  << NUM_WORKERS << TAB
                  << "nepp " << NEPP << TAB
				  << "atom " << NUM_ATOMS << TAB
                  << end_time - start_time << TAB
                  << CACHE_SIZE_BYTES << TAB
                  << (runPrefetchVersion? (prefetchWorked?"Y":"N"): "U") << " "
                  << hostname
                  << endl;

        } else {
            CkExit();
        }
    }
};

// Returns start and end
void GetMyIndices(unsigned int maxIndex, unsigned int myNum, unsigned int numWorkers,
                  unsigned int& start, unsigned int& end)
{
    int rangeSize = maxIndex / numWorkers;
    if(myNum < maxIndex % numWorkers)
    {
        start = myNum * (rangeSize + 1);
        end = start + rangeSize;
    }
    else
    {
        start = myNum * rangeSize + maxIndex % numWorkers;
        end = start + rangeSize - 1;
    }
}

class WorkerArray : public CBase_WorkerArray
{
private:
    // prefetchWorked keeps track of whether the prefetches succeeded or not.
    bool prefetchWorked;
    CkVec<double> times;
    CkVec<const char*> description;

    // ================================================================
    // 2D calculations

    inline int numWorkers2D() {
        static int n = 0;

        if (n==0) {
            n = (int)(sqrt(numWorkers));
            CkAssert(n*n == numWorkers);
        }

        return n;
    }

    // Convert a 1D ChareArray index into a 2D x dimension index
    inline unsigned int toX() {
        return thisIndex/numWorkers2D();
    }
    // Convert a 1D ChareArray index into a 2D y dimension index
    inline unsigned int toY() {
        return thisIndex%numWorkers2D();
    }

    // ================================================================

protected:
    XyzMSA coords;
    XyzMSA forces;
    AtomInfoMSA atominfo;
	AtomInfoMSA::Read rAtominfo;
    NeighborMSA nbrList;

    unsigned int numAtoms, numWorkers;

    void EnrollArrays()
    {
        coords.enroll(numWorkers);
        forces.enroll(numWorkers);
        atominfo.enroll(numWorkers);
        nbrList.enroll(numWorkers);
    }

    void FillArrays()
    {
        /*
        // fill in our portion of the array
        unsigned int rowStart, rowEnd, colStart, colEnd;
        GetMyIndices(rows1, thisIndex, numWorkers, rowStart, rowEnd);
        GetMyIndices(cols2, thisIndex, numWorkers, colStart, colEnd);

        // fill them in with 1
        for(unsigned int r = rowStart; r <= rowEnd; r++)
            for(unsigned int c = 0; c < cols1; c++)
                arr1.set(r, c) = 1.0;

        for(unsigned int c = colStart; c <= colEnd; c++)
            for(unsigned int r = 0; r < rows2; r++)
                arr2.set(r, c) = 1.0;
        */
    }

    XYZ calculateForce(const XYZ &coordsi, const AtomInfo &atominfoi, const XYZ &coordsj, const AtomInfo &atominfoj)
    {
        XYZ result;
        return result;
    }

    XYZ integrate(const AtomInfo &atominfok, const XYZ &forcesk)
    {
        XYZ result;
        return result;
    }

    double distance(unsigned int i, unsigned int j)
    {
        return 0;
    }

  double distance(const XYZ &a, const XYZ &b)
  {
	double dx = a.x - b.x,
	  dy = a.y - b.y,
	  dz = a.z - b.z;
	return sqrt(dx*dx + dy*dy + dz*dz);
  }

  void PlimptonMD(XyzMSA::Handle hCoords, XyzMSA::Handle hForces, NeighborMSA::Handle hNbr)
  {
	unsigned int i_start, i_end, j_start, j_end;
	GetMyIndices(NUM_ATOMS-1, toX(), numWorkers2D(), i_start, i_end);
	GetMyIndices(NUM_ATOMS-1, toY(), numWorkers2D(), j_start, j_end);

	XyzMSA::Read rCoords = hCoords.syncToRead();
	NeighborMSA::Read rNbr = hNbr.syncToRead();

	for (unsigned int timestep = 0; timestep < NUM_TIMESTEPS; timestep++) {
	  // Force calculation for a section of the interaction matrix
	  XyzMSA::Accum aForces = hForces.syncToAccum();
	  for (unsigned int i = i_start; i< i_end; i++)
		for (unsigned int j = j_start; j< j_end; j++)
		  if (rNbr(i,j)) {
			XYZ force = calculateForce(rCoords(i),
									   rAtominfo(i),
									   rCoords(j),
									   rAtominfo(j));
			aForces(i) += force;
			aForces(j) += force.negate();
		  }

	  // Movement Integration for our subset of atoms
	  unsigned int myAtomsBegin, myAtomsEnd;
	  XyzMSA::Read rForces = aForces.syncToRead();
	  XyzMSA::Write wCoords = rCoords.syncToWrite();
	  for (unsigned int k = myAtomsBegin; k<myAtomsEnd; k++)
		wCoords(k) = integrate(rAtominfo(k), rForces(k));

	  // Neighbor list recalculation for our section of the interaction matrix
	  rCoords = wCoords.syncToRead();
	  if  (timestep % 8 == 0) { // update neighbor list every 8 steps
		NeighborMSA::Write wNbr = rNbr.syncToWrite();
		for (unsigned int i = i_start; i< i_end; i++)
		  for (unsigned int j = j_start; j< j_end; j++)
			if (distance(rCoords(i), rCoords(j)) < CUTOFF_DISTANCE) {
			  wNbr.set(i,j) = true;
			  wNbr.set(j,i) = true;
			} else {
			  wNbr.set(i,j) = false;
			  wNbr.set(j,i) = false;
			}
		rNbr = wNbr.syncToRead();
	  }

	  hForces = rForces;
	}
  }

    void TestResults(bool prod_test=true)
    {
        /*
        int errors = 0;
        bool ok=true;

        // verify the results, print out first error only
        ok=true;
        for(unsigned int r = 0; ok && r < rows1; r++) {
            for(unsigned int c = 0; ok && c < cols1; c++) {
                if(notequal(arr1.get(r, c), 1.0)) {
                    ckout << "[" << CkMyPe() << "," << thisIndex << "] arr1 -- Illegal element at (" << r << "," << c << ") " << arr1.get(r,c) << endl;
                    ok=false;
                    errors++;
                }
            }
        }

        ok=true;
        for(unsigned int c = 0; ok && c < cols2; c++) {
            for(unsigned int r = 0; ok && r < rows2; r++) {
                if(notequal(arr2.get(r, c), 1.0)) {
                    ckout << "[" << CkMyPe() << "," << thisIndex << "] arr2 -- Illegal element at (" << r << "," << c << ") " << arr2.get(r,c) << endl;
                    ok=false;
                    errors++;
                }
            }
        }

        //arr1.FreeMem();
        //arr2.FreeMem();

        if(prod_test)
        {
            ok = true;
            for(unsigned int c = 0; ok && c < cols2; c++) {
                for(unsigned int r = 0; ok && r < rows1; r++) {
                    if(notequal(prod.get(r,c), 1.0 * cols1)) {
                        ckout << "[" << CkMyPe() << "] result  -- Illegal element at (" << r << "," << c << ") " << prod.get(r,c) << endl;
                        ok=false;
                        errors++;
                    }
                }
            }
        }

        if (errors!=0) CkAbort("Incorrect array elements detected!");
        */
    }

    void Contribute()
    {
        int dummy = prefetchWorked?0:1;
        contribute(sizeof(int), &dummy, CkReduction::sum_int);
    }

public:
    WorkerArray(const XyzMSA &coords_, const XyzMSA &forces_, AtomInfoMSA &atominfo_,
                NeighborMSA &nbrList_, unsigned int numWorkers_)
        : coords(coords_), forces(forces_), atominfo(atominfo_), nbrList(nbrList_),
          numWorkers(numWorkers_), prefetchWorked(false), numAtoms(coords.length())
    {
        // ckout << "w" << thisIndex << ":" << rows1 << " " << cols1 << " " << cols2 << endl;
        times.push_back(CkWallTimer());
        description.push_back("constr");
    }

    WorkerArray(CkMigrateMessage* m) {}

    ~WorkerArray()
    {
    }

    void Start()
    {
        times.push_back(CkWallTimer()); // 1
        description.push_back("   start");

        EnrollArrays();
        times.push_back(CkWallTimer()); // 2
        description.push_back("   enroll");

        if(verbose) ckout << thisIndex << ": filling" << endl;
        FillArrays();
        times.push_back(CkWallTimer()); // 3
        description.push_back("  fill");

        if(verbose) ckout << thisIndex << ": syncing" << endl;
        //SyncArrays();
        times.push_back(CkWallTimer()); // 4
        description.push_back("    sync");

        if (do_test) TestResults(0);

        if(verbose) ckout << thisIndex << ": product" << endl;

        PlimptonMD(coords.getInitialWrite(), forces.getInitialWrite(), nbrList.getInitialWrite());
        times.push_back(CkWallTimer()); // 5
        description.push_back("    work");

        Contribute();
    }

    void Kontinue()
    {
        times.push_back(CkWallTimer()); // 6
        description.push_back("    redn");

        if(verbose) ckout << thisIndex << ": testing" << endl;
        if (do_test) TestResults();
        times.push_back(CkWallTimer()); // 7
        description.push_back("    test");
        Contribute();

        if (detailedTimings) {
            if (thisIndex == 0) {
                for(int i=1; i<description.length(); i++)
                    ckout << description[i] << " ";
                ckout << endl;
            }
            ckout << "w" << thisIndex << ":";
            for(int i=1; i<times.length(); i++)
                ckout << times[i]-times[i-1] << " ";
            ckout << endl;
        }
    }
};

#include "moldyn.def.h"
