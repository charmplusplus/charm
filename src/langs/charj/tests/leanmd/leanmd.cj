
#define HYDROGEN_MASS         (1.67 * pow( 10.0,-24))
#define VDW_A                        (1.60694452 * pow(10.0, -134))
#define VDW_B            (1.031093844 * pow(10.0, -77))

#define ENERGY_VAR        (1.0 * pow(10.0,-5))
#define PARTICLES_PER_CELL   300

#define DEFAULT_DELTA         1// interact femtoseconds

#define DEFAULT_FIRST_LDB     20
#define DEFAULT_LDB_PERIOD    20
#define DEFAULT_FT_PERIOD     100000

#define KAWAY_X               1
#define KAWAY_Y               1
#define KAWAY_Z               1
#define NBRS_X                (2*KAWAY_X+1)
#define NBRS_Y                  (2*KAWAY_Y+1)
#define NBRS_Z                  (2*KAWAY_Z+1)
#define NUM_NEIGHBORS        (NBRS_X * NBRS_Y * NBRS_Z)

#define CELLARRAY_DIM_X       3
#define CELLARRAY_DIM_Y       3
#define CELLARRAY_DIM_Z       3
#define PTP_CUT_OFF             12
#define CELL_MARGIN             4  
#define CELL_SIZE_Y             ((PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Y)
#define CELL_SIZE_Z             ((PTP_CUT_OFF + CELL_MARGIN)/KAWAY_Z)
#define CELL_ORIGIN_X             0
#define CELL_ORIGIN_Y             0
#define CELL_ORIGIN_Z             0

#define MIGRATE_STEPCOUNT    20
#define DEFAULT_FINALSTEPCOUNT 1001
#define MAX_VELOCITY      30.0

#define WRAP_X(a)(((a)+cellArrayDimX)%cellArrayDimX)
#define WRAP_Y(a)(((a)+cellArrayDimY)%cellArrayDimXDimY)
#define WRAP_Z(a)(((a)+cellArrayDimZ)%cellArrayDimZ)
#define BLOCK_SIZE 512

extern srand48;
extern drand48;
extern pow;
extern fabs;
extern abs;
extern atoi;
extern CkSectionInfo;
extern ParticleDataMsg;

readonly Main@ mainProxy;       //central controller
//readonly Cell@ cellArray;       //array that houses atoms
//readonly Compute@ computeArray; //computational kernels
//readonly CkGroupID mCastGrpID;  //multicast group handle

readonly int cellArrayDimX;         // X dimension of the Cell array
readonly int cellArrayDimY;         // Y dimension of the Cell array
readonly int cellArrayDimZ;         // Z dimension of the Cell array
readonly int finalStepCount;        // number of steps in the simulaion
readonly int firstLdbStep;          // begin load balancing after this many steps
readonly int ldbPeriod;             // load balancing period

class vec3 {
    double x, y, z;
    void copy(vec3 v2) {
        x = v2.x; y = v2.y; z = v2.z;
    }
    void mult(double s) {
        x *= s; y *= s; z *= s;        
    }
    void sub(vec3 v2) {
        x -= v2.x; y -= v2.y; z -= v2.z;
    }
    void add(vec3 v2) {
        x += v2.x; y += v2.y; z += v2.z;
    }
    double dot(vec3 v2) {
        return x*v2.x + y*v2.y + z*v2.z;
    }
}

multicast_message ParticleDataMsg {
    Array<vec3> part;
}

class Particle {
    double mass;
    vec3 pos;
    vec3 acc;
    vec3 vel;
}

public mainchare Main {
    public entry Main(CkArgMsg[~]@ m) {
        CkPrintf("\nLENNARD JONES MOLECULAR DYNAMICS START UP ...\n");

        //set variable values to a default set
        cellArrayDimX = CELLARRAY_DIM_X;
        cellArrayDimY = CELLARRAY_DIM_Y;
        cellArrayDimZ = CELLARRAY_DIM_Z;
        finalStepCount = DEFAULT_FINALSTEPCOUNT;
        firstLdbStep = DEFAULT_FIRST_LDB;
        ldbPeriod = DEFAULT_LDB_PERIOD;
/*

        mainProxy = thisProxy;

        //branch factor for spanning tree of multicast
        int bFactor = 4;
        //creating the multicast spanning tree
        mCastGrpID = CkMulticastMgr@.ckNew(bFactor);

        int numPes = CkNumPes();
        int currPe = -1;
        int cur_arg = 1;

        CkPrintf("\nInput Parameters...\n");

        //read user parameters
        //number of celles/cells in each dimension
        if (m.argc > cur_arg) {
            cellArrayDimX = atoi(m.argv[cur_arg++]);
        }
            cellArrayDimY=atoi(m.argv[cur_arg++]);
            cellArrayDimZ=atoi(m.argv[cur_arg++]);
            CkPrintf("Cell Array Dimension X:%d Y:%d Z:%d of size %d %d %d\n",cellArrayDimX,cellArrayDimY,cellArrayDimZ,CELL_SIZE_X,CELL_SIZE_Y,CELL_SIZE_Z);
        }

        //number of steps in simulation
        if (m.argc > cur_arg) {
            finalStepCount=atoi(m.argv[cur_arg++]);
            CkPrintf("Final Step Count:%d\n",finalStepCount);
        }

        //step after which load balancing starts
        if (m.argc > cur_arg) {
            firstLdbStep=atoi(m.argv[cur_arg++]);
            CkPrintf("First LB Step:%d\n",firstLdbStep);
        }

        //periodicity of load balancing
        if (m.argc > cur_arg) {
            ldbPeriod=atoi(m.argv[cur_arg++]);
            CkPrintf("LB Period:%d\n",ldbPeriod);
        }

        //initializing the 3D cell array
        cellArray = CProxy_Cell::ckNew();
        for (int x=0; x<cellArrayDimX; x++)
            for (int y=0; y<cellArrayDimY; y++)
                for (int z=0; z<cellArrayDimZ; z++) {
                    pe = (++currPe) % numPes;
                    cellArray(x, y, z).insert(pe);
                }
        cellArray.doneInserting();

        CkPrintf("\nCells: %d X %d X %d .... created\n", cellArrayDimX, cellArrayDimY, cellArrayDimZ);

        //initializing the 6D compute array
        computeArray = CProxy_Compute::ckNew();
        for (int x=0; x<cellArrayDimX; x++)
            for (int y=0; y<cellArrayDimY; y++)
                for (int z=0; z<cellArrayDimZ; z++)
                    cellArray(x, y, z).createComputes();

        thisProxy@run();
        delete m;
        */
    }

    entry void run() {
        /*
        when computesCreated() {
            computeArray@doneInserting();
            CkPrintf("Computes: %d .... created\n", (NUM_NEIGHBORS/2+1)*cellArrayDimX*cellArrayDimY*cellArrayDimZ);
            cellArray@createSection();
        }
        when sectionsCreated() {
            CkPrintf("Multicast sections .... created\n");
            CkPrintf("Starting simulation .... \n\n");
            cellArray@run();
            computeArray@run();
        }
        when energySumP(double fP, double iP), energySumK(double fK, double iK) {
            if(abs(fP + fK - iP - iK)>ENERGY_VAR) {
                CkPrintf("Energy value has varied significantly from %E to %E\n",iP+iK,fP+fK);
                CkPrintf("\nEnergy conservation test failed for maximum allowed variation of %E units.\nSIMULATION UNSUCCESSFULL\n",ENERGY_VAR);  
            } else {
                CkPrintf("\nEnergy conservation test passed for maximum allowed variation of %E units.\nSIMULATION SUCCESSFULL \n",ENERGY_VAR);
            }
            CkExit();
        }
        */
    }
}

chare_array [3d] Cell {
    Array<Particle> particles;
    int stepCount;
    int myNumParts;
    int inbrs;
    int stepTime;
    int updateCount;
    Array<double> energy;
//    Compute[%]@ mCastSecProxy;

    entry Cell() {
    /*
        int i;
        inbrs = NUM_NEIGHBORS;

        myNumParts = PARTICLES_PER_CELL;
        // starting random generator
        srand48(thisIndex.x+cellArrayDimX*(thisIndex.y+thisIndex.z*cellArrayDimY));

        // Particle initialization
        particles = new Array<Particle>([myNumParts]);
        for (i=0; i < myNumParts; i++) {
            particles[i] = new Particle();
            particles[i].mass = HYDROGEN_MASS;

            //FIXME: for some reason these array accesses are not type-checked to
            // be PointerType. Figure out why and fix.

            //give random values for position and velocity
            particles[i].pos.x = drand48() * CELL_SIZE_X + thisIndex.x * CELL_SIZE_X;
            particles[i].pos.y = drand48() * CELL_SIZE_Y + thisIndex.y * CELL_SIZE_Y;
            particles[i].pos.z = drand48() * CELL_SIZE_Z + thisIndex.z * CELL_SIZE_Z;
            particles[i].vel.x = (drand48() - 0.5) * .2 * MAX_VELOCITY;
            particles[i].vel.y = (drand48() - 0.5) * .2 * MAX_VELOCITY;
            particles[i].vel.z = (drand48() - 0.5) * .2 * MAX_VELOCITY;
        }

        stepCount = 1;
        updateCount = 0;
        stepTime = 0; 
        energy = new Array<double>([2]);
        energy[0] = energy[1] = 0;
    */
    }

/*
    entry void createComputes() {
        int num;  

        int x = thisIndex.x;
        int y = thisIndex.y;
        int z = thisIndex.z;
        int px1, py1, pz1, dx, dy, dz, px2, py2, pz2;

        // For Round Robin insertion
        int numPes = CkNumPes();
        int currPe = CkMyPe();

        computesList = new Array<int, 2>([inbrs, 6]);
        for (num=0; NUM_NEIGHBORS<inbrs; num++) {
            dx = num / (NBRS_Y * NBRS_Z)            - NBRS_X/2;
            dy = (num % (NBRS_Y * NBRS_Z)) / NBRS_Z   - NBRS_Y/2;
            dz = num % NBRS_Z                       - NBRS_Z/2;

            if (num >= inbrs/2){
                px1 = x + 2;
                px2 = x+dx+2;
                py1 = y + 2;
                py2 = y+dy+2;
                pz1 = z + 2;
                pz2 = z+dz+2;
                computeArray(px1, py1, pz1, px2, py2, pz2).insert((++currPe)%numPes);
                computesList[num][0] = px1; computesList[num][1] = py1; computesList[num][2] = pz1; 
                computesList[num][3] = px2; computesList[num][4] = py2; computesList[num][5] = pz2;
            }
            else {
                // these computes will be created by pairing celles
                px2 = WRAP_X(x+dx);
                py2 = WRAP_Y(y+dy);
                pz2 = WRAP_Z(z+dz);
                px1 = x;
                py1 = y;
                pz1 = z; 
                px1 = px2 - dx + 2;
                px2 = px2+2;
                py1 = py2 - dy + 2;
                py2 = py2+2;
                pz1 = pz2 - dz + 2;
                pz2 = pz2+2;
                computesList[num][0] = px2; computesList[num][1] = py2; computesList[num][2] = pz2; 
                computesList[num][3] = px1; computesList[num][4] = py1; computesList[num][5] = pz1;
            }
        } // end of for loop
        contribute(0,CkReduction.nop,CkCallback(Main.computesCreated,mainProxy));
    }

    entry void createSection() {
        CkVec<CkArrayIndex6D> elems;
        //create a vector list of my computes
        for (int num=0; num<inbrs; num++)
            elems.push_back(CkArrayIndex6D(computesList[num][0], computesList[num][1], computesList[num][2], computesList[num][3], computesList[num][4], computesList[num][5]));

        CkArrayID computeArrayID = computeArray.ckGetArrayID();
        //knit the computes into a section
        mCastSecProxy = Compute[%]@ckNew(computeArrayID, elems.getVec(), elems.size()); 

        //delegate the communication responsibility for this section to multicast library
        CkMulticastMgr mCastGrp = CkMulticastMgr@(mCastGrpID).ckLocalBranch();
        mCastSecProxy.ckSectionDelegate(mCastGrp);
        mCastGrp.setReductionClient(mCastSecProxy, new CkCallback(Cell.reduceForces, thisProxy(thisIndex.x, thisIndex.y, thisIndex.z)));
        contribute(0,CkReduction.nop,CkCallback(Main.sectionsCreated,mainProxy));
    }

    void sendPositions() {
        //create the particle and control message to be sent to computes
        Array<vec3> particle_data = new Array<vec3>([particles.size(0)]);
        for (int i = 0; i < particles.size(0); i++)
            particle_data[i].copy(particles[i].pos);
        ParticleDataMsg m = new ParticleDataMsg();
        m.part = particle_data;
        mCastSecProxy@calculateForces(m);
    }

    void migrateParticles() {
        int i, x1, y1, z1;
        vec3 dest = new vec3();
        Array<Particle> outgoing = new Array<Particle>([inbrs]);
        for(i=0; i<inbrs; i++) outgoing[i] = new Particle();

        for(i=0; i<particles.size(0); i++) {
            dest = migrateToCell(particles[i], x1, y1, z1);
            x1 = dest.x; y1 = dest.y; z1 = dest.z;
            delete dest;
            if(x1!=0 || y1!=0 || z1!=0) {
                outgoing[(x1+1)*NBRS_Y*NBRS_Z + (y1+1)*NBRS_Z + (z1+1)][i].copy(wrapAround(particles[i]));
            }
        }
        for(int num=0; num<inbrs; num++) {
            x1 = num / (NBRS_Y * NBRS_Z)            - NBRS_X/2;
            y1 = (num % (NBRS_Y * NBRS_Z)) / NBRS_Z - NBRS_Y/2;
            z1 = num % NBRS_Z                       - NBRS_Z/2;
            cellArray(WRAP_X(thisIndex.x+x1), WRAP_Y(thisIndex.y+y1), WRAP_Z(thisIndex.z+z1))@receiveParticles(outgoing[num]);
        }

        for(i=0; i<inbrs; i++) delete outgoing[i];
        delete outgoing;
    }

    vec3 migrateToCell(Particle p, int px, int py, int pz) {
        // currently this is assuming that particles are
        // migrating only to the immediate neighbors
        int x = thisIndex.x * CELL_SIZE_X + CELL_ORIGIN_X;
        int y = thisIndex.y * CELL_SIZE_Y + CELL_ORIGIN_Y;
        int z = thisIndex.z * CELL_SIZE_Z + CELL_ORIGIN_Z;
        px = py = pz = 0;

        if (p.pos.x < x) px = -1;
        else if (p.pos.x > x+CELL_SIZE_X) px = 1;

        if (p.pos.y < y) py = -1;
        else if (p.pos.y > y+CELL_SIZE_Y) py = 1;

        if (p.pos.z < z) pz = -1;
        else if (p.pos.z > z+CELL_SIZE_Z) pz = 1;
        return new vec3(px, py, pz);
    }

    void updateProperties(Array<vec3> forces, int lengthUp) {
        int i;
        double powTen, powFteen, realTimeDelta, invMassParticle;
        powTen = pow(10.0, -10);
        powFteen = pow(10.0, -15);
        realTimeDelta = DEFAULT_DELTA * powFteen;
        for(i = 0; i < particles.size(0); i++) {
            //calculate energy only in begining and end
            if(stepCount == 1) {
                energy[0] += (0.5 * particles[i].mass * particles[i].vel.dot(particles[i].vel));
            } else if(stepCount == finalStepCount) { 
                energy[1] += (0.5 * particles[i].mass * particles[i].vel.dot(particles[i].vel));
            }
            // applying kinetic equations
            invMassParticle = 1 / particles[i].mass;
            particles[i].acc = forces[i] * invMassParticle;
            particles[i].acc.copy(forces[i];
            particles[i].mult(invMassParticle);
            particles[i].vel.copy(particles[i].acc);
            particles[i].vel.mult(realTimeDelta);

            limitVelocity(particles[i]);

            vec3 tmp = new vec3();
            tmp.copy(particles[i].vel);
            tmp.mult(realTimeDelta);
            particles[i].pos.add(tmp);
            delete tmp;
        }
    }

    double velocityCheck(double inVelocity) {
        if (fabs(inVelocity) > MAX_VELOCITY) {
            if(inVelocity < 0.0 )
                return -MAX_VELOCITY;
            else
                return MAX_VELOCITY;
        } else {
            return inVelocity;
        }
    }

    void limitVelocity(Particle p) {
        p.vel.x = velocityCheck(p.vel.x);
        p.vel.y = velocityCheck(p.vel.y);
        p.vel.z = velocityCheck(p.vel.z);
    }

    Particle wrapAround(Particle p) {
        if(p.pos.x < CELL_ORIGIN_X) p.pos.x += CELL_SIZE_X*cellArrayDimX;
        if(p.pos.y < CELL_ORIGIN_Y) p.pos.y += CELL_SIZE_Y*cellArrayDimY;
        if(p.pos.z < CELL_ORIGIN_Z) p.pos.z += CELL_SIZE_Z*cellArrayDimZ;
        if(p.pos.x > CELL_ORIGIN_X + CELL_SIZE_X*cellArrayDimX) p.pos.x -= CELL_SIZE_X*cellArrayDimX;
        if(p.pos.y > CELL_ORIGIN_Y + CELL_SIZE_Y*cellArrayDimY) p.pos.y -= CELL_SIZE_Y*cellArrayDimY;
        if(p.pos.z > CELL_ORIGIN_Z + CELL_SIZE_Z*cellArrayDimZ) p.pos.z -= CELL_SIZE_Z*cellArrayDimZ;
        return p;
    }

    entry void run() {
        if(thisIndex.x==0 && thisIndex.y==0 && thisIndex.z==0) {
            stepTime = CkWallTimer();
        }

        for(int stepCount = 1; stepCount <= finalStepCount; stepCount++) {
            //send current atom positions to my computes 
            sendPositions();

            //update properties of atoms using new force values 
            when reduceForces(Array<vec3> forces) updateProperties(forces);

            if ((stepCount %  MIGRATE_STEPCOUNT) == 0) {
                //send atoms that have moved beyond my cell to neighbors
                migrateParticles();

                //receive particles from my neighbors
                for(int updateCount = 0; updateCount < inbrs; updateCount++) {
                    when receiveParticles(Array<Particle> updates) {
                        Array<Particle> tmp = new Array<Particle>([particles.size(0) + updates.size(0)]);
                        for(int i=0; i < particles.size(0); i++) {
                            tmp[i] = particles[i];
                        }
                        for(int i=0; i < updates.size(0); i++) {
                            tmp[i + particles.size(0)] = updates[i];
                        }
                        delete particles;
                        particles = tmp;
                    }
                }
            }
            if (thisIndex.x==0 && thisIndex.y==0 && thisIndex.z==0 && stepCount%20==0) {
                CkPrintf("Step %d Benchmark Time %lf ms/step\n", 
                        stepCount, ((CkWallTimer() - stepTime)/20)*1000);
                stepTime = CkWallTimer();
            }
            //periodically call load balancer
            //if (stepCount >= firstLdbStep && (stepCount - firstLdbStep) % ldbPeriod == 0) {
            //    AtSync();
            //    when ResumeFromSync() stepTime = CkWallTimer();
            //}
        }
        //everything done, reduction on kinetic energy
        contribute(energy,
                CkReduction.sum_double,
                CkCallback(Main.energySumK, mainProxy));
    } 
*/
}

chare_array [6d] Compute {
    Array<double> energy;
    int stepCount, cellCount;
    // FIXME: CkSectionInfo mcast1, mcast2;
    //CkSectionInfo mcast1;
    //CkSectionInfo mcast2;
    ParticleDataMsg bufferedMsg;

    entry Compute() {
        cellCount = 0;
        stepCount = 0;
        energy = new Array<double>([2]);
        // FIXME: energy[0] = energy[1] = 0;
        energy[0] = 0;
        energy[1] = 0;
    }

/*
    void interact(ParticleDataMsg msg) {
        double energyP = 0;

        //self interaction check
        if (thisIndex.x1 ==thisIndex.x2 && thisIndex.y1 ==thisIndex.y2 && thisIndex.z1 ==thisIndex.z2) {
            CkGetSectionInfo(mcast1,msg);
            energyP = calcInternalForces(msg, mcast1, stepCount);
        } else {
            //check if this is the first message or second
            if (cellCount == 0) {
                bufferedMsg = msg;
                cellCount++;
                return;
            }
            // Both particle sets have been received, so compute interaction
            cellCount = 0;
            ParticleDataMsg msgA = msg;
            ParticleDataMsg msgB = bufferedMsg;
            CkSectionInfo handleA = mcast1;
            CkSectionInfo handleB = mcast2;
            if (bufferedMsg.x*cellArrayDimY*cellArrayDimZ + bufferedMsg.y*cellArrayDimZ + bufferedMsg.z < msg.x*cellArrayDimY*cellArrayDimZ + msg.y*cellArrayDimZ + msg.z){ 
                CkSectionInfo tmp = handleA;
                handleA = handleB;
                handleB = tmp;
            }
            if (bufferedMsg.lengthAll <= msg.lengthAll) {
                {
                    ParticleDataMsg tmp = msgA;
                    msgA = msgB;
                    msgB = tmp;
                }
                CkSectionInfo tmp = handleA;
                handleA = handleB;
                handleB = tmp;
            }

            energyP = calcPairForces(msgA, msgB, handleA, handleB, stepCount);
            bufferedMsg = null;
        }
        //energy assignment only in begining and end
        if(stepCount == 1) {
            energy[0] = energyP;
        } else if(stepCount == finalStepCount) {
            energy[1] = energyP;
        }
    }

    entry void run() {
        for (stepCount = 1; stepCount <= finalStepCount; stepCount++) {
            //self interaction check
            if (thisIndex.x1==thisIndex.x2 && thisIndex.y1==thisIndex.y2 && thisIndex.z1==thisIndex.z2) {
                when calculateForces(ParticleDataMsg msg) interact(msg);
            } else {
                //receive positions from two cells
                when calculateForces(ParticleDataMsg msg) interact(msg);
                when calculateForces(ParticleDataMsg msg) interact(msg);
            }
            //periodically call load balancer
            //if (stepCount >= firstLdbStep && (stepCount - firstLdbStep) % ldbPeriod == 0) {
            //    AtSync();
            //    when ResumeFromSync() { }
            //}
        }
        //everything done, reduction on potential energy
        contribute(energy,
            CkReduction.sum_double,
            CkCallback(Main.energySumP, mainProxy));
    }

    double calcInternalForces(ParticleDataMsg first, CkSectionInfo mcast1, int stepCount) {
        int i, j, ptpCutOffSqd;
        int firstLen = first.lengthAll;
        double powTwenty, powTen, firstx, firsty, firstz, rx, ry, rz, r, rsqd, fx, fy, fz, f, fr;
        vec3 separation = new vec3();
        vec3 force = new vec3();
        double rSix, rTwelve;
        double energy = 0;
        int doEnergy = 0;
        if(stepCount == 1 || stepCount == finalStepCount)
            doEnergy = 1;
        Array<vec3> firstmsg = new Array<vec3>([firstLen]);

        ptpCutOffSqd = PTP_CUT_OFF * PTP_CUT_OFF;
        powTen = pow(10.0, -10);
        powTwenty = pow(10.0, -20);
        for(i = 0; i < firstLen; i++){
            for(j = i+1; j < firstLen; j++) {
                // computing base values
                separation.copy(first.part[i]);
                separation.sub(first.part[j];
                rsqd = separation.dot(separation);
                if(rsqd >= 0.001 && rsqd < ptpCutOffSqd) {
                    rsqd = rsqd * powTwenty;
                    r = sqrt(rsqd);
                    rSix = ((double)rsqd) * rsqd * rsqd;
                    rTwelve = rSix * rSix;
                    f = (double)(VDW_A / rTwelve - VDW_B / rSix);
                    if(doEnergy)
                        energy += (double)( VDW_A / (12*rTwelve) - VDW_B / (6*rSix));

                    fr = f / r;
                    force.copy(separation);
                    force.mult(fr * powTen);
                    firstmsg[j].add(force);
                    firstmsg[i].sub(force);
                }
            }
        }
        CkMulticastMgr mCastGrp = CkMulticastMgr@(mCastGrpID).ckLocalBranch();
        mCastGrp.contribute(firstmsg, CkReduction.sum_double, mcast1);
        delete firstmsg;
        delete first;
        delete separation;
        delete force;
        return energy;
    }

    double calcPairForces(ParticleDataMsg first, ParticleDataMsg second,
            CkSectionInfo mcast1, CkSectionInfo mcast2, int stepCount) {
        int i, j, jpart, ptpCutOffSqd, diff;
        int firstLen = first.lengthAll;
        int secondLen = second.lengthAll;
        double powTwenty, powTen, r, rsqd, f, fr;
        vec3 separation = new vec3();
        vec3 force = new vec3();
        double rSix, rTwelve;
        double energy = 0;
        int doEnergy = 0;
        if(stepCount == 1 || stepCount == finalStepCount)
            doEnergy = 1;

        Array<vec3> firstmsg = new Array<vec3>([firstLen]);
        Array<vec3> secondmsg = new Array<vec3>([secondLen]);
        //check for wrap around and adjust locations accordingly
        if (abs(first.x - second.x) > 1){
            diff = CELL_SIZE_X * cellArrayDimX;
            if (second.x < first.x)
                diff = -1 * diff; 
            for (i = 0; i < firstLen; i++)
                first.part[i].x += diff;
        }
        if (abs(first.y - second.y) > 1){
            diff = CELL_SIZE_Y * cellArrayDimY;
            if (second.y < first.y)
                diff = -1 * diff; 
            for (i = 0; i < firstLen; i++)
                first.part[i].y += diff;
        }
        if (abs(first.z - second.z) > 1){
            diff = CELL_SIZE_Z * cellArrayDimZ;
            if (second.z < first.z)
                diff = -1 * diff; 
            for (i = 0; i < firstLen; i++)
                first.part[i].z += diff;
        } 
        ptpCutOffSqd = PTP_CUT_OFF * PTP_CUT_OFF;
        powTen = pow(10.0, -10);
        powTwenty = pow(10.0, -20);

        int i1, j1;
        for(i1 = 0; i1 < firstLen; i1=i1+BLOCK_SIZE)
            for(j1 = 0; j1 < secondLen; j1=j1+BLOCK_SIZE)
                for(i = i1; i < i1+BLOCK_SIZE && i < firstLen; i++) {
                    for(jpart = j1; jpart < j1+BLOCK_SIZE && jpart < secondLen; jpart++) {
                        separation.copy(first.part[i]);
                        separation.sub(second.part[j]);
                        rsqd = separation.dot(separation);
                        if (rsqd >= 0.001 && rsqd < ptpCutOffSqd) {
                            rsqd = rsqd * powTwenty;
                            r = sqrt(rsqd);
                            rSix = ((double)rsqd) * rsqd * rsqd;
                            rTwelve = rSix * rSix;
                            f = (double)(VDW_A / rTwelve - VDW_B / rSix);
                            if(doEnergy)
                                energy += (double)( VDW_A / (12*rTwelve) - VDW_B / (6*rSix));
                            fr = f / r;
                            force.copy(separation);
                            force.mult(fr * powTen);
                            firstmsg[i].add(force);
                            secondmsg[jpart].sub(force);
                        }
                    }
                }

        CkMulticastMgr mCastGrp = CkMulticastMgr@(mCastGrpID).ckLocalBranch();
        CkGetSectionInfo(mcast1, first);
        mCastGrp.contribute(firstmsg, CkReduction.sum_double, mcast1);
        CkGetSectionInfo(mcast2, second);
        mCastGrp.contribute(secondmsg, CkReduction.sum_double, mcast2);

        delete firstmsg;
        delete secondmsg;
        delete first;
        delete second;
        delete force;
        delete separation;
        return energy;
    }
*/
}
