#ifndef __PHYSICS_H__
#define __PHYSICS_H__


extern /* readonly */ CkGroupID mCastGrpID;

extern /* readonly */ int cellArrayDimX;	// Number of Chare Rows
extern /* readonly */ int cellArrayDimY;	// Number of Chare Columns
extern /* readonly */ int cellArrayDimZ;
extern /* readonly */ int finalStepCount; 

#define BLOCK_SIZE	512

//function to calculate forces among 2 lists of atoms
inline void calcPairForces(ParticleDataMsg* first, ParticleDataMsg* second, int stepCount) {
  int i, j, jpart, ptpCutOffSqd, diff;
  int firstLen = first->lengthAll;
  int secondLen = second->lengthAll;
  double powTwenty, powTen, r, rsqd, f, fr;
  vec3 separation, force;
  double rSix, rTwelve;

  vec3 *firstmsg = new vec3[firstLen];
  vec3 *secondmsg = new vec3[secondLen];
  //check for wrap around and adjust locations accordingly
  if (abs(first->x - second->x) > 1){
    diff = CELL_SIZE_X * cellArrayDimX;
    if (second->x < first->x)
      diff = -1 * diff; 
    for (i = 0; i < firstLen; i++)
      first->part[i].x += diff;
  }
  if (abs(first->y - second->y) > 1){
    diff = CELL_SIZE_Y * cellArrayDimY;
    if (second->y < first->y)
      diff = -1 * diff; 
    for (i = 0; i < firstLen; i++)
      first->part[i].y += diff;
  }
  if (abs(first->z - second->z) > 1){
    diff = CELL_SIZE_Z * cellArrayDimZ;
    if (second->z < first->z)
      diff = -1 * diff; 
    for (i = 0; i < firstLen; i++)
      first->part[i].z += diff;
  } 
  ptpCutOffSqd = PTP_CUT_OFF * PTP_CUT_OFF;
  powTen = pow(10.0, -10);
  powTwenty = pow(10.0, -20);

  int i1, j1;
  for(i1 = 0; i1 < firstLen; i1=i1+BLOCK_SIZE)
    for(j1 = 0; j1 < secondLen; j1=j1+BLOCK_SIZE)
      for(i = i1; i < i1+BLOCK_SIZE && i < firstLen; i++) {
        for(jpart = j1; jpart < j1+BLOCK_SIZE && jpart < secondLen; jpart++) {
          separation = first->part[i] - second->part[jpart];
          rsqd = dot(separation, separation);
          if (rsqd >= 0.001 && rsqd < ptpCutOffSqd) {
            rsqd = rsqd * powTwenty;
            r = sqrt(rsqd);
            rSix = ((double)rsqd) * rsqd * rsqd;
            rTwelve = rSix * rSix;
            f = (double)(VDW_A / rTwelve - VDW_B / rSix);
            fr = f /r;
            force = separation * (fr * powTen);
            firstmsg[i] += force;
            secondmsg[jpart] -= force;
          }
        }
      }

  cellArray(first->x, first->y, first->z).receiveForces(stepCount,firstmsg,firstLen);
  cellArray(second->x, second->y, second->z).receiveForces(stepCount,secondmsg,secondLen);

  delete [] firstmsg;
  delete [] secondmsg;
  delete first;
  delete second;
}

//function to calculate forces among atoms in a single list
inline void calcInternalForces(ParticleDataMsg* first, int stepCount) {
  int i, j, ptpCutOffSqd;
  int firstLen = first->lengthAll;
  double powTwenty, powTen, firstx, firsty, firstz, rx, ry, rz, r, rsqd, fx, fy, fz, f, fr;
  vec3 firstpos, separation, force;
  double rSix, rTwelve;
  vec3 *firstmsg = new vec3[firstLen];

  ptpCutOffSqd = PTP_CUT_OFF * PTP_CUT_OFF;
  powTen = pow(10.0, -10);
  powTwenty = pow(10.0, -20);
  for(i = 0; i < firstLen; i++){
    firstpos = first->part[i];
    for(j = i+1; j < firstLen; j++) {
      // computing base values
      separation = firstpos - first->part[j];
      rsqd = dot(separation, separation);
      if(rsqd >= 0.001 && rsqd < ptpCutOffSqd){
        rsqd = rsqd * powTwenty;
        r = sqrt(rsqd);
        rSix = ((double)rsqd) * rsqd * rsqd;
        rTwelve = rSix * rSix;
        f = (double)(VDW_A / rTwelve - VDW_B / rSix);
        fr = f /r;
        force = separation * (fr * powTen);
        firstmsg[j] += force;
        firstmsg[i] -= force;
      }
    }
  }
  cellArray(first->x, first->y, first->z).receiveForces(stepCount,firstmsg,firstLen);
  delete [] firstmsg;
  delete first;
}
#endif
