#ifndef __GRAVITY_H__
#define __GRAVITY_H__

#include "Node.h"
#include "Vector3D.h"
#include "Space.h"
#include "defines.h"
#include "Parameters.h"

extern Parameters globalParams;

// The opening criterion. Both forms accept a cell when its opening radius
// subtends less than theta from the target; they differ in where the distance
// is measured from.
//
// The stock form measures centre of mass to centre of mass. The bucket's
// centre of mass can sit well behind its near face, so the test overstates the
// separation and accepts cells that particles at the near edge of the bucket
// are well inside the opening radius of. The error that admits grows with
// bucket size, which matters here because the GPU path wants -b=128 where the
// CPU default was 10.
//
// The default form measures from the cell's centre of mass to the nearest
// point of the bucket's bounding box, which is a lower bound on the distance
// to any particle in the bucket. It is strictly more conservative, so at a
// fixed theta it opens more cells and costs more -- the point is that it moves
// the error/cost curve, and theta is then the knob. Build with
// -DOPEN_CRITERION_CM to get the old test back for comparison.
inline bool
openCriterionBucket(Node<ForceData> *node,
                   Node<ForceData> *bucketNode) {
#ifdef OPEN_CRITERION_CM
  Vector3D<Real> dr = node->data.moments.cm - bucketNode->data.moments.cm;
  Real drsq = dr.lengthSquared();
#else
  const OrientedBox<Real> &box = bucketNode->data.box;
  const Vector3D<Real> &cm = node->data.moments.cm;

  // Per axis: zero if the centre of mass lies within the bucket's extent,
  // otherwise the gap to the nearer face.
  Real dx = box.lesser_corner.x - cm.x;
  Real ex = cm.x - box.greater_corner.x;
  if(dx < ex) dx = ex;
  if(dx < 0) dx = 0;

  Real dy = box.lesser_corner.y - cm.y;
  Real ey = cm.y - box.greater_corner.y;
  if(dy < ey) dy = ey;
  if(dy < 0) dy = 0;

  Real dz = box.lesser_corner.z - cm.z;
  Real ez = cm.z - box.greater_corner.z;
  if(dz < ez) dz = ez;
  if(dz < 0) dz = 0;

  Real drsq = dx*dx + dy*dy + dz*dz;
#endif
  return (globalParams.tolsq*drsq < node->data.moments.rsq);
}

inline 
void grav(Particle *pstart, Particle *pend, Real mass, const Vector3D<Real> &position){
  Vector3D<Real> dr;
  Real drsq;
  Real drabs;
  Real phii;
  Real mor3;

  for(Particle *p = pstart; p != pend; p++){
    dr = position - p->position;
    drsq = dr.lengthSquared();
    drsq += globalParams.epssq;
    drabs = sqrt((double) drsq);
    phii = mass/drabs;
    p->potential -= phii;
    mor3 = phii/drsq;
    p->acceleration += mor3*dr;
  }
}

// A cell acts through its monopole and its reduced quadrupole. With
//   dr  = cm - x,      r^2 = |dr|^2 + eps^2
//   Q dr           the quadrupole contracted once with dr
//   dr.Q.dr        contracted twice
// the potential is  -(M/r + (1/2) dr.Q.dr / r^5)  and the acceleration is
//   M dr/r^3 - (Q dr)/r^5 + (5/2)(dr.Q.dr) dr/r^7.
// The monopole terms are exactly what grav() computes, so -DMONOPOLE_ONLY
// falls back to it and the two can be compared.
inline
void gravQuad(Particle *pstart, Particle *pend, const MultipoleMoments &m){
  const Real qzz = m.qzz();

  for(Particle *p = pstart; p != pend; p++){
    Vector3D<Real> dr = m.cm - p->position;
    Real drsq = dr.lengthSquared() + globalParams.epssq;

    Real rinv  = 1.0/sqrt((double) drsq);
    Real rinv2 = rinv*rinv;
    Real rinv3 = rinv*rinv2;
    Real rinv5 = rinv3*rinv2;
    Real rinv7 = rinv5*rinv2;

    Real qx = m.qxx*dr.x + m.qxy*dr.y + m.qxz*dr.z;
    Real qy = m.qxy*dr.x + m.qyy*dr.y + m.qyz*dr.z;
    Real qz = m.qxz*dr.x + m.qyz*dr.y + qzz   *dr.z;
    Real drQdr = dr.x*qx + dr.y*qy + dr.z*qz;

    p->potential -= (m.totalMass*rinv + 0.5*drQdr*rinv5);

    Real coef = m.totalMass*rinv3 + 2.5*drQdr*rinv7;
    p->acceleration += coef*dr;
    p->acceleration -= rinv5*Vector3D<Real>(qx,qy,qz);
  }
}

inline
int nodeBucketForce(Node<ForceData> *node, 
		    Node<ForceData> *req){
  
  Particle *particles = req->getParticles();
  int numParticles = req->getNumParticles();
#ifdef MONOPOLE_ONLY
  grav(particles,particles+numParticles,node->data.moments.totalMass,node->data.moments.cm);
#else
  gravQuad(particles,particles+numParticles,node->data.moments);
#endif
  return req->getNumParticles();
}

inline int partBucketForce(ExternalParticle *part, 
			   Node<ForceData> *req){ 

  Particle *particles = req->getParticles();
  int numParticles = req->getNumParticles();
  grav(particles,particles+numParticles,part->mass,part->position);
  return numParticles;
}

#endif
