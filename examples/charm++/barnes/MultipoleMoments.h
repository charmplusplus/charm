#ifndef MULTIPOLEMOMENTS_H
#define MULTIPOLEMOMENTS_H

#include "Vector3D.h"
#include "defines.h"

class FullTreeNode;
class Particle;

/// A representation of a multipole expansion.
class MultipoleMoments {
public:
	/// A physical size for this multipole expansion, calculated by an external function using some other information
	Real rsq;
	/// The total mass represented by this expansion
	Real totalMass;
	/// The center of mass (zeroth order multipole)
	Vector3D<Real> cm;

	/// The reduced quadrupole about the centre of mass,
	///   Q_ij = sum_k m_k (3 s_i s_j - delta_ij |s|^2),   s = x_k - cm
	/// which is traceless by construction, so qzz is not stored -- see
	/// qzz() below. Five floats per node buys an opening angle of roughly
	/// 0.7 to 0.8 at the force error a monopole expansion reaches at 0.5,
	/// which is the largest accuracy-per-flop lever this code has.
	///
	/// Building with -DMONOPOLE_ONLY drops the term from the force
	/// evaluation (the moments are still carried, they just are not used),
	/// which is how the two are compared.
	Real qxx, qxy, qxz, qyy, qyz;

        MultipoleMoments() {
          clear();
        }
	
	/// Reset this expansion to nothing
        void clear() {
          rsq = 0;
          totalMass = 0;
          cm.x = cm.y = cm.z = 0;
          qxx = qxy = qxz = qyy = qyz = 0;
        }

	/// The trace vanishes, so the last diagonal component is implied.
	inline Real qzz() const { return -(qxx + qyy); }
};

#include "pup.h"
inline void operator|(PUP::er& p, MultipoleMoments& m) {
	p | m.rsq;
	p | m.totalMass;
	p | m.cm;
	p | m.qxx;
	p | m.qxy;
	p | m.qxz;
	p | m.qyy;
	p | m.qyz;
}
#endif //MULTIPOLEMOMENTS_H
