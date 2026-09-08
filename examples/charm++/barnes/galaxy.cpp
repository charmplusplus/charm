/*
 * galaxy.cpp -- initial conditions for a collision between two disk galaxies.
 *
 * The datasets that existed here were a uniform cube (gen) and a Plummer
 * sphere with a copy of itself displaced by 4 (plummer, which is what
 * clustered.bin holds). The copy shares its original's velocities exactly, so
 * the two halves fall together with no relative orbit, no rotation and
 * perfectly correlated structure. It clusters, but it does not evolve into
 * anything, and the two halves are not independent samples.
 *
 * This builds the standard thing instead: two compound galaxies, each a
 * rotating exponential disk inside a Hernquist bulge and a Hernquist halo,
 * placed on a Kepler encounter orbit and started a given time before
 * pericentre. The geometry defaults to the Antennae-like one (both disks
 * inclined 60 degrees to the orbital plane, Toomre & Toomre 1972; Barnes
 * 1988), which is the case that throws long tidal tails.
 *
 * Why this shape of dataset is worth having here: the work in a Barnes-Hut
 * step follows the density, and in a collision the density is both very
 * non-uniform AND non-stationary. The centres are cuspy, the tails are thin
 * and grow, and the whole configuration translates as the galaxies swing
 * through pericentre. A partition that balances iteration 10 is wrong by
 * iteration 30. A uniform cube exercises none of that, and two Plummer
 * spheres at rest exercise only the first half of it.
 *
 * Everything is in N-body units: G = 1, and the two galaxies together carry
 * unit mass by default. All particles have the same mass, so the particle
 * count of a component is proportional to its mass.
 *
 * Equilibrium:
 *   spheroids  isotropic, from the spherical Jeans equation integrated on a
 *              log grid against the TOTAL enclosed mass (so the bulge knows
 *              about the halo and the disk), speeds capped at the local
 *              escape speed;
 *   disk       circular speed from the spheroids' enclosed mass plus the exact
 *              razor-thin exponential-disk term (Freeman 1970), radial
 *              dispersion set by a Toomre Q, azimuthal dispersion from the
 *              epicyclic ratio, mean azimuthal speed from the asymmetric-drift
 *              solution of the radial Jeans equation, vertical dispersion from
 *              the isothermal sheet.
 * The generator prints 2T/|W| per galaxy so that this is checkable rather than
 * asserted; both come out within about a per cent of 1.
 *
 *   ./galaxy <nbody> <outfile> [key=value ...]
 *
 * See the parameter table printed by ./galaxy with no arguments.
 */

#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;

namespace {

// ------------------------------------------------------------------ options

map<string, string> opts;

double dparam(const char* k, double def)
{
  map<string, string>::iterator i = opts.find(k);
  return i == opts.end() ? def : atof(i->second.c_str());
}
int iparam(const char* k, int def)
{
  map<string, string>::iterator i = opts.find(k);
  return i == opts.end() ? def : atoi(i->second.c_str());
}
string sparam(const char* k, const char* def)
{
  map<string, string>::iterator i = opts.find(k);
  return i == opts.end() ? string(def) : i->second;
}

// ------------------------------------------------------------------ vectors

struct Vec
{
  double x, y, z;
};
inline Vec vec(double x, double y, double z) { Vec v = {x, y, z}; return v; }
inline Vec operator+(Vec a, Vec b) { return vec(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec operator-(Vec a, Vec b) { return vec(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec operator*(double s, Vec a) { return vec(s * a.x, s * a.y, s * a.z); }
inline double len(Vec a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

// R_z(node) R_x(inc): tips the disk plane away from the orbital plane and then
// turns the line of nodes. inc = 0 is a direct (prograde) passage, inc = 180 a
// retrograde one, and prograde disks are the ones that grow tails.
struct Rotation
{
  double c_i, s_i, c_n, s_n;
  Rotation(double incDeg, double nodeDeg)
  {
    const double i = incDeg * M_PI / 180.0, n = nodeDeg * M_PI / 180.0;
    c_i = cos(i); s_i = sin(i); c_n = cos(n); s_n = sin(n);
  }
  Vec operator()(Vec v) const
  {
    const double y = c_i * v.y - s_i * v.z;
    const double z = s_i * v.y + c_i * v.z;
    return vec(c_n * v.x - s_n * y, s_n * v.x + c_n * y, z);
  }
};

// ------------------------------------------------------------------- galaxy

struct Particle
{
  Vec pos, vel;
};

// Hernquist (1990): rho = M a / (2 pi r (r+a)^3), M(<r) = M r^2/(r+a)^2.
// Cuspy, so the centres are genuinely deep -- which is the point: a cusp is
// what makes one tree piece cost many times another.
inline double hernquistRho(double M, double a, double r)
{
  return M * a / (2.0 * M_PI * r * (r + a) * (r + a) * (r + a));
}
inline double hernquistM(double M, double a, double r)
{
  return M * r * r / ((r + a) * (r + a));
}

struct Galaxy
{
  // Masses. md/mb/mh is what a component's particles actually carry; the
  // *Inf values normalise the analytic profile it was cut out of. They differ
  // because every component is truncated -- a Hernquist halo has no finite
  // extent and an untruncated one would put particles a thousand scale
  // lengths out, which is both unphysical and ruinous for a tree whose key
  // resolution is spread over the bounding box. Sampling a truncated profile
  // but normalising it as if it were complete is the mistake that matters
  // here: it packs the whole component mass inside the cut, making the model
  // denser than the dispersions were computed for. At the default 15 rd halo
  // cut that is a 44% error in density and the galaxy collapses (2T/|W| came
  // out 0.72 before this was separated out).
  double m, md, mb, mh;
  double mdInf, mbInf, mhInf;
  // disk
  double rd, z0, rdcut;
  double toomreQ;
  // spheroids
  double ab, abcut, ah, ahcut;
  // counts
  int nd, nb, nh;
  // placement
  Vec ctr, vctr;
  Rotation rot;

  Galaxy() : rot(0.0, 0.0) {}

  // ---- mass model ------------------------------------------------------
  // The disk enters the spherically averaged mass profile through its
  // enclosed mass within cylindrical radius R. That is the usual
  // approximation for the spheroids' Jeans equation; the disk's own rotation
  // curve below does not use it.
  double diskMenc(double R) const
  {
    const double x = min(R, rdcut) / rd;
    return mdInf * (1.0 - (1.0 + x) * exp(-x));
  }
  // Enclosed mass of the model as sampled: each component's own profile up to
  // its cut, flat beyond it. Flat is the whole point -- outside the halo cut
  // there is no more mass, so an orbit out there sees only what is inside.
  double massEnclosed(double r) const
  {
    return hernquistM(mbInf, ab, min(r, abcut)) +
           hernquistM(mhInf, ah, min(r, ahcut)) + diskMenc(r);
  }
  // Potential of the same. A Hernquist sphere truncated at rcut has the
  // potential of the complete one shifted by the shell that is no longer
  // there, int_rcut^inf dM/r' = Minf a/(rcut+a)^2. Only the escape-speed cap
  // uses this, and only inside the cut, so the exterior form is not needed;
  // the disk is a Hernquist-like term of scale rd, which is good to a few per
  // cent and does not have to be better.
  double phi(double r) const
  {
    double f = 0.0;
    f -= mbInf / (r + ab) - mbInf * ab / ((abcut + ab) * (abcut + ab));
    f -= mhInf / (r + ah) - mhInf * ah / ((ahcut + ah) * (ahcut + ah));
    f -= mdInf / (r + rd);
    return f;
  }
  double sigma(double R) const   // disk surface density
  {
    return mdInf / (2.0 * M_PI * rd * rd) * exp(-R / rd);
  }

  // Circular speed squared in the plane: spheroids through their enclosed
  // mass, disk through the exact razor-thin exponential term (Freeman 1970),
  //   v^2 = 4 pi G Sigma0 Rd y^2 [I0(y)K0(y) - I1(y)K1(y)],  y = R/2Rd.
  double vc2(double R) const
  {
    if (R < 1e-6 * rd) R = 1e-6 * rd;
    double v2 = (hernquistM(mbInf, ab, min(R, abcut)) +
                 hernquistM(mhInf, ah, min(R, ahcut))) / R;
    const double y = R / (2.0 * rd);
    const double s0 = mdInf / (2.0 * M_PI * rd * rd);
    const double bess = std::cyl_bessel_i(0, y) * std::cyl_bessel_k(0, y) -
                        std::cyl_bessel_i(1, y) * std::cyl_bessel_k(1, y);
    v2 += 4.0 * M_PI * s0 * rd * y * y * bess;
    return v2 > 0.0 ? v2 : 0.0;
  }
  double vc(double R) const { return sqrt(vc2(R)); }

  // Epicyclic frequency, kappa^2 = (2 Omega / R) d(R^2 Omega)/dR, differenced.
  double kappa2(double R) const
  {
    const double h = 1e-3 * rd;
    const double om = vc(R) / R;
    const double omp = vc(R + h) / (R + h), omm = vc(R - h) / (R - h);
    const double d = ((R + h) * (R + h) * omp - (R - h) * (R - h) * omm) / (2.0 * h);
    return 2.0 * om / R * d;
  }
  // Toomre Q = sigma_R kappa / (3.36 G Sigma), inverted for sigma_R. A floor
  // keeps the outer disk from being exactly cold, which would make the
  // asymmetric-drift term below meaningless there.
  double sigmaR(double R) const
  {
    const double k = sqrt(max(kappa2(R), 1e-12));
    const double s = toomreQ * 3.36 * sigma(R) / k;
    const double floorv = 0.02 * vc(rd);
    return s > floorv ? s : floorv;
  }

  // ---- spheroid velocity dispersions -----------------------------------
  // Isotropic spherical Jeans, integrated inward on a log grid:
  //   rho_c(r) sigma_r^2(r) = \int_r^inf rho_c(s) G M_tot(<s) / s^2 ds
  // One table per spheroid component: they share M_tot but have their own
  // rho, so they do not share a dispersion. The integration runs far past the
  // truncation radius with the untruncated profile, which is what supplies the
  // outer pressure boundary; truncating the integral instead leaves the edge
  // of the halo unsupported and it falls in.
  vector<double> jlr, jsb, jsh;
  void buildJeans()
  {
    const int N = 4096;
    const double lo = log(1e-5 * rd), hi = log(1e4 * rd);
    jlr.resize(N); jsb.resize(N); jsh.resize(N);
    vector<double> r(N), fb(N), fh(N);
    for (int i = 0; i < N; i++)
    {
      jlr[i] = lo + (hi - lo) * i / (N - 1);
      r[i] = exp(jlr[i]);
      const double M = massEnclosed(r[i]);
      fb[i] = hernquistRho(mbInf, ab, r[i]) * M / (r[i] * r[i]);
      fh[i] = hernquistRho(mhInf, ah, r[i]) * M / (r[i] * r[i]);
    }
    double Ib = 0.0, Ih = 0.0;
    jsb[N - 1] = jsh[N - 1] = 0.0;
    for (int i = N - 2; i >= 0; i--)
    {
      const double dr = r[i + 1] - r[i];
      Ib += 0.5 * (fb[i] + fb[i + 1]) * dr;
      Ih += 0.5 * (fh[i] + fh[i + 1]) * dr;
      const double rb = hernquistRho(mbInf, ab, r[i]), rh = hernquistRho(mhInf, ah, r[i]);
      jsb[i] = (mb > 0.0 && rb > 0.0) ? Ib / rb : 0.0;
      jsh[i] = (mh > 0.0 && rh > 0.0) ? Ih / rh : 0.0;
    }
  }
  double lookup(const vector<double>& t, double r) const
  {
    const double lr = log(r);
    if (lr <= jlr.front()) return t.front();
    if (lr >= jlr.back()) return t.back();
    const double f = (lr - jlr.front()) / (jlr.back() - jlr.front()) * (jlr.size() - 1);
    const int i = (int)f;
    const double w = f - i;
    return t[i] * (1.0 - w) + t[i + 1] * w;
  }
};

// ------------------------------------------------------------------ sampling

typedef mt19937_64 Rng;

inline double u01(Rng& g)
{
  return uniform_real_distribution<double>(0.0, 1.0)(g);
}
inline Vec randomDirection(Rng& g)
{
  const double c = 2.0 * u01(g) - 1.0, s = sqrt(max(0.0, 1.0 - c * c));
  const double p = 2.0 * M_PI * u01(g);
  return vec(s * cos(p), s * sin(p), c);
}

// Hernquist radius: M(<r)/M = (r/(r+a))^2 inverts in closed form, and drawing
// the mass fraction only up to the truncation radius truncates the profile
// without changing its shape.
double hernquistRadius(double a, double rcut, Rng& g)
{
  const double umax = (rcut / (rcut + a)) * (rcut / (rcut + a));
  const double s = sqrt(u01(g) * umax);
  return a * s / (1.0 - s);
}

void sampleSpheroid(const Galaxy& G, const vector<double>& sig2, double a,
                    double rcut, int n, Rng& g, vector<Particle>& out)
{
  for (int i = 0; i < n; i++)
  {
    const double r = hernquistRadius(a, rcut, g);
    const double s = sqrt(max(G.lookup(sig2, r), 0.0));
    const double vesc = sqrt(max(-2.0 * G.phi(r), 0.0));
    normal_distribution<double> nd(0.0, s);
    Vec v;
    // Redraw anything that would leave: the Jeans solution is a second moment
    // and says nothing about the shape of the tail, and an unbound particle
    // here becomes a straggler that stretches the tree's bounding box for the
    // whole run. Deep in a cusp the dispersion can approach the escape speed,
    // so give up after a while and scale the draw instead of spinning.
    int tries = 0;
    do
    {
      v = vec(nd(g), nd(g), nd(g));
      if (++tries > 100) { const double l = len(v); if (l > 0.0) v = (0.9 * vesc / l) * v; break; }
    } while (len(v) > vesc);
    Particle p;
    p.pos = r * randomDirection(g);
    p.vel = v;
    out.push_back(p);
  }
}

// Exponential surface density: the enclosed-mass fraction 1-(1+x)e^-x has no
// closed-form inverse, so bisect it. Truncating renormalises the same way the
// spheroids do.
double diskRadius(double rd, double rcut, Rng& g)
{
  const double xmax = rcut / rd;
  const double fmax = 1.0 - (1.0 + xmax) * exp(-xmax);
  const double t = u01(g) * fmax;
  double lo = 0.0, hi = xmax;
  for (int i = 0; i < 60; i++)
  {
    const double mid = 0.5 * (lo + hi);
    if (1.0 - (1.0 + mid) * exp(-mid) < t) lo = mid; else hi = mid;
  }
  return rd * 0.5 * (lo + hi);
}

void sampleDisk(const Galaxy& G, int n, Rng& g, vector<Particle>& out)
{
  for (int i = 0; i < n; i++)
  {
    const double R = diskRadius(G.rd, G.rdcut, g);
    const double phi = 2.0 * M_PI * u01(g);

    // Isothermal sheet, rho ~ sech^2(z/z0): the CDF is (1+tanh(z/z0))/2, so z
    // inverts directly. Clipped at 5 scale heights.
    double z;
    do { z = G.z0 * atanh(2.0 * u01(g) - 1.0); } while (fabs(z) > 5.0 * G.z0);

    const double vcirc2 = G.vc2(R);
    const double sR = G.sigmaR(R);
    const double k2 = max(G.kappa2(R), 1e-12);
    const double om2 = vcirc2 / (R * R);
    const double ratio = k2 / (4.0 * om2);            // sigma_phi^2 / sigma_R^2
    const double sPhi = sR * sqrt(max(ratio, 1e-6));
    const double sZ = sqrt(max(M_PI * G.sigma(R) * G.z0, 1e-12));

    // Asymmetric drift, from the radial Jeans equation with the sigma_Rz term
    // dropped:
    //   vbar_phi^2 = v_c^2 + sigma_R^2 [1 - kappa^2/4Omega^2 + dln(Sigma
    //   sigma_R^2)/dlnR]
    // The logarithmic derivative is differenced from the actual profiles
    // rather than assumed exponential, because sigma_R has a floor.
    const double h = 1e-3 * G.rd;
    const double Rp = R + h, Rm = max(R - h, 1e-6 * G.rd);
    const double lp = log(G.sigma(Rp) * G.sigmaR(Rp) * G.sigmaR(Rp));
    const double lm = log(G.sigma(Rm) * G.sigmaR(Rm) * G.sigmaR(Rm));
    const double dln = (lp - lm) / (log(Rp) - log(Rm));
    const double vbar2 = vcirc2 + sR * sR * (1.0 - ratio + dln);
    const double vbar = vbar2 > 0.0 ? sqrt(vbar2) : 0.0;

    normal_distribution<double> nR(0.0, sR), nP(vbar, sPhi), nZ(0.0, sZ);
    const double vR = nR(g), vP = nP(g), vZ = nZ(g);

    Particle p;
    p.pos = vec(R * cos(phi), R * sin(phi), z);
    p.vel = vec(vR * cos(phi) - vP * sin(phi), vR * sin(phi) + vP * cos(phi), vZ);
    out.push_back(p);
  }
}

// -------------------------------------------------------------------- orbit

Vec twoBodyAccel(Vec r, double M)
{
  const double s = len(r);
  const double f = -M / (s * s * s);
  return f * r;
}

// Start at pericentre and integrate the relative orbit BACKWARDS. Doing it
// this way means the encounter happens where the caller wants it -- at a named
// time, or from a named separation -- instead of at whatever time an initial
// separation happens to imply. That matters because a run here is tens to
// hundreds of iterations and the pericentre passage is the part worth
// capturing.
//
// Stops at tmax if rstart <= 0, otherwise when the separation first reaches
// rstart. A bound orbit whose apocentre is inside rstart never gets there, so
// the turning point stops it too and the caller is told what it actually got.
void orbitBack(double M, double rp, double ecc, double tmax, double rstart,
               Vec& r, Vec& v, double& tused)
{
  const double h = sqrt(M * rp * (1.0 + ecc));   // G = 1
  r = vec(rp, 0.0, 0.0);
  v = vec(0.0, h / rp, 0.0);
  const double tscale = sqrt(rp * rp * rp / M);
  const double dt = -tscale / 20000.0;
  const double limit = (rstart > 0.0) ? 200.0 * tscale : tmax;
  double t = 0.0, prev = rp;
  while (t < limit)
  {
    const Vec k1r = v,                        k1v = twoBodyAccel(r, M);
    const Vec k2r = v + (0.5 * dt) * k1v,     k2v = twoBodyAccel(r + (0.5 * dt) * k1r, M);
    const Vec k3r = v + (0.5 * dt) * k2v,     k3v = twoBodyAccel(r + (0.5 * dt) * k2r, M);
    const Vec k4r = v + dt * k3v,             k4v = twoBodyAccel(r + dt * k3r, M);
    r = r + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
    v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    t += -dt;
    const double s = len(r);
    if (rstart > 0.0 && s >= rstart) break;
    if (rstart > 0.0 && s < prev) break;   // apocentre of a bound orbit
    prev = s;
  }
  tused = t;
}

// ---------------------------------------------------------------- diagnostic

// 2T/|W| for one galaxy on its own, from a Monte Carlo subsample carrying the
// galaxy's whole mass. Direct summation over the subsample estimates W without
// the tree's approximation, and a few thousand particles is enough to say
// whether the model is in equilibrium or about to breathe.
double virialRatio(const vector<Particle>& p, double mtot, int nsub, Rng& g)
{
  const int n = (int)p.size();
  nsub = min(nsub, n);
  vector<int> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  for (int i = 0; i < nsub; i++)
    swap(idx[i], idx[i + (int)(u01(g) * (n - i))]);
  const double m = mtot / nsub;
  double T = 0.0, W = 0.0;
  for (int i = 0; i < nsub; i++)
  {
    const Vec& vi = p[idx[i]].vel;
    T += 0.5 * m * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
    for (int j = i + 1; j < nsub; j++)
    {
      const double d = len(p[idx[i]].pos - p[idx[j]].pos);
      W -= m * m / d;
    }
  }
  return 2.0 * T / fabs(W);
}

}  // namespace

// --------------------------------------------------------------------- main

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    fprintf(stderr,
      "usage: galaxy <nbody> <outfile> [key=value ...]\n\n"
      "  mratio=1.0    mass of galaxy 2 over galaxy 1\n"
      "  mtotal=1.0    total mass of the pair\n"
      "  fdisk=0.4     disk fraction of a galaxy's mass\n"
      "  fbulge=0.1    bulge fraction; the halo takes the rest\n"
      "  rd=1.0        disk scale length of galaxy 1 (the unit of length)\n"
      "  z0=0.1        disk scale height, in rd\n"
      "  rdcut=5.0     disk truncation, in rd\n"
      "  ab=0.2        bulge Hernquist scale, in rd\n"
      "  ah=3.0        halo Hernquist scale, in rd\n"
      "  abcut=2.0     bulge truncation, in rd\n"
      "  ahcut=15.0    halo truncation, in rd\n"
      "  q=1.5         Toomre Q of the disks\n"
      "  rp=2.0        orbital pericentre, in rd\n"
      "  ecc=1.0       eccentricity: 1 parabolic, <1 bound, >1 hyperbolic\n"
      "  rstart=8.0    initial separation, in rd; the encounter time follows\n"
      "  tperi=<t>     instead of rstart: time from the start to pericentre\n"
      "  steps=N       instead of tperi: put pericentre two thirds of the way\n"
      "                through a run of N iterations, at the suggested dtime\n"
      "  inc1=60 node1=90 inc2=60 node2=-30   disk orientations, degrees\n"
      "  eps=<value>   override the softening (default: the mean interparticle\n"
      "                separation in the disk, which also sets the timestep)\n"
      "  seed=12345    random seed\n"
      "  dump=<file>   also write a plain-text subsample (x y z component galaxy)\n"
      "  dumpn=20000   how many particles to dump\n");
    return 1;
  }

  const long nbody = atol(argv[1]);
  const char* outname = argv[2];
  for (int i = 3; i < argc; i++)
  {
    const char* eq = strchr(argv[i], '=');
    if (!eq) { fprintf(stderr, "galaxy: not a key=value argument: %s\n", argv[i]); return 1; }
    opts[string(argv[i], eq - argv[i])] = string(eq + 1);
  }

  const double mratio = dparam("mratio", 1.0);
  const double mtotal = dparam("mtotal", 1.0);
  const double fdisk = dparam("fdisk", 0.4);
  const double fbulge = dparam("fbulge", 0.1);
  const double rd1 = dparam("rd", 1.0);
  const double z0f = dparam("z0", 0.1);
  const double rdcutf = dparam("rdcut", 5.0);
  const double abf = dparam("ab", 0.2);
  const double ahf = dparam("ah", 3.0);
  const double abcutf = dparam("abcut", 2.0);
  const double ahcutf = dparam("ahcut", 15.0);
  const double toomreQ = dparam("q", 1.5);
  const double rp = dparam("rp", 2.0) * rd1;
  const double ecc = dparam("ecc", 1.0);
  const unsigned long seed = (unsigned long)iparam("seed", 12345);

  if (fdisk + fbulge > 1.0) { fprintf(stderr, "galaxy: fdisk+fbulge > 1\n"); return 1; }

  const double m1 = mtotal / (1.0 + mratio), m2 = mtotal - m1;
  // A smaller galaxy is also a physically smaller one. r ~ m^(1/2) holds the
  // mean density of the two fixed, which is what keeps the less massive one
  // from being an implausibly diffuse copy.
  const double rd2 = rd1 * sqrt(m2 / m1);

  Galaxy G[2];
  const double mg[2] = {m1, m2}, rdv[2] = {rd1, rd2};
  const double incs[2] = {dparam("inc1", 60.0), dparam("inc2", 60.0)};
  const double nodes[2] = {dparam("node1", 90.0), dparam("node2", -30.0)};
  for (int k = 0; k < 2; k++)
  {
    Galaxy& g = G[k];
    g.m = mg[k];
    g.md = g.m * fdisk; g.mb = g.m * fbulge; g.mh = g.m - g.md - g.mb;
    g.rd = rdv[k];
    g.z0 = z0f * g.rd; g.rdcut = rdcutf * g.rd;
    g.ab = abf * g.rd; g.abcut = abcutf * g.rd;
    g.ah = ahf * g.rd; g.ahcut = ahcutf * g.rd;
    g.toomreQ = toomreQ;
    g.rot = Rotation(incs[k], nodes[k]);

    // Normalise each profile so that the part inside the cut carries the mass
    // the component was asked for.
    const double xd = g.rdcut / g.rd;
    g.mdInf = g.md / (1.0 - (1.0 + xd) * exp(-xd));
    const double fb_ = (g.abcut / (g.abcut + g.ab)) * (g.abcut / (g.abcut + g.ab));
    const double fh_ = (g.ahcut / (g.ahcut + g.ah)) * (g.ahcut / (g.ahcut + g.ah));
    g.mbInf = fb_ > 0.0 ? g.mb / fb_ : 0.0;
    g.mhInf = fh_ > 0.0 ? g.mh / fh_ : 0.0;

    g.buildJeans();
  }

  // Equal particle masses, so counts follow the masses. Unequal masses would
  // let the disks be sampled more finely than the halos, but they also make
  // the tree's work per particle depend on which component it came from, and
  // this dataset exists to be a load-balancing test.
  const double pmass = mtotal / nbody;
  long n1 = (long)llround(m1 / pmass);
  long n2 = nbody - n1;
  for (int k = 0; k < 2; k++)
  {
    const long n = (k == 0) ? n1 : n2;
    G[k].nd = (int)llround(n * fdisk);
    G[k].nb = (int)llround(n * fbulge);
    G[k].nh = (int)(n - G[k].nd - G[k].nb);
  }

  // ---- sample ----------------------------------------------------------
  // Each galaxy in its own frame first. The orbit is applied afterwards
  // because the timestep the run should use follows from the sample, and the
  // steps= option turns that timestep back into the encounter time -- so the
  // orbit cannot be fixed before the sample exists.
  Rng rng(seed);
  vector<Particle> raw[2];
  double virial[2];
  for (int k = 0; k < 2; k++)
  {
    Galaxy& g = G[k];
    vector<Particle>& p = raw[k];
    p.reserve(g.nd + g.nb + g.nh);
    sampleDisk(g, g.nd, rng, p);
    sampleSpheroid(g, g.jsb, g.ab, g.abcut, g.nb, rng, p);
    sampleSpheroid(g, g.jsh, g.ah, g.ahcut, g.nh, rng, p);

    // Equilibrium is a property of the galaxy on its own, so measure it before
    // the orientation and the orbit are applied.
    virial[k] = virialRatio(p, g.m, 6000, rng);

    // Recentre: the sample's own centre of mass is not exactly the origin, and
    // a residual drift would show up as the pair missing its own pericentre.
    Vec cm = vec(0, 0, 0), cv = vec(0, 0, 0);
    for (size_t i = 0; i < p.size(); i++) { cm = cm + p[i].pos; cv = cv + p[i].vel; }
    cm = (1.0 / p.size()) * cm; cv = (1.0 / p.size()) * cv;
    for (size_t i = 0; i < p.size(); i++) { p[i].pos = p[i].pos - cm; p[i].vel = p[i].vel - cv; }
  }

  // ---- scales ----------------------------------------------------------
  // Softening at the mean interparticle separation where the disk is densest
  // enough to matter: the sphere holding one particle at the disk's half-mass
  // radius. Two-body relaxation is what a smaller value buys, and a larger one
  // erases the cusp that makes this dataset interesting.
  // eps= overrides it. Coarsening the force resolution on purpose is the only
  // lever that shortens a run: the step scales with eps, so twice the
  // softening is half the iterations to the same physical time.
  const double Rhalf = 1.68 * G[0].rd;
  const double rhoDisk = G[0].sigma(Rhalf) / (2.0 * G[0].z0);
  const double eps = dparam("eps", pow(3.0 * pmass / (4.0 * M_PI * rhoDisk), 1.0 / 3.0));

  double v2sum = 0.0;
  long nv = 0;
  for (int k = 0; k < 2; k++)
    for (size_t i = 0; i < raw[k].size(); i++)
    {
      const double sp = len(raw[k][i].vel);
      v2sum += sp * sp; nv++;
    }
  // The bulk motion counts too, and at pericentre it is the larger of the two.
  // v_peri^2 = G M (1+e)/rp for any conic, which needs no integration.
  const double vperi = sqrt(mtotal * (1.0 + ecc) / rp);
  const double vint = sqrt(v2sum / nv);
  const double vrms = sqrt(vint * vint + vperi * vperi);
  const double trot = 2.0 * M_PI * G[0].rd / G[0].vc(G[0].rd);

  // barnes integrates with one fixed step for every particle, so the step is
  // set by the worst place in the model, which is the bottom of a bulge cusp.
  // Two criteria, both standard for a softened collisionless run: a particle
  // must not cross a softening length in a step, and the step must resolve the
  // free-fall time at the largest acceleration present.
  double amax = 0.0;
  for (int i = 0; i < 400; i++)
  {
    const double r = eps * pow(G[0].ahcut / eps, i / 399.0);
    const double a = G[0].massEnclosed(r) / (r * r + eps * eps);
    if (a > amax) amax = a;
  }
  const double dtVel = 0.25 * eps / vrms;
  const double dtAcc = 0.2 * sqrt(eps / amax);
  const double dt = min(dtVel, dtAcc);

  // ---- orbit -----------------------------------------------------------
  // steps=N says how long the run is going to be, and puts the pericentre
  // two thirds of the way through it. That is the useful thing to ask for:
  // the encounter, not the approach, is what makes the work move, and a
  // dataset whose pericentre falls after the last iteration is just a pair of
  // static galaxies. It is also where the model is pushed hardest -- see the
  // note printed when the time it implies is shorter than a disk rotation.
  // Three ways to say where on the orbit to start, most specific first.
  const int steps = iparam("steps", 0);
  const double tset = dparam("tperi", 0.0);
  double rstart = 0.0, twant = 0.0;
  if (steps > 0) twant = 0.66 * steps * dt;
  else if (tset > 0.0) twant = tset;
  else rstart = dparam("rstart", 8.0) * rd1;

  Vec r, v;
  double tperi = 0.0;
  orbitBack(mtotal, rp, ecc, twant, rstart, r, v, tperi);
  G[0].ctr = (-m2 / mtotal) * r;  G[0].vctr = (-m2 / mtotal) * v;
  G[1].ctr = (m1 / mtotal) * r;   G[1].vctr = (m1 / mtotal) * v;

  vector<Particle> all;
  all.reserve(nbody);
  vector<int> comp;               // 0 disk, 1 bulge, 2 halo -- for the dump
  vector<int> which;
  double vmax = 0.0;
  for (int k = 0; k < 2; k++)
  {
    Galaxy& g = G[k];
    for (size_t i = 0; i < raw[k].size(); i++)
    {
      Particle q;
      q.pos = g.rot(raw[k][i].pos) + g.ctr;
      q.vel = g.rot(raw[k][i].vel) + g.vctr;
      all.push_back(q);
      comp.push_back(i < (size_t)g.nd ? 0 : (i < (size_t)(g.nd + g.nb) ? 1 : 2));
      which.push_back(k);
      const double sp = len(q.vel);
      if (sp > vmax) vmax = sp;
    }
  }

  // ---- write -----------------------------------------------------------
  const int np = (int)all.size();
  const int ndims = 3;
  const Real tnow = 0.0;
  ofstream out(outname, ios::out | ios::binary);
  if (!out) { fprintf(stderr, "galaxy: cannot write %s\n", outname); return 1; }
  out.write((char*)&np, sizeof(int));
  out.write((char*)&ndims, sizeof(int));
  out.write((char*)&tnow, sizeof(Real));
  {
    const int BATCH = 4096;
    vector<Real> buf((size_t)BATCH * REALS_PER_PARTICLE);
    int done = 0;
    while (done < np)
    {
      const int n = min(BATCH, np - done);
      for (int i = 0; i < n; i++)
      {
        Real* t = &buf[(size_t)i * REALS_PER_PARTICLE];
        const Particle& p = all[done + i];
        t[0] = (Real)p.pos.x; t[1] = (Real)p.pos.y; t[2] = (Real)p.pos.z;
        t[3] = (Real)p.vel.x; t[4] = (Real)p.vel.y; t[5] = (Real)p.vel.z;
        t[6] = (Real)pmass;   t[7] = (Real)eps;
      }
      out.write((char*)&buf[0], (streamsize)n * SIZE_PER_PARTICLE);
      done += n;
    }
  }
  out.close();

  const string dump = sparam("dump", "");
  if (!dump.empty())
  {
    const int dn = iparam("dumpn", 20000);
    const int stride = max(1, np / max(dn, 1));
    FILE* f = fopen(dump.c_str(), "w");
    if (!f) { fprintf(stderr, "galaxy: cannot write %s\n", dump.c_str()); return 1; }
    fprintf(f, "# x y z component galaxy\n");
    for (int i = 0; i < np; i += stride)
      fprintf(f, "%.6g %.6g %.6g %d %d\n", all[i].pos.x, all[i].pos.y, all[i].pos.z,
              comp[i], which[i]);
    fclose(f);
  }

  // ---- report ----------------------------------------------------------
  const double sep = len(G[1].ctr - G[0].ctr);
  const int itPeri = (int)ceil(tperi / dt);
  printf("galaxy: %d particles -> %s\n", np, outname);
  printf("  units          G = 1, total mass %g, disk scale length %g\n", mtotal, rd1);
  printf("  galaxy 1       m %.4g  rd %.3g  disk/bulge/halo %d/%d/%d  inc %g node %g\n",
         G[0].m, G[0].rd, G[0].nd, G[0].nb, G[0].nh, incs[0], nodes[0]);
  printf("  galaxy 2       m %.4g  rd %.3g  disk/bulge/halo %d/%d/%d  inc %g node %g\n",
         G[1].m, G[1].rd, G[1].nd, G[1].nb, G[1].nh, incs[1], nodes[1]);
  printf("  particle mass  %.4g\n", pmass);
  printf("  orbit          pericentre %.3g, eccentricity %.3g, pericentre at t = %.3g\n",
         rp, ecc, tperi);
  printf("                 initial separation %.4g, relative speed %.4g\n", sep, len(v));
  printf("  equilibrium    2T/|W| = %.3f (galaxy 1), %.3f (galaxy 2)\n", virial[0], virial[1]);
  printf("  timescales     disk rotation at rd %.3g, internal v_rms %.3g, v_max %.3g\n",
         trot, vint, vmax);
  printf("  suggested      -eps=%.4g -dtime=%.4g -killat=%d\n",
         eps, dt, (int)ceil(1.5 * tperi / dt));
  printf("                 pericentre falls at iteration %d; dtime is the smaller of\n", itPeri);
  printf("                 %.4g (a quarter of a softening length per step) and %.4g\n", dtVel, dtAcc);
  printf("                 (the free-fall time at the largest acceleration, %.3g)\n", amax);
  if (tperi < 0.25 * trot)
    printf("  NOTE           pericentre arrives in %.3g, less than a quarter of a disk\n"
           "                 rotation (%.3g). The galaxies start already interpenetrating\n"
           "                 and the disks barely turn during the run: this is a dense,\n"
           "                 moving, evolving configuration, but it is not a resolved\n"
           "                 encounter. Drop steps= and use tperi= for that.\n", tperi, trot);
  return 0;
}
