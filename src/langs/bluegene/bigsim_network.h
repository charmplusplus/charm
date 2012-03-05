#ifndef __BLUE_NETWORK_H
#define __BLUE_NETWORK_H

class BigSimNetwork
{
protected:
  double alpha;
  const char *myname;
  int dimNX, dimNY, dimNZ, dimNT;	// dimensions for a 3D Torus/Mesh
public:
  inline double alphacost() { return alpha; }
  inline const char *name() { return myname; }
  inline void setDimensions(int x, int y, int z, int t) {
    dimNX = x; dimNY = y;
    dimNZ = z; dimNT = t;
  }
  virtual double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) = 0;
  virtual void print() = 0;
};

class DummyNetwork: public BigSimNetwork
{
public:
  DummyNetwork() { 
    myname = "dummy";
    alpha = 0.0;
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    return 0;
  }
  void print() {
    CmiPrintf("Dummy network.\n");
  }
};

const double BANDWIDTH = 256E6;

class LemieuxNetwork: public BigSimNetwork
{
private:
  double bandwidth;
public:
  LemieuxNetwork() { 
    myname = "lemieux";
    bandwidth = BANDWIDTH; alpha = 8E-6; 
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    if (ox == nx && oy == ny && oz == nz) return 0.0;    // same PE
    return bytes/bandwidth;
  }
  void print() {
    CmiPrintf("bandwidth: %e; alpha: %e.\n", bandwidth, alpha);
  }
};

const int CYCLES_PER_HOP  =   5;
const int CYCLES_PER_CORNER = 75;
const double CYCLE_TIME_FACTOR = 0.001;   /* one cycle = nanosecond = 10^(-3) us */
const int PACKETSIZE = 1024;

class BlueGeneNetwork: public BigSimNetwork
{
private:
  int packetsize;
public:
  BlueGeneNetwork() { 
    myname = "bluegene";
    alpha = 0.1E-6;    // 2E-6; 
    packetsize = PACKETSIZE;
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    int numpackets;
    int xd=BG_ABS(ox-nx), yd=BG_ABS(oy-ny), zd=BG_ABS(oz-nz);
    int ncorners = 2;
    ncorners -= (xd?0:1 + yd?0:1 + zd?0:1);
    ncorners = (ncorners<0)?0:ncorners;
    double packetcost = (ncorners*CYCLES_PER_CORNER + (xd+yd+zd)*CYCLES_PER_HOP)*CYCLE_TIME_FACTOR*1E-6;
    numpackets = bytes/packetsize;
    if (bytes%packetsize) numpackets++;
    return  packetcost * numpackets;
  }
  void print() {
    CmiPrintf("alpha: %e	packetsize: %d	CYCLE_TIME_FACTOR:%e.\n", alpha, packetsize, CYCLE_TIME_FACTOR);
    CmiPrintf("CYCLES_PER_HOP: %d	CYCLES_PER_CORNER: %d.\n", CYCLES_PER_HOP, CYCLES_PER_CORNER);
  }
};

class BlueGenePNetwork: public BigSimNetwork
{
private:
  double bandwidth;
  int packetsize;
public:
  BlueGenePNetwork() { 
    myname = "bluegenep";
    packetsize = 256;
    bandwidth = 0.18E9; alpha = 6E-6;
  }
  inline void set_latency(double lat) {alpha = lat;}
  inline void set_bandwidth(double bw) {bandwidth = bw;}
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    /* (int xd=BG_ABS(ox-nx), yd=BG_ABS(oy-ny), zd=BG_ABS(oz-nz);
    if (xd > dimNX/2) xd = dimNX - xd;
    if (yd > dimNY/2) yd = dimNY - yd;
    if (zd > dimNZ/2) zd = dimNZ - zd;
    CmiAssert(xd>=0 && yd>=0 && zd>=0);
    int hops = xd + yd + zd; 

    int numpackets = bytes/packetsize;
    if (bytes%packetsize) numpackets++;
    return  alpha + (numpackets * packetsize)/bandwidth; */
    return alpha + (bytes/bandwidth);
  }
  void print() {
    CmiPrintf("bandwidth: %e; alpha: %e.\n", bandwidth, alpha);
  }
};

class RedStormNetwork: public BigSimNetwork
{
private:
  int packetsize;
  double hoplatency;
  double neighborlatency;
public:
  RedStormNetwork() { 
    myname = "redstorm";
    alpha = 0.1E-6; 
    packetsize = PACKETSIZE;
    neighborlatency = 2e-6;
    hoplatency = 44.8e-9;
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    if (ox == nx && oy == ny && oz == nz) return 0.0;    // same PE
    int xd=BG_ABS(ox-nx), yd=BG_ABS(oy-ny), zd=BG_ABS(oz-nz);
    CmiAssert(xd>=0 && yd>=0 && zd>=0);
    int hops = xd+yd+zd;
    double packetcost = neighborlatency + hoplatency * hops + bytes * 1e-9;
    return packetcost;
  }
  void print() {
    CmiPrintf("alpha: %e	latency: %es	hop latency:%es.\n", alpha, neighborlatency, hoplatency);
  }
};


class IBMPowerNetwork: public BigSimNetwork
{
private:
  double permsg;
  double bandwidth;
public:
  IBMPowerNetwork() { 
    myname = "ibmpower";
    alpha = 1*1e-6;
    // permsg = 2.5*1e-6;
    permsg = 2.5*(1e-6)*20;
    // bandwidth = 12*1e9; 
    bandwidth = 1.7*1e9; 
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    if (ox == nx && oy == ny && oz == nz) return 0.0;    // same PE
    return permsg + bytes/bandwidth;
  }
  void print() {
    CmiPrintf("alpha: %e	bandwidth :%e.\n", alpha, bandwidth);
  }
};


// Used for the Simple Latency model
class ParamNetwork: public BigSimNetwork
{
private:
  double bandwidth;        // in Bytes/second
  double cost_per_packet;  // in seconds
  int    packet_size;      // in bytes
public:
  ParamNetwork() {
    myname = "parameter";
    alpha = 0.123;
    bandwidth = 0.0;
    cost_per_packet = 0.0;
    packet_size = 0;
  }
  inline void set_latency(double lat) {alpha = lat;}
  inline void set_bandwidth(double bw) {bandwidth = bw;}
  inline void set_cost_per_packet(double cpp) {cost_per_packet = cpp;}
  inline void set_packet_size(int ps) {packet_size = ps;}
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    double lat = 0.0;
    if (cost_per_packet > 0.0) {
      CmiAssert(packet_size != 0);
      int num_packets = bytes / packet_size;
      if (bytes > (num_packets * packet_size)) {  // ceiling of (bytes / packet_size)
	num_packets++;
      }
      return (alpha + (bytes / bandwidth) + (cost_per_packet * num_packets));
    }
    return (alpha + (bytes / bandwidth));
  }
  void print() {
    CmiPrintf("alpha: %e, bandwidth: %e, cost per packet: %e, packet size: %d\n", 
	      alpha, bandwidth, cost_per_packet, packet_size);
  }
};

#endif
