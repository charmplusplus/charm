
const double CHARM_OVERHEAD = 0.5E-6;    // time to enqueue a msg - 0.5 us

class BigSimNetwork
{
protected:
  double alpha;
  char *myname;
public:
  inline double alphacost() { return alpha; }
  inline double charmcost() { return CHARM_OVERHEAD; }
  inline char *name() { return myname; }
  virtual double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) = 0;
  virtual void print() = 0;
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

class BlueGeneLNetwork: public BigSimNetwork
{
private:
  double bandwidth;
  int packetsize;
  double linkcost;
public:
  BlueGeneLNetwork() { 
    myname = "bluegenel";
    packetsize = 256;
    bandwidth = 175E6; alpha = 2E-6; 
    linkcost = packetsize/bandwidth;
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    int sx, sy, sz;
    int xd=BG_ABS(ox-nx), yd=BG_ABS(oy-ny), zd=BG_ABS(oz-nz);
    BgGetSize(&sx, &sy, &sz);
    if (xd>sx/2) xd = sx-xd;
    if (yd>sy/2) yd = sy-yd;
    if (zd>sz/2) zd = sz-zd;
    CmiAssert(xd>=0 && yd>=0 && zd>=0);
    int hops = xd + yd + zd;
    int numpackets = bytes/packetsize;
    if (bytes%packetsize) numpackets++;
    return  linkcost * hops * numpackets;
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


class IBMNetwork: public BigSimNetwork
{
private:
  double alpha;
  double bandwidth;
public:
  IBMNetwork() { 
    myname = "ibmNetwork";
    alpha = 2.5*1e-6;
    //bandwidth = 12*1e9; 
    bandwidth = 1.7*1e9; 
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    return alpha + bytes/bandwidth;
  }
  void print() {
    CmiPrintf("alpha: %e	bandwidth :%e.\n", alpha, bandwidth);
  }
};



