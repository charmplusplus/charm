

class BigSimNetwork
{
protected:
  double alpha;
  char *myname;
public:
  inline double alphacost() { return alpha; }
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
    alpha = 0.1E-6; 
    packetsize = PACKETSIZE;
  }
  inline double latency(int ox, int oy, int oz, int nx, int ny, int nz, int bytes) {
    int numpackets;
    int xd=ABS(ox-nx), yd=ABS(oy-ny), zd=ABS(oz-nz);
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
    int xd=ABS(ox-nx), yd=ABS(oy-ny), zd=ABS(oz-nz);
    CmiAssert(xd>=0 && yd>=0 && zd>=0);
    int hops = xd+yd+zd;
    double packetcost = neighborlatency + hoplatency * hops + bytes * 1e-9;
    return packetcost;
  }
  void print() {
    CmiPrintf("alpha: %e	latency: %es	hop latency:%es.\n", alpha, neighborlatency, hoplatency);
  }
};



