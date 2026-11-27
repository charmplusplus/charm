// commbench driver: argument parsing, the case list, validation,
// calibration, the shuffled repeat loop and the JSON-lines output.
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "build_info.h"
#include "endpoints.h"
#include "persistent.h"

int g_maxSize;
int g_deliverUser, g_sourceUser;
extern char **Cmi_argvcopy;  // reconverse: argv as launched, for tracing

void commbenchInitNode() {
  sideInitNode();
  converseInitNode();
}
void commbenchInitProc() { converseInitProc(); }

namespace {

const char *const kUsage =
    "Usage: commbench [charm options] [--entities LIST] [--methods LIST]\n"
    "         [--sizes LIST|MIN:MAX:FACTOR] [--extra-sizes LIST] [--reps N]\n"
    "         [--sample-time SEC] [--pairs auto|A:B,C:D] [--seed N] [--out FILE]\n"
    "         [--deliver message|user] [--source message|user]\n"
    "         [--no-validate] [--list] [--progress] [--run-id ID] [--machine NAME]\n"
    "  entities: handler chare group nodegroup array (or all)\n"
    "  methods:  marshal msg nocopy.{reg,unreg,prereg} post.{reg,unreg,prereg}\n"
    "            get.{reg,unreg,prereg}.{nodereg,dereg} put.{reg,unreg,prereg}.{nodereg,dereg}\n"
    "            sync node rget rput persistent.{send,recv} (or all); a prefix\n"
    "            selects every method below it, e.g. nocopy, get, get.reg\n"
    "  sizes accept k and m suffixes (powers of 1024).\n"
    "  --deliver user: received data must end in the receiver's own buffer;\n"
    "  --source user: sent data starts in the sender's own buffer.\n";

[[noreturn]] void usageError(const char *fmt, const char *arg) {
  fprintf(stderr, "commbench: ");
  fprintf(stderr, fmt, arg);
  fprintf(stderr, "\n%s", kUsage);
  CkExit(2);
  abort();
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(sep, start);
    if (end == std::string::npos) end = s.size();
    if (end > start) out.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

long long parseSize(const std::string &t) {
  char *end = nullptr;
  errno = 0;
  long long v = strtoll(t.c_str(), &end, 10);
  if (errno || end == t.c_str()) usageError("bad size '%s'", t.c_str());
  if (*end == 'k' || *end == 'K') v <<= 10, end++;
  else if (*end == 'm' || *end == 'M') v <<= 20, end++;
  if (*end) usageError("bad size '%s'", t.c_str());
  if (v < 1 || v > (1LL << 30)) usageError("size '%s' out of range [1, 1G]", t.c_str());
  return v;
}

// "8,64,1k" or "8:16m:2" (min:max:factor).
std::vector<int> parseSizes(const std::string &spec) {
  std::vector<int> out;
  for (const std::string &t : split(spec, ',')) {
    std::vector<std::string> r = split(t, ':');
    if (r.size() == 1) {
      out.push_back((int)parseSize(r[0]));
    } else if (r.size() == 3) {
      long long lo = parseSize(r[0]), hi = parseSize(r[1]);
      double f = atof(r[2].c_str());
      if (f <= 1.0) usageError("size factor in '%s' must be > 1", t.c_str());
      for (double v = (double)lo; v <= (double)hi * (1 + 1e-9); v *= f) out.push_back((int)llround(v));
    } else {
      usageError("bad size range '%s'", t.c_str());
    }
  }
  return out;
}

std::vector<int> defaultSizes() {
  std::vector<int> s;
  for (int v = 8; v <= (16 << 20); v *= 2) s.push_back(v);
  s.push_back(8096);
  s.push_back(8097);
  return s;
}

int entityByName(const std::string &n) {
  for (int e = 0; e < E_COUNT; e++)
    if (n == kEntityName[e]) return e;
  return -1;
}

std::vector<bool> parseEntities(const std::string &spec) {
  std::vector<bool> on(E_COUNT, false);
  for (const std::string &t : split(spec, ',')) {
    if (t == "all") {
      std::fill(on.begin(), on.end(), true);
      continue;
    }
    int e = entityByName(t);
    if (e < 0) usageError("unknown entity '%s'", t.c_str());
    on[e] = true;
  }
  return on;
}

std::vector<bool> parseMethods(const std::string &spec) {
  std::vector<bool> on(kNumMethods, false);
  for (const std::string &t : split(spec, ',')) {
    bool hit = false;
    for (int m = 0; m < kNumMethods; m++) {
      std::string n = kMethods[m].name;
      // A prefix ending at a dot selects every method below it: "get" or "get.reg".
      if (t == "all" || t == n || n.compare(0, t.size() + 1, t + ".") == 0) on[m] = hit = true;
    }
    if (!hit) usageError("unknown method '%s'", t.c_str());
  }
  return on;
}

const char *localityOf(int a, int b) {
  if (a == b) return "same_pe";
  if (CmiNodeOf(a) == CmiNodeOf(b)) return "same_process";
  if (CmiPhysicalNodeID(a) == CmiPhysicalNodeID(b)) return "same_host";
  return "cross_host";
}

std::string jstr(const std::string &s) {
  std::ostringstream o;
  o << '"';
  for (char ch : s) {
    unsigned char c = (unsigned char)ch;
    if (c == '"' || c == '\\')
      o << '\\' << ch;
    else if (c < 0x20)
      o << "\\u" << std::hex << std::setfill('0') << std::setw(4) << (int)c << std::dec;
    else
      o << ch;
  }
  o << '"';
  return o.str();
}

// as printf's %.7g
std::string jnum(double v) {
  std::ostringstream o;
  o << std::setprecision(7) << v;
  return o.str();
}

template <class T>
std::string jarr(const std::vector<T> &v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); i++) o += (i ? "," : "") + std::to_string(v[i]);
  return o + "]";
}

// splitmix64: the shuffle must give the same order for a seed on every
// platform, which std::shuffle does not promise.
struct Rng {
  CmiUInt8 x;
  CmiUInt8 next() {
    CmiUInt8 z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
};

struct CaseSpec {
  int entity, method, pair;
  std::string skip;  // non-empty: not run, with this reason
};

struct Item {
  int cs;  // index into cases
  int size;
  int iters = 0, warmup = 0;
  bool live = true, validated = false;
};

struct Outcome {
  double elapsed = 0;
  bool ok = true;
  std::string error;
  std::vector<int> pesA, pesB;
  std::string aux;  // side A's aux note
};

}  // namespace

class Driver : public CBase_Driver {
  // options
  std::vector<bool> entities, methods;
  std::vector<int> sizes;
  int reps = 5;
  double sampleTime = 0.1;
  std::string pairSpec = "auto", outPath, runId, machine;
  CmiUInt8 seed = 0;
  bool validate = true, listOnly = false, progress = false;
  std::vector<std::string> argvCopy;

  std::vector<std::pair<int, int>> pairs;
  std::vector<CaseSpec> cases;
  std::vector<Item> items;
  FILE *out = stdout;
  std::string started;

  CProxy_GroupEP groupP;
  CProxy_NodeGroupEP nodeGroupP;
  CProxy_ArrayEP arrayP;
  CProxy_HandlerEP handlerP;
  std::vector<CProxy_ChareEP> charesA, charesB;  // per pair

 public:
  Driver(CkArgMsg *m) {
    char **argv = m->argv;
    for (int i = 1; i < m->argc; i++) argvCopy.push_back(argv[i]);
    bool help = !parseArgs(argv);
    delete m;
    if (help) {
      CkExit();
      return;
    }

    g_maxSize = *std::max_element(sizes.begin(), sizes.end());
    makePairs();
    makeCases();

    if (listOnly) {
      printList();
      CkExit();
      return;
    }

    groupP = CProxy_GroupEP::ckNew();
    nodeGroupP = CProxy_NodeGroupEP::ckNew();
    handlerP = CProxy_HandlerEP::ckNew();
    CkArrayOptions opts(CkNumPes());
    opts.setMap(CProxy_PeMap::ckNew());
    arrayP = CProxy_ArrayEP::ckNew(opts);
    thisProxy.go();
  }

  // Returns false for --help.
  bool parseArgs(char **argv) {
    char *s = nullptr;
    if (CmiGetArgFlagDesc(argv, "--help", "print usage")) {
      fprintf(stderr, "%s", kUsage);
      return false;
    }
    entities.assign(E_COUNT, true);
    methods.assign(kNumMethods, true);
    if (CmiGetArgStringDesc(argv, "--entities", &s, "entities to run")) entities = parseEntities(s);
    if (CmiGetArgStringDesc(argv, "--methods", &s, "methods to run")) methods = parseMethods(s);
    sizes = defaultSizes();
    if (CmiGetArgStringDesc(argv, "--sizes", &s, "payload sizes")) sizes = parseSizes(s);
    if (CmiGetArgStringDesc(argv, "--extra-sizes", &s, "added payload sizes")) {
      std::vector<int> x = parseSizes(s);
      sizes.insert(sizes.end(), x.begin(), x.end());
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    if (sizes.empty()) usageError("%sno sizes", "");
    if (CmiGetArgIntDesc(argv, "--reps", &reps, "repeats") && reps < 1) usageError("%s--reps must be >= 1", "");
    if (CmiGetArgDoubleDesc(argv, "--sample-time", &sampleTime, "seconds per sample") && sampleTime <= 0)
      usageError("%s--sample-time must be > 0", "");
    if (CmiGetArgStringDesc(argv, "--pairs", &s, "auto or A:B,C:D")) pairSpec = s;
    if (CmiGetArgStringDesc(argv, "--seed", &s, "shuffle seed")) {
      seed = strtoull(s, nullptr, 0);
    } else {
      seed = (CmiUInt8)time(nullptr) * 1000003ULL ^ (CmiUInt8)getpid();
    }
    if (CmiGetArgStringDesc(argv, "--out", &s, "output file")) outPath = s;
    if (CmiGetArgStringDesc(argv, "--run-id", &s, "run id")) runId = s;
    if (CmiGetArgStringDesc(argv, "--machine", &s, "machine name")) machine = s;
    g_deliverUser = g_sourceUser = 0;
    if (CmiGetArgStringDesc(argv, "--deliver", &s, "message or user")) g_deliverUser = parseMode("--deliver", s);
    if (CmiGetArgStringDesc(argv, "--source", &s, "message or user")) g_sourceUser = parseMode("--source", s);
    validate = !CmiGetArgFlagDesc(argv, "--no-validate", "skip validation");
    listOnly = CmiGetArgFlagDesc(argv, "--list", "print the case list and exit");
    progress = CmiGetArgFlagDesc(argv, "--progress", "report each step on stderr");
    if (CmiGetArgc(argv) > 1) usageError("unknown argument '%s'", argv[1]);
    return true;
  }

  static int parseMode(const char *opt, const std::string &v) {
    if (v == "message") return 0;
    if (v == "user") return 1;
    std::string m = std::string(opt) + " must be message or user, not '%s'";
    usageError(m.c_str(), v.c_str());
  }

  void makePairs() {
    int n = CkNumPes();
    if (pairSpec == "auto") {
      const char *classes[] = {"same_process", "same_host", "cross_host"};
      for (const char *c : classes)
        for (int b = 1; b < n; b++)
          if (!strcmp(localityOf(0, b), c)) {
            pairs.push_back({0, b});
            break;
          }
      if (pairs.empty()) usageError("%s--pairs auto needs at least 2 PEs", "");
      return;
    }
    for (const std::string &t : split(pairSpec, ',')) {
      int a, b;
      char extra;
      if (sscanf(t.c_str(), "%d:%d%c", &a, &b, &extra) != 2) usageError("bad pair '%s'", t.c_str());
      if (a < 0 || b < 0 || a >= n || b >= n) usageError("pair '%s' names a PE that does not exist", t.c_str());
      if (a == b) usageError("pair '%s': same_pe pairs are not supported", t.c_str());
      pairs.push_back({a, b});
    }
  }

  void makeCases() {
    for (int p = 0; p < (int)pairs.size(); p++) {
      bool sameNode = CmiNodeOf(pairs[p].first) == CmiNodeOf(pairs[p].second);
      for (int e = 0; e < E_COUNT; e++) {
        if (!entities[e]) continue;
        for (int m = 0; m < kNumMethods; m++) {
          if (!methods[m] || (e == E_HANDLER) != isConverseMethod(m)) continue;
          CaseSpec c{e, m, p, ""};
          if (sameNode && e == E_NODEGROUP) c.skip = "same_process: self-send";
          if (sameNode && familyOf(m) == F_NODE) c.skip = "same_process: node send to its own node";
          cases.push_back(c);
        }
      }
    }
    for (int c = 0; c < (int)cases.size(); c++) {
      if (!cases[c].skip.empty()) continue;
      for (int sz : sizes) {
        Item it;
        it.cs = c;
        it.size = sz;
        items.push_back(it);
      }
    }
  }

  void printList() {
    CkPrintf("commbench: %d PEs, %d processes; %zu pairs, %zu cases, %zu sizes, %zu (case, size) items\n",
             CkNumPes(), CkNumNodes(), pairs.size(), cases.size(), sizes.size(), items.size());
    std::string sz;
    for (int s : sizes) sz += (sz.empty() ? "" : ",") + std::to_string(s);
    CkPrintf("sizes: %s\n", sz.c_str());
    for (const CaseSpec &c : cases) {
      int a = pairs[c.pair].first, b = pairs[c.pair].second;
      CkPrintf("%-9s %-18s pair %d:%d %-12s %s\n", kEntityName[c.entity], kMethods[c.method].name, a, b,
               localityOf(a, b), c.skip.empty() ? "run" : ("skip: " + c.skip).c_str());
    }
  }

  void go() {
    if (!outPath.empty()) {
      out = fopen(outPath.c_str(), "w");
      if (!out) CkAbort("commbench: cannot open %s: %s\n", outPath.c_str(), strerror(errno));
    }
    makeChares();
    writeRunRecord();
    for (const CaseSpec &c : cases)
      if (!c.skip.empty()) writeSkip(c, -1, c.skip);

    for (Item &it : items) prepare(it);

    Rng rng{seed};
    std::vector<int> order;
    for (int i = 0; i < (int)items.size(); i++)
      if (items[i].live) order.push_back(i);
    for (int rep = 0; rep < reps; rep++) {
      for (int i = (int)order.size() - 1; i > 0; i--) std::swap(order[i], order[rng.next() % (CmiUInt8)(i + 1)]);
      for (int idx : order) {
        Item &it = items[idx];
        step("rep", rep, it);
        Outcome o = sample(it, false, it.iters, it.warmup);
        writeSample(it, rep, o);
      }
    }
    if (out != stdout) fclose(out);
    CkExit();
  }

  void makeChares() {
    for (auto &p : pairs) {
      CkFuture fa = CkCreateFuture(), fb = CkCreateFuture();
      CProxy_ChareEP::ckNew(fa, p.first);
      CProxy_ChareEP::ckNew(fb, p.second);
      ResultMsg *ra = (ResultMsg *)CkWaitFuture(fa);
      ResultMsg *rb = (ResultMsg *)CkWaitFuture(fb);
      charesA.push_back(CProxy_ChareEP(ra->cid));
      charesB.push_back(CProxy_ChareEP(rb->cid));
      delete ra;
      delete rb;
      CkReleaseFuture(fa);
      CkReleaseFuture(fb);
    }
  }

  // Validation, then calibration: 10 iterations set the count for all repeats.
  void prepare(Item &it) {
    if (validate) {
      step("validate", -1, it);
      Outcome o = sample(it, true, 1, 0);
      if (!o.ok) {
        it.live = false;
        writeSkip(cases[it.cs], it.size, "validation failed: " + o.error);
        return;
      }
      it.validated = true;
    }
    step("calibrate", -1, it);
    Outcome o = sample(it, false, 10, 2);
    double per = o.elapsed / 10;
    double n = per > 0 ? std::ceil(sampleTime / per) : 100000;
    it.iters = (int)std::min(100000.0, std::max(10.0, n));
    it.warmup = std::max(2, it.iters / 10);
  }

  void step(const char *what, int rep, const Item &it) {
    if (!progress) return;
    const CaseSpec &c = cases[it.cs];
    fprintf(stderr, "commbench: %s%s%s %s %s %d:%d size %d\n", what, rep >= 0 ? " " : "",
            rep >= 0 ? std::to_string(rep).c_str() : "", kEntityName[c.entity], kMethods[c.method].name,
            pairs[c.pair].first, pairs[c.pair].second, it.size);
    fflush(stderr);
  }

  static ResultMsg *waitResult(CkFuture f) {
    ResultMsg *r = (ResultMsg *)CkWaitFuture(f);
    CkReleaseFuture(f);
    return r;
  }

  // setup on both sides, the timed run on side A, teardown A then B (the
  // persistent case needs A to release its message before B's channel goes).
  template <class P>
  Outcome runPair(P A, P B, int a, int b, const Item &it, bool val, int iters, int warmup) {
    int m = cases[it.cs].method;
    CkFuture fa = CkCreateFuture(), fb = CkCreateFuture();
    A.setup(m, it.size, ROLE_A, a, b, val, B, fa);
    B.setup(m, it.size, ROLE_B, b, a, val, A, fb);
    delete waitResult(fa);
    delete waitResult(fb);

    Outcome o;
    CkFuture fr = CkCreateFuture();
    A.run(iters, warmup, fr);
    ResultMsg *r = waitResult(fr);
    o.elapsed = r->elapsed;
    delete r;

    const char *names[2] = {"side A", "side B"};
    for (int side = 0; side < 2; side++) {
      CkFuture ft = CkCreateFuture();
      if (side == 0) A.teardown(ft);
      else B.teardown(ft);
      ResultMsg *t = waitResult(ft);
      if (!t->ok) {
        o.ok = false;
        o.error += (o.error.empty() ? "" : "; ") + std::string(names[side]) + ": " + t->error;
      }
      if (side == 0) o.aux = t->aux;
      std::vector<int> &pes = side == 0 ? o.pesA : o.pesB;
      pes.assign(t->pes, t->pes + t->npes);
      delete t;
    }
    return o;
  }

  Outcome sample(const Item &it, bool val, int iters, int warmup) {
    const CaseSpec &c = cases[it.cs];
    int a = pairs[c.pair].first, b = pairs[c.pair].second;
    // Element proxies are built from the ids, not with proxy[i]: that path
    // goes through CProxy(CK_DELCTOR_PARAM), which leaves delegatedGroupId
    // uninitialized, and a nonzero value makes sending or packing the proxy
    // spin in CkLocalNodeBranch.
    switch (c.entity) {
      case E_CHARE: return runPair(charesA[c.pair], charesB[c.pair], a, b, it, val, iters, warmup);
      case E_GROUP: {
        CkGroupID g = groupP.ckGetGroupID();
        return runPair(CProxyElement_GroupEP(g, a), CProxyElement_GroupEP(g, b), a, b, it, val, iters, warmup);
      }
      case E_NODEGROUP: {
        CkGroupID g = nodeGroupP.ckGetGroupID();
        return runPair(CProxyElement_NodeGroupEP(g, CkNodeOf(a)), CProxyElement_NodeGroupEP(g, CkNodeOf(b)), a, b,
                       it, val, iters, warmup);
      }
      case E_ARRAY: {
        CkArrayID id = arrayP.ckGetArrayID();
        return runPair(CProxyElement_ArrayEP(id, CkArrayIndex1D(a)), CProxyElement_ArrayEP(id, CkArrayIndex1D(b)), a,
                       b, it, val, iters, warmup);
      }
      default: {
        CkGroupID g = handlerP.ckGetGroupID();
        return runPair(CProxyElement_HandlerEP(g, a), CProxyElement_HandlerEP(g, b), a, b, it, val, iters, warmup);
      }
    }
  }

  void emit(const std::string &line) {
    fputs(line.c_str(), out);
    fputc('\n', out);
    fflush(out);
  }

  std::string caseFields(const CaseSpec &c) {
    int a = pairs[c.pair].first, b = pairs[c.pair].second;
    return "\"run_id\":" + jstr(runId) + ",\"tool\":\"commbench\",\"entity\":" + jstr(kEntityName[c.entity]) +
           ",\"method\":" + jstr(kMethods[c.method].name) + ",\"memory\":\"host\",\"pattern\":\"lat\"" +
           ",\"pair\":[" + std::to_string(a) + "," + std::to_string(b) + "],\"locality\":" +
           jstr(localityOf(a, b));
  }

  void writeSkip(const CaseSpec &c, int size, const std::string &reason) {
    std::string s = "{\"type\":\"skip\"," + caseFields(c);
    if (size >= 0) s += ",\"size\":" + std::to_string(size);
    emit(s + ",\"reason\":" + jstr(reason) + "}");
  }

  void writeSample(const Item &it, int rep, const Outcome &o) {
    const CaseSpec &c = cases[it.cs];
    int a = pairs[c.pair].first, b = pairs[c.pair].second;
    bool sameNode = CmiNodeOf(a) == CmiNodeOf(b);
    double rt = 1e6 * o.elapsed / it.iters;
    std::string aux = "{\"lat_roundtrip_us\":" + jnum(rt) + ",\"elapsed_s\":" + jnum(o.elapsed);
    if (c.entity == E_NODEGROUP || familyOf(c.method) == F_NODE)
      aux += ",\"recv_pes_a\":" + jarr(o.pesA) + ",\"recv_pes_b\":" + jarr(o.pesB);
    if (!o.aux.empty()) aux += "," + o.aux;
    aux += "}";
    emit("{\"type\":\"sample\"," + caseFields(c) + ",\"protocol\":" + jstr(protocolOf(c.method, sameNode)) +
         ",\"size\":" + std::to_string(it.size) + ",\"iters\":" + std::to_string(it.iters) +
         ",\"warmup\":" + std::to_string(it.warmup) + ",\"window\":1,\"pairs_concurrent\":1,\"rep\":" +
         std::to_string(rep) + ",\"metric\":\"lat_oneway_us\",\"value\":" + jnum(rt / 2) + ",\"aux\":" + aux +
         ",\"validated\":" + (it.validated ? "true" : "false") + "}");
  }

  static std::string cmkFlags() {
    std::vector<std::pair<const char *, long>> f;
#define CB_FLAG(x) f.push_back({#x, (long)(x)})
#ifdef CMK_OPTIMIZE
    CB_FLAG(CMK_OPTIMIZE);
#endif
#ifdef CMK_ERROR_CHECKING
    CB_FLAG(CMK_ERROR_CHECKING);
#endif
#ifdef CMK_SMP
    CB_FLAG(CMK_SMP);
#endif
#ifdef CMK_USE_SHMEM
    CB_FLAG(CMK_USE_SHMEM);
#endif
#ifdef CMK_USE_CMA
    CB_FLAG(CMK_USE_CMA);
#endif
#ifdef CMK_TRACE_ENABLED
    CB_FLAG(CMK_TRACE_ENABLED);
#endif
#ifdef CMK_WITH_STATS
    CB_FLAG(CMK_WITH_STATS);
#endif
#ifdef CMK_REPLAYSYSTEM
    CB_FLAG(CMK_REPLAYSYSTEM);
#endif
#ifdef CMK_ONESIDED_IMPL
    CB_FLAG(CMK_ONESIDED_IMPL);
#endif
#ifdef CMK_REG_REQUIRED
    CB_FLAG(CMK_REG_REQUIRED);
#endif
#ifdef CMK_NOCOPY_DIRECT_BYTES
    CB_FLAG(CMK_NOCOPY_DIRECT_BYTES);
#endif
#ifdef CMK_PERSISTENT_COMM
    CB_FLAG(CMK_PERSISTENT_COMM);
#endif
#ifdef CMK_CUDA
    CB_FLAG(CMK_CUDA);
#endif
#ifdef CMK_HIP
    CB_FLAG(CMK_HIP);
#endif
#undef CB_FLAG
    std::string o = "{";
    for (size_t i = 0; i < f.size(); i++) o += (i ? ",\"" : "\"") + std::string(f[i].first) + "\":" + std::to_string(f[i].second);
    return o + "}";
  }

  // The launch arguments as Converse saw them (it removes +pe, +p and +ppn
  // before copying); Charm++ hands the program a copy with every runtime
  // argument removed.
  static std::vector<std::string> launchArgs() {
    std::vector<std::string> v;
    for (char **a = Cmi_argvcopy; a && *a; a++) v.push_back(*a);
    return v;
  }

  static std::string runtimeArgv() {
    std::string o = "[";
    std::vector<std::string> v = launchArgs();
    for (size_t i = 0; i < v.size(); i++) o += (i ? "," : "") + jstr(v[i]);
    return o + "]";
  }

  // IPC settings: ipc is what the launch asked for (+ipc or +ipcmode, and no
  // +noipc); ipc_active is whether the shared-memory pool came up
  // (CmiIpcEnabled), which a single-process job never has.
  static std::string ipcFlags() {
    std::vector<std::string> v = launchArgs();
    bool on = false, off = false;
    std::string mode, cutoff, pool;
    for (size_t i = 0; i < v.size(); i++) {
      const std::string &a = v[i];
      std::string next = i + 1 < v.size() ? v[i + 1] : "";
      if (a == "+ipc") on = true;
      else if (a == "+noipc") off = true;
      else if (a == "+ipcmode") on = true, mode = next;
      else if (a == "++" CMI_IPC_CUTOFF_ARG) cutoff = next;
      else if (a == "++" CMI_IPC_POOL_SIZE_ARG) pool = next;
    }
    if (cutoff.empty() && getenv(CMI_IPC_CUTOFF_ENV_VAR)) cutoff = getenv(CMI_IPC_CUTOFF_ENV_VAR);
    if (pool.empty() && getenv(CMI_IPC_POOL_SIZE_ENV_VAR)) pool = getenv(CMI_IPC_POOL_SIZE_ENV_VAR);
    auto optStr = [](const std::string &x) { return x.empty() ? std::string("null") : jstr(x); };
    bool active = CmiIpcEnabled();
    std::string o = std::string("{\"ipc\":") + (on && !off ? "true" : "false") + ",\"ipcmode\":" +
                    optStr(mode) + ",\"ipccutoff\":" + optStr(cutoff) + ",\"ipcpoolsize\":" + optStr(pool) +
                    ",\"ipc_active\":" + (active ? "true" : "false") + ",\"ipc_impl\":" + jstr(CmiIpcImplName()) +
                    ",\"ipc_peers\":" + std::to_string(CmiIpcNumPeers());
    if (active) o += ",\"ipc_cutoff_effective\":" + std::to_string(CmiRecommendedIpcBlockCutoff());
    return o + ",\"copy_based_rdma\":" + (CmiUseCopyBasedRDMA ? "true" : "false") + ",\"deliver\":" +
           (g_deliverUser ? "\"user\"" : "\"message\"") + ",\"source\":" +
           (g_sourceUser ? "\"user\"" : "\"message\"") +
           ",\"lci_packet_size\":" + std::to_string(COMMBENCH_LCI_PACKET_SIZE) + "}";
  }

  void writeRunRecord() {
    char host[256] = "unknown";
    gethostname(host, sizeof host);
    host[sizeof host - 1] = 0;
    if (char *dot = strchr(host, '.')) *dot = 0;
    if (machine.empty()) machine = host;

    time_t now = time(nullptr);
    struct tm lt;
    localtime_r(&now, &lt);
    char ts[40], tz[8], idts[32];
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &lt);
    strftime(tz, sizeof tz, "%z", &lt);  // +hhmm -> +hh:mm
    started = std::string(ts) + std::string(tz, 3) + ":" + std::string(tz + 3);
    strftime(idts, sizeof idts, "%Y%m%dT%H%M%S", &lt);
    if (runId.empty()) runId = machine + "-" + idts + "-" + COMMBENCH_CHARM_COMMIT;

    std::set<int> hosts;
    std::vector<int> perProc, physOfPe;
    for (int pe = 0; pe < CkNumPes(); pe++) {
      hosts.insert(CmiPhysicalNodeID(pe));
      physOfPe.push_back(CmiPhysicalNodeID(pe));
    }
    for (int n = 0; n < CkNumNodes(); n++) perProc.push_back(CkNodeSize(n));
    bool production = CMK_OPTIMIZE && !CMK_ERROR_CHECKING;

    std::string argvJ = "[";
    for (size_t i = 0; i < argvCopy.size(); i++) argvJ += (i ? "," : "") + jstr(argvCopy[i]);
    argvJ += "]";
    std::string pairsJ = "[";
    for (size_t i = 0; i < pairs.size(); i++)
      pairsJ += (i ? ",[" : "[") + std::to_string(pairs[i].first) + "," + std::to_string(pairs[i].second) + "]";
    pairsJ += "]";
    std::string ents = "[", meths = "[";
    for (int e = 0; e < E_COUNT; e++)
      if (entities[e]) ents += (ents.size() > 1 ? "," : "") + jstr(kEntityName[e]);
    for (int m = 0; m < kNumMethods; m++)
      if (methods[m]) meths += (meths.size() > 1 ? "," : "") + jstr(kMethods[m].name);
    ents += "]";
    meths += "]";

    emit(std::string("{\"type\":\"run\",\"schema\":1,\"tool\":\"commbench\",\"run_id\":") + jstr(runId) +
         ",\"machine\":" + jstr(machine) + ",\"started\":" + jstr(started) +
         ",\"charm_commit\":" + jstr(COMMBENCH_CHARM_COMMIT) +
         ",\"charm_dirty\":" + (COMMBENCH_CHARM_DIRTY ? "true" : "false") +
         ",\"reconverse_commit\":" + jstr(COMMBENCH_RECONVERSE_COMMIT) +
         ",\"lci_commit\":" + jstr(COMMBENCH_LCI_COMMIT) + ",\"build\":{\"target\":" +
         jstr(COMMBENCH_BUILD_TARGET) + ",\"production\":" + (production ? "true" : "false") +
         ",\"cmk\":" + cmkFlags() + ",\"device\":null},\"argv\":" + argvJ +
         ",\"topology\":{\"processes\":" + std::to_string(CkNumNodes()) +
         ",\"pes\":" + std::to_string(CkNumPes()) + ",\"hosts\":" + std::to_string(hosts.size()) +
         ",\"pes_per_process\":" + jarr(perProc) + ",\"physical_node_of_pe\":" + jarr(physOfPe) +
         "},\"argv_runtime\":" + runtimeArgv() + ",\"flags\":" + ipcFlags() + ",\"seed\":" + std::to_string(seed) +
         ",\"options\":{\"entities\":" + ents + ",\"methods\":" + meths + ",\"sizes\":" + jarr(sizes) +
         ",\"reps\":" + std::to_string(reps) + ",\"sample_time\":" + jnum(sampleTime) +
         ",\"validate\":" + (validate ? "true" : "false") + ",\"pairs\":" + pairsJ + "}}");
  }
};

#include "commbench.def.h"
