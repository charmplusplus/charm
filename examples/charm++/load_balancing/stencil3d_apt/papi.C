#include "papi.h"
#define MAX_ITER 300
//#define PAPI_INST
#define PERF_ENERGY

#ifdef PERF_ENERGY
#include <sys/syscall.h>
#include <linux/perf_event.h>

#define MAX_CPUS  1024
#define MAX_PACKAGES  16

#define NUM_CONFIGS 5//7
#define TRIES 5
//#define WPN_LIST (int[]){48,24,20,16,12,8,4}
#define WPN_LIST (int[]){48,24,16,12,8}

#define NUM_RAPL_DOMAINS  5

char rapl_domain_names[NUM_RAPL_DOMAINS][30]= {
  "energy-cores",
  "energy-gpu",
  "energy-pkg",
  "energy-ram",
  "energy-psys",
};
#endif

class PAPI_grp: public CBase_PAPI_grp {
  int eventset, result;
  long long values[4];
#ifdef PERF_ENERGY
  FILE *fff;
  int type;
  int config[NUM_RAPL_DOMAINS];
  char units[NUM_RAPL_DOMAINS][BUFSIZ];
  char filename[BUFSIZ];
  int fd[NUM_RAPL_DOMAINS][MAX_PACKAGES];
  double scale[NUM_RAPL_DOMAINS];
  struct perf_event_attr attr;
  long long value;
  int i,j;
  int paranoid_value;
  int total_cores=0,total_packages=0;
  int package_map[MAX_PACKAGES];
#endif

  public:
  double end_time[MAX_ITER];
  int nxt_set;
  int iter;
  int wpn;
  int config_step;
  double avg_time;
  PAPI_grp() {
    iter = 0;
    avg_time = 0.0;
    nxt_set = 0;
    wpn = CkNodeSize(CkMyNode());
    config_step = 0;
#ifdef PERF_ENERGY
    detect_packages();
#ifdef DEBUG
    printf("\nTrying perf_event interface to gather results\n\n");
#endif

    fff=fopen("/sys/bus/event_source/devices/power/type","r");
    if (fff==NULL) {
      printf("\tNo perf_event rapl support found (requires Linux 3.14)\n");
      printf("\tFalling back to raw msr support\n\n");
      CkAbort("No perf_event rapl support found (requires Linux 3.14)\n");
    }
    fscanf(fff,"%d",&type);
    fclose(fff);

    for(i=0;i<NUM_RAPL_DOMAINS;i++) {

      sprintf(filename,"/sys/bus/event_source/devices/power/events/%s",
        rapl_domain_names[i]);

      fff=fopen(filename,"r");

      if (fff!=NULL) {
        fscanf(fff,"event=%x",&config[i]);
#ifdef DEBUG
        printf("\tEvent=%s Config=%d ",rapl_domain_names[i],config[i]);
#endif
        fclose(fff);
      } else {
        continue;
      }

      sprintf(filename,"/sys/bus/event_source/devices/power/events/%s.scale",
        rapl_domain_names[i]);
      fff=fopen(filename,"r");

      if (fff!=NULL) {
        fscanf(fff,"%lf",&scale[i]);
#ifdef DEBUG
        printf("scale=%g ",scale[i]);
#endif
        fclose(fff);
      }

      sprintf(filename,"/sys/bus/event_source/devices/power/events/%s.unit",
        rapl_domain_names[i]);
      fff=fopen(filename,"r");

      if (fff!=NULL) {
        fscanf(fff,"%s",units[i]);
#ifdef DEBUG
        printf("units=%s ",units[i]);
#endif
        fclose(fff);
      }

//      printf("\n");
    }
#endif

#ifdef PAPI_INST
    eventset=PAPI_NULL;
    result = PAPI_register_thread();
    if ( result != PAPI_OK ) {
      printf("Error register thread!\n");
    }

    result=PAPI_create_eventset(&eventset);
    if (result!=PAPI_OK) {
      printf("Error PAPI create eventset: %s\n",PAPI_strerror(result));
    }

    result=PAPI_add_named_event(eventset,"PAPI_TOT_INS");
    if (result!=PAPI_OK) {
      printf("Error PAPI add_event %s\n",
        PAPI_strerror(result));
    }

    result=PAPI_add_named_event(eventset,"PAPI_TOT_CYC");
    if (result!=PAPI_OK) {
      printf("Error PAPI add_event %s\n",
        PAPI_strerror(result));
    }

    result=PAPI_add_named_event(eventset,"PAPI_RES_STL");//PAPI_L3_TCA");
    if (result!=PAPI_OK) {
      printf("Error PAPI add_event %s\n",
        PAPI_strerror(result));
    }

    result=PAPI_add_named_event(eventset,"PAPI_L3_TCM");
    if (result!=PAPI_OK) {
      printf("Error PAPI add_event %s\n",
        PAPI_strerror(result));
    }
#endif
  }
  void start() {
    PAPI_start(eventset);
  }
  void stop(double k_time, long long* inst, long long* cycles, long long* stalled_cycles) {
    PAPI_stop(eventset,values);
    *inst = values[0];
    *cycles = values[1];
    *stalled_cycles = values[2];
    //printf("Total instructions %lld, Total cycles %lld, Stalled cycles %lld misses %lld Kernel time= %lf s\n",values[0],values[1], values[2], values[3], k_time);
  }

#ifdef PERF_ENERGY
  int check_paranoid(void) {

    int paranoid_value;
    FILE *fff;

    fff=fopen("/proc/sys/kernel/perf_event_paranoid","r");
    if (fff==NULL) {
      fprintf(stderr,"Error! could not open /proc/sys/kernel/perf_event_paranoid %s\n",
        strerror(errno));

      /* We can't return a negative value as that implies no paranoia */
      return 500;
    }

    fscanf(fff,"%d",&paranoid_value);
    fclose(fff);

    return paranoid_value;

  }
#endif

#ifdef PERF_ENERGY
  int detect_packages(void) {

    char filename[BUFSIZ];
    FILE *fff;
    int package;
    int i;

    for(i=0;i<MAX_PACKAGES;i++) package_map[i]=-1;

    printf("\t");
    for(i=0;i<MAX_CPUS;i++) {
      sprintf(filename,"/sys/devices/system/cpu/cpu%d/topology/physical_package_id",i);
      fff=fopen(filename,"r");
      if (fff==NULL) break;
      fscanf(fff,"%d",&package);
#ifdef DEBUG
      printf("%d (%d)",i,package);
      if (i%8==7) printf("\n\t"); else printf(", ");
#endif
      fclose(fff);

      if (package_map[package]==-1) {
        total_packages++;
        package_map[package]=i;
      }

    }

//    printf("\n");

    total_cores=i;
#ifdef DEBUG
    printf("\tDetected %d cores in %d packages\n\n",
      total_cores,total_packages);
#endif

    return 0;
  }

#endif

#ifdef PERF_ENERGY
  int perf_event_open(struct perf_event_attr *hw_event_uptr,
                    pid_t pid, int cpu, int group_fd, unsigned long flags) {

        return syscall(__NR_perf_event_open,hw_event_uptr, pid, cpu,
                        group_fd, flags);
  }
#endif
std::unordered_map<int, double> ppn_time;
double prev_time;
void report_time(int wpn, double time) {
  ppn_time[wpn] = time;
  printf("\nEntered time for wpn %d = %lf (%lf)s", wpn, ppn_time[wpn], time);
}

int go_lower(int config_idx, int skip_over) {
  int val = -1;
  for(int i=config_idx+1;i<NUM_CONFIGS;i++) {
    if(ppn_time.find(WPN_LIST[i]) == ppn_time.end()) {
      if(!skip_over) return i;
      val = i;
      skip_over--;
    }
  }
  return val;
}

int go_higher(int config_idx, int skip_over) {
  int val = -1;
  for(int i=config_idx-1;i>=0;i--) {
    if(ppn_time.find(WPN_LIST[i]) == ppn_time.end()) {
      if(!skip_over) return i;
      val = i;
      skip_over--;
    }
  }
  return val;
}

int get_best_ppn() {
  double min = 1000.0;
  std::vector<int> min_wpn;
  for (auto it : ppn_time) {
    double diff = abs(min-it.second);
    double add = false;
    if(diff/min < 0.1) add = true;
    if(add)
      min_wpn.push_back(it.first);
    else if(it.second < min) {
      min = it.second;
      min_wpn.clear();
      min_wpn.push_back(it.first);
    }
  }
  auto min_value = *std::min_element(min_wpn.begin(), min_wpn.end());
  return (int)min_value;
}
double end_etime;
double start_etime;
#ifdef PERF_ENERGY
  void start_energy(double time) {
      start_etime = time;
    for(j=0;j<total_packages;j++) {

      for(i=0;i<NUM_RAPL_DOMAINS;i++) {

        fd[i][j]=-1;

        memset(&attr,0x0,sizeof(attr));
        attr.type=type;
        attr.config=config[i];
        if (config[i]==0) continue;

        fd[i][j]=perf_event_open(&attr,-1, package_map[j],-1,0);
        if (fd[i][j]<0) {
          if (errno==EACCES) {
            paranoid_value=check_paranoid();
            if (paranoid_value>0) {
              printf("\t/proc/sys/kernel/perf_event_paranoid is %d\n",paranoid_value);
              printf("\tThe value must be 0 or lower to read system-wide RAPL values\n");
            }

            printf("\tPermission denied; run as root or adjust paranoid value\n\n");
            CkAbort("\tPermission denied; run as root or adjust paranoid value\n\n");
          }
          else {
            printf("\terror opening core %d config %d: %s\n\n",
              package_map[j], config[i], strerror(errno));
            CkAbort("\terror opening core");
          }
        }
      }
    }
  }
#endif

#ifdef PERF_ENERGY
  void stop_energy(double time) {
    end_etime = time;
    for(j=0;j<total_packages;j++) {
    printf("\tPackage %d:\n",j);
    for(i=0;i<NUM_RAPL_DOMAINS;i++) {
        if (fd[i][j]!=-1) {
          read(fd[i][j],&value,8);
          close(fd[i][j]);

          printf("\t\t%s Energy Consumed: %lf %s, Power = %lf W\n",
            rapl_domain_names[i],
            (double)value*scale[i],
            units[i], (double)value*scale[i]/(end_etime-start_etime));
        }
      }
    }
    printf("\n");
  }
#endif
};

