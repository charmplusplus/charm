#ifndef MACHINE_H
#define MACHINE_H

#ifndef NULL
#define NULL 0
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif


typedef double SIM_TIME;
typedef double REL_TIME;


#define time_add(time1,time2)    *time1 += time2
#define time_zero(time1)         *time1 = 0.0
#define time_assign(time1,time2) *time1 = *time2
#define greater_time(time1,time2) ((time1>time2) ? TRUE : FALSE)
#define less_time(time1,time2)    ((time1<time2) ? TRUE : FALSE)
#define time_assign_max(time1)    *time1=max_time_const

#define MESSAGE_QUEUE       0
#define MESSAGE             0

#define ADDR_Q_MSG(pno)        &(pe_table[pno].msg)
#define ADDR_Q_MSG_FRONT(pno)  &(pe_table[pno].msg.front)

#define Q_MSG(pno)             pe_table[pno].msg
#define Q_MSG_FRONT(pno)       pe_table[pno].msg.front



#define WAIT_ON_COND     0x2
#define IDLE_STATE       0x1
#define NO_SPLIT         0x0
#define SPLIT_BEGIN_LOOP 0x10
#define SPLIT_END_LOOP   0x20
#define SPLIT_SEND       0x30
#define CPU_TYPE         0x100
#define SCP_TYPE         0x200
#define RCP_TYPE         0x400
#define SET_IDLE(pno)           pe_table[pno].flag |= IDLE_STATE
#define RESET_IDLE(pno)         pe_table[pno].flag &= ~IDLE_STATE
#define IS_IDLE(pno)            (pe_table[pno].flag & IDLE_STATE)
#define SET_WAIT_ON_COND(pno)   pe_table[pno].flag |= WAIT_ON_COND
#define RESET_WAIT_ON_COND(pno) pe_table[pno].flag &= ~WAIT_ON_COND
#define IS_WAITING_ON_COND(pno) (pe_table[pno].flag & WAIT_ON_COND)
#define SET_EVENT(pno,etype)    pe_table[pno].flag = (pe_table[pno].flag & ~0x30)  | etype
#define EVENT_TYPE(pno)         (pe_table[pno].flag & 0x30) 
#define SET_TYPE(pno,t)         pe_table[pno].flag |= t
#define IS_CPU(p)               (pe_table[p].flag & CPU_TYPE) 
#define IS_SCP(p)               (pe_table[p].flag & SCP_TYPE) 
#define IS_RCP(p)               (pe_table[p].flag & RCP_TYPE) 
#define CPU_TO_RCP(pno)         (pno+num.pe+num.pe)
#define CPU_TO_SCP(pno)         (pno+num.pe)
#define RCP_TO_CPU(pno)         (pno-(num.pe+num.pe))
#define SCP_TO_CPU(pno)         (pno-num.pe)


#define ERR_MSG1 "out of memory"
#define ERR_MSG2 "invalid queue type"
#define ERR_MSG6 "can't open file"
#define ERR_MSG7 "possible queueing error"
#define ERR_MSG10 "timing error"
#define ERR_MSG14 "error"
#define ERR_MSG15 "usage: sim1 [-v] [-s seed] [-a] -i p d c t -o o"


#define SMALLOC(p,e)  malloc(p)
#define SFREE(p)      free(p)


typedef struct str_msg {
    SIM_TIME       arrival_time;
    int            length;
    int            broadcast;
    int            dest;
    void           *envelope;
    struct str_msg *next;
} MSG;



typedef struct {
    MSG   *front;
    MSG   *rear;
    int   size;
    int   num_of_elem;
} PE_MSG_QUEUE;

    
typedef struct  {
    SIM_TIME     etime; 
    SIM_TIME     lower_bound; 
    unsigned int index;
    int          flag; 
    PE_MSG_QUEUE msg;
    MSG          *rcp_buffer;
    int          wait_on_ix; 
} PROCESSOR;


typedef struct {
    REL_TIME    alpha;
    REL_TIME    beta; 
} COST;



typedef struct {
    int   pe;
    int   broadcast_flag;
    int   latency_flag;
    double latency_argv1;
    long   latency_argv2;
    double processor_scale ;   
    double periodic_interval; 
  
    struct {
       COST      cpu_recv;   
       COST      cpu_send;       
       COST      rcp;
       COST      scp;
       COST      net;
    } cost;
 
} PARAMETERS;




typedef struct {
   int cpu_limit_type;
   int scp_limit_type;
   int net_limit_type;
   int network_wide;
   struct {
     int cpu;
     int rcp;
     int scp;
     int net;
   } size, number;

} THRESHOLD;



/* PERIODIC_INTERVAL is the amount of time in us in between times which
 * the processor automatically goes back to check up on each processor
 */
#define MIN_INC           0.000001

SIM_TIME max_time_const;

#define COST_F(cost,l) (REL_TIME)((REL_TIME) cost.alpha + (REL_TIME) (cost.beta * l ))
#define COST_F2(cost) ((REL_TIME)cost)


#endif
