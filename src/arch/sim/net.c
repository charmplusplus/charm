#define UNIFORM_BROADCAST  0
#define SPANTREE_BROADCAST 1
#define UNIFORM_L          0 
#define F1_L               1



static int       net_load_l;
static int       net_load_s;
static long      stream_f1;


static int broadcast_flag;
static int latency_flag;
static REL_TIME latency_f1();


static comm_init()
{
    float expdev();

    net_load_l = 0;
    net_load_s = 0;
    broadcast_flag = num.broadcast_flag; 
    latency_flag   = num.latency_flag;
    stream_f1 = (long) num.latency_argv2;
    ran1(&stream_f1);
}






static REL_TIME latency(cpu_pno,msg)
int cpu_pno;
MSG *msg;
{
    REL_TIME delay;

    if (cpu_pno == msg->dest) return 1;

    switch (latency_flag) {
        case UNIFORM_L : delay =  COST_F(num.cost.net,msg->length); break;
        case F1_L      : delay = latency_f1(msg);          break;
        default : error_msg("invalid latency function selection",100);
    } 
     
    return delay;

}





static REL_TIME broadcast_effect(source,dest,delay)
int      source; /* source CPU */
int      dest;   /* dest   CPU */
REL_TIME delay;
{
    int          distance;
    unsigned int temp;
    REL_TIME     new_delay;

    switch (broadcast_flag) {
      case UNIFORM_BROADCAST : return delay;
      case SPANTREE_BROADCAST:
           distance  = dest-source;
           if (distance<0) distance += num.pe;
           temp = distance+1;
           for(new_delay=delay; temp>>1; new_delay+=delay) temp>>=1;
           return new_delay;
           break;
      default : error_msg("invalid broadcast function selection",100);
    } 
}





static network_ok_l(s,d,l)
int s,d,l;
{
   return (pe_table[d].msg.num_of_elem < threshold.number.rcp);
}




static network_ok_s(s,d,l)
int s,d,l;
{
   /* return (pe_table[d].msg.size+l < threshold.size.rcp); */
   return (pe_table[d].msg.size < threshold.size.rcp);
}


static network_ok_l_g(s,d,l)
int s,d,l;
{
   return (net_load_l < threshold.number.net);
}


static network_ok_s_g(s,d,l)
int s,d,l;
{
   /* return (net_load_s+l < threshold.size.net); */
   return (net_load_s < threshold.size.net);
}



static cpu_accepts_l(pno,l)
int pno;
int l;
{
   return (pe_table[pno].msg.num_of_elem < threshold.number.cpu);
}




static cpu_accepts_s(pno,l)
int pno;
int l;
{
   return (pe_table[pno].msg.size+l < threshold.size.cpu);
}




static scp_accepts_l(pno,l)
int pno;
int l;
{
   return (pe_table[pno].msg.num_of_elem < threshold.number.scp);
}



static scp_accepts_s(pno,l)
int pno;
int l;
{
   return (pe_table[pno].msg.size+l < threshold.size.scp);
}



static always_accept() { return 1;}


static increase_net_load(s)
int s;
{
    net_load_l++;
    net_load_s += s;
}

static decrease_net_load(s)
int s;
{
    net_load_l--;
    net_load_s -= s;
}





static REL_TIME latency_f1(msg)
MSG *msg;
{

     float expdev();
     REL_TIME inc;
     REL_TIME d;
  
     float temp;
 
     /* num.latency_argv1 is the mean of exp distirbution */
     /* a message's fixed latency is increased randomly */
  
     temp = expdev(&stream_f1);
      
     inc = (REL_TIME) ( num.latency_argv1*temp); 
     
     d = num.cost.net.alpha + inc + num.cost.net.beta*msg->length;

     return d; 
}
