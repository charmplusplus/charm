#define TRACE_ELAPSED_TIME 0
#define TRACE_SENDCP_TIME 0
#define TRACE_RECVCP_TIME 0

static PROCESSOR    *pe_table;
static SIM_TIME     gclock;
static PARAMETERS   num;
static THRESHOLD threshold;
static int          (*cpu_accepts)();
static int          (*scp_accepts)();
static int          (*network_accepts)();
static int          seed;
static int          scp_wait_on_list;




/***********************************************************************/

static int sim_read_parameters(filename,num,threshold)
char *filename;
PARAMETERS *num;
THRESHOLD  *threshold;
{
#define MAXLINE 100
    char line[100];
    char key[80], gunk[80];
    FILE *fp;

    if ( (fp = fopen(filename,"r")) == NULL) 
	return 0 ;

    while (fgets(line,MAXLINE-1,fp) != 0) {

	sscanf(line,"%s ",key) ;
	if(key[0]=='#') continue;

        if (!strcmp(key,"cpu_nolimit")) continue;
        if (!strcmp(key,"scp_nolimit")) continue;
        if (!strcmp(key,"rcp_net_nolimit")) continue;

	if(!strcmp(key,"cpu_recv_cost"))
	  sscanf(line,"%s %le %le",gunk,&num->cost.cpu_recv.alpha,
		   &num->cost.cpu_recv.beta) ;
                                         
	else if (!strcmp(key,"cpu_send_cost"))
	    sscanf(line,"%s %le %le",gunk,&num->cost.cpu_send.alpha,
		   &num->cost.cpu_send.beta) ;

	else if (!strcmp(key,"rcp_cost"))
	    sscanf(line,"%s %le %le",gunk,&num->cost.rcp.alpha,
		   &num->cost.rcp.beta);

	else if (!strcmp(key,"scp_cost"))
	    sscanf(line,"%s %le %le",gunk, &num->cost.scp.alpha,
		   &num->cost.scp.beta);

	else if (!strcmp(key,"net_cost"))
	    sscanf(line,"%s %le %le",gunk,&num->cost.net.alpha,
		   &num->cost.net.beta);
	
        else if (!strcmp(key,"cpu_queue_threshold_number"))
    	    { sscanf(line,"%s %d",gunk,&threshold->number.cpu);
              threshold->cpu_limit_type = 1;
            }

        else if (!strcmp(key,"cpu_queue_threshold_size"))
            { sscanf(line,"%s %d",gunk,&threshold->size.cpu);
              threshold->cpu_limit_type = 2;
            }

        else if (!strcmp(key,"scp_queue_threshold_number"))
            { sscanf(line,"%s %d",gunk,&threshold->number.scp);
              threshold->scp_limit_type = 1;
            }

        else if (!strcmp(key,"scp_queue_threshold_size"))
            { sscanf(line,"%s %d",gunk,&threshold->size.scp);
              threshold->scp_limit_type = 2;
            }

        else if (!strcmp(key,"rcp_queue_threshold_number"))
            { sscanf(line,"%s %d",gunk,&threshold->number.rcp);
              threshold->net_limit_type = 1;
              threshold->network_wide = 0;
            }

        else if (!strcmp(key,"rcp_queue_threshold_size"))
            { sscanf(line,"%s %d",gunk,&threshold->size.rcp);
              threshold->net_limit_type = 2;
              threshold->network_wide = 0; 
            }

        else if (!strcmp(key,"net_queue_threshold_number"))
            { sscanf(line,"%s %d",gunk,&threshold->number.net);
              threshold->net_limit_type = 1;
              threshold->network_wide = 1;
            }

        else if (!strcmp(key,"net_queue_threshold_size"))
            { sscanf(line,"%s %d",gunk,&threshold->size.net);
              threshold->net_limit_type = 2;
              threshold->network_wide = 1;
            }


	else if (!strcmp(key,"latency_fixed"))
            num->latency_flag = 0;

        else if (!strcmp(key,"latency_rand"))
            {
            num->latency_flag = 1;
            sscanf(line,"%s %le %d",gunk,&num->latency_argv1,
                   &num->latency_argv2);
            }

	else if (!strcmp(key,"processor_scale")) {
	    sscanf(line,"%s %le",gunk,&num->processor_scale) ;
	    /* This parameter is assumed to be a "times faster" thing.
	     * Get inverse so we can use to multiply elapsed_time */  
	    num->processor_scale = 1.0/num->processor_scale ;
	}

        else if (!strcmp(key,"periodic_interval"))
            sscanf(line,"%s %le",gunk,&num->periodic_interval);

	else {
	    CmiPrintf("Unknown key field '%s' in %s.  Skipping.\n",
		     key,filename) ;
	    /* fscanf(fp,"\n",gunk) ; */
	}
    }
    fclose(fp);
    return 1 ;
}


/***********************************************************************/

CsvExtern(int, CsdStopCount);

static void simulate()
{
    int pno;
    
    while ( (pno = select_processor())  )  {
        _Cmi_mype = pno-1;
        if (IS_CPU(pno)) {

            if (  CpvAccess(CsdStopFlag) == 0)
               {
	           cpu_event(pno);
                   return;
               }
            else {
	       CsvAccess(CsdStopCount)--;
               become_idle(pno);
	    }
        }
        else if (IS_RCP(pno))
	    switch ( EVENT_TYPE(pno) ) {
	    case NO_SPLIT         :  recv_cp_event(pno);    break;
	    case SPLIT_SEND       :  recv_cp_deposit(pno);  break;
   	    default               :  error_msg(ERR_MSG14,1100) ;
	    }
        else  /* (IS_SCP(pno)) */
            send_cp_event(pno);
    }
}

/***********************************************************************/

CpvExtern(int,    CcdCheckNum) ;

static cpu_event(pno)
int pno;
{
    int             rcp_pno;
    REL_TIME        cpu_recv_cost;
    REL_TIME        elapsed_time ;
    double          temp_time; 
    int             *msg;
    int             flag;

#ifdef DEBUG
    if (IS_IDLE(pno)) error_msg(ERR_MSG14,1000);
#endif

    advance_clock(pno);
    flag = 0;
    Csi_global_time = gclock;
    Csi_start_time  = CsiTimer();

    if ( !CdsFifo_Empty(CpvAccess(CmiLocalQueue)) )
      {
         
         msg = CdsFifo_Dequeue(CpvAccess(CmiLocalQueue)); 
         CmiHandleMessage(msg);

         elapsed_time = (REL_TIME) (CsiTimer() - Csi_start_time); 
         elapsed_time += (REL_TIME) MIN_INC;
         update_lower_bound(pno,elapsed_time);
         update_etime(pno); 
         return;
      }


    if ( !fifo_empty(ADDR_Q_MSG(pno)) ) {
        MSG *sim_msg;
        void *env;
        
        /* get the message from the network */
        sim_msg = (MSG *) fifo_dequeue(ADDR_Q_MSG(pno));
        msg = sim_msg->envelope;
        sim_msg->envelope = NULL;

        CmiHandleMessage(msg);

        cpu_recv_cost = (REL_TIME) (CsiTimer() - Csi_start_time);
        cpu_recv_cost += (REL_TIME) MIN_INC;
        update_lower_bound(pno,cpu_recv_cost);
        rcp_pno = CPU_TO_RCP(pno);

        if (IS_WAITING_ON_COND(rcp_pno) )
            release_waiting_local(rcp_pno,pe_table[pno].lower_bound);
        update_etime(pno);
        return;
      }
    

    if ( !CqsEmpty(CpvAccess(CsdSchedQueue)) ) 
      {
   
        CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
        (CmiGetHandlerFunction(msg))(msg);
 
        elapsed_time = (REL_TIME) (CsiTimer() - Csi_start_time);
        elapsed_time *= (REL_TIME) num.processor_scale;
        /* Make sure elapsed_time isn't zero by adding one usec */
        elapsed_time += (REL_TIME) MIN_INC;
#if TRACE_ELAPSED_TIME
        CmiPrintf("cpu_event: elapsed_time %d\n",elapsed_time) ;
#endif
        flag = 1;
      }
    else 
      {
	CcdRaiseCondition(CcdPROCESSOR_STILL_IDLE);
        elapsed_time = (REL_TIME) (CsiTimer() - Csi_start_time);
      }

    temp_time = CsiTimer(); 

    CsdPeriodic();

    if (flag == 0)  elapsed_time += num.periodic_interval;

    elapsed_time += (REL_TIME) (CsiTimer() - temp_time);
    elapsed_time += (REL_TIME) MIN_INC;
    update_lower_bound(pno,elapsed_time);
    update_etime(pno);
}




/***********************************************************************/

static recv_cp_event(pno)
int pno;
{
    SIM_TIME msg_arr_time;
    REL_TIME recv_cost;
    MSG      *msg;
    SIM_TIME next_msg_arrival();

    advance_clock(pno);
    if ( Q_MSG_FRONT(pno) == NULL ) {
        become_idle(pno);
    }
    else { 
        /* modify this call, make it return the result by parameter */
        msg_arr_time = next_msg_arrival(Q_MSG_FRONT(pno));
	
        if ( !greater_time(msg_arr_time,gclock) ) {
	    release_waiting_network(pe_table[pno].lower_bound);
	    msg = remove_front(ADDR_Q_MSG_FRONT(pno));
	    pe_table[pno].msg.size -= msg->length;
	    pe_table[pno].msg.num_of_elem--;
	    
	    decrease_net_load(msg->length);
	    
	    /* save the message */ 
	    pe_table[pno].rcp_buffer = msg; 
	    /* post the event to deposit the message after recv_cost time */
	    
	    SET_EVENT(pno,SPLIT_SEND);
	    recv_cost = COST_F(num.cost.rcp,msg->length);
#if TRACE_RECVCP_TIME
	    CmiPrintf("recv_cp_event: recv_cost %d\n",recv_cost) ;
#endif
	    update_lower_bound(pno,recv_cost);
	    update_etime(pno);
	}
        else {
	    assign_etime(pno,msg_arr_time);

	}
    }
}

/***********************************************************************/

static recv_cp_deposit(pno)
int pno;
{
    int cpu;
    MSG *msg;
    
    advance_clock(pno);
    cpu = RCP_TO_CPU(pno);
    msg = pe_table[pno].rcp_buffer;
    if ( (*cpu_accepts)(cpu,msg->length) ) { 
        pe_table[pno].rcp_buffer = NULL;
        fifo_enqueue(ADDR_Q_MSG(cpu),msg);
        release_waiting_network(pe_table[pno].lower_bound);

	if (less_time(gclock, pe_table[cpu].lower_bound)) {
	    if (less_time(pe_table[cpu].lower_bound,pe_table[cpu].etime)) 
		update_etime(cpu) ;
	}
	else
	    assign_etime(cpu,gclock) ;

        if ( !IS_WAITING_ON_COND(cpu) && IS_IDLE(cpu)) make_active(cpu);
        SET_EVENT(pno,NO_SPLIT);
    }
    else
	wait_on_local(pno);
}

/***********************************************************************/

static send_cp_event(pno)
int pno;
{
    int      dest,cpu;
    REL_TIME delay,send_cost;
    MSG      *msg;
    MSG      *fifo_dequeue();

    advance_clock(pno);
    
    if (fifo_empty(ADDR_Q_MSG(pno)) ){
	become_idle(pno);
    }
    else {
	int l;
	MSG *temp_msg;
	temp_msg = Q_MSG_FRONT(pno);
	dest = CPU_TO_RCP(temp_msg->dest);
	l    = temp_msg->length;
	if ((*network_accepts)(pno,dest,l)) {
	    msg = fifo_dequeue(ADDR_Q_MSG(pno));
	    delay = latency(SCP_TO_CPU(pno),msg); 
	    if (msg->broadcast) 
		actual_broadcast(pno,msg,delay);
	    else
		actual_send(dest,msg,delay);
	    send_cost = COST_F(num.cost.scp,msg->length);
#if TRACE_SENDCP_TIME
	    CmiPrintf("send_cp_event: send_cost %d\n",send_cost) ;
#endif
	    update_lower_bound(pno,send_cost);
	    update_etime(pno);
	    cpu = SCP_TO_CPU(pno);
	    if (IS_WAITING_ON_COND(cpu) )
		release_waiting_local(cpu,pe_table[pno].lower_bound);
	}
	else {
	    wait_on_network(pno);
	}
    }
    
}

/***********************************************************************/

static actual_broadcast(source,msg,delay)
MSG      *msg;
REL_TIME delay;
int      source;
{
    int i,j,cpu;
    MSG *msg2;
    void *usrMsg ;
    
    char *tempmsg;

    void * env, *tempenv;

    cpu = SCP_TO_CPU(source);
    env = msg->envelope ;
    
    for(i=1; i<num.pe; i++) {
	
	msg2 = (MSG *) SMALLOC(sizeof(MSG),10);
	msg2->length     = msg->length;
	msg2->broadcast  = msg->broadcast;
	msg2->dest       = i ;
        msg2->arrival_time = msg->arrival_time;

        
        tempmsg = (char *)CmiAlloc(msg->length);
	memcpy(tempmsg, (char*)env, msg->length) ;
	tempenv = (void *)tempmsg ;
	msg2->envelope = tempenv ;
	msg2->next       = NULL;
	
	j = CPU_TO_RCP(i);
	actual_send(j,msg2,broadcast_effect(cpu,i,delay));
    }
    j = CPU_TO_RCP(num.pe);
    actual_send(j,msg,broadcast_effect(cpu,num.pe,delay));
}

/***********************************************************************/

static actual_send(d,msg,delay)
int      d;
MSG      *msg;
REL_TIME delay;
{
    SIM_TIME arrival_time;
    time_assign(&arrival_time,&gclock);
    time_add(&arrival_time,delay);
    time_add(&arrival_time,msg->arrival_time);
    time_assign(&msg->arrival_time,&arrival_time); 
    insert(ADDR_Q_MSG_FRONT(d),ge,msg);
    
    pe_table[d].msg.size += msg->length;
    pe_table[d].msg.num_of_elem++;
    
    increase_net_load(msg->length);
    
    if (   is_first_element(Q_MSG_FRONT(d),msg) && 
	(EVENT_TYPE(d) == NO_SPLIT)          &&
	!IS_WAITING_ON_COND(d)                   )  {
	if (  IS_IDLE(d) ) {
	    time_assign(&pe_table[d].etime,&arrival_time);
	    RESET_IDLE(d);
	} 
	else if ( less_time(pe_table[d].lower_bound,arrival_time) &&
                 less_time(pe_table[d].etime,arrival_time) ) {
	    time_assign(&pe_table[d].etime,&arrival_time);
	}
	relocate_event(pe_table[d].index);
    }
}

/***********************************************************************/

static wait_on_network(pno)
int pno;
{
    int temp;
    /* valid for sender communication processor */
    
    temp = scp_wait_on_list;
    scp_wait_on_list = pno; 
    pe_table[pno].wait_on_ix  = temp;
    
    SET_IDLE(pno);
    SET_WAIT_ON_COND(pno);
    become_idle(pno);
}

/***********************************************************************/

static wait_on_local(pno)
int pno;
{
    /* only cpu or rcp can call this */
    SET_WAIT_ON_COND(pno);
    SET_IDLE(pno);
    become_idle(pno);
}

/***********************************************************************/

static release_waiting_network(t)
SIM_TIME t;
{
    int temp;
    
    while (scp_wait_on_list) {
	RESET_WAIT_ON_COND(scp_wait_on_list);
	RESET_IDLE(scp_wait_on_list);
	assign_etime(scp_wait_on_list,t);
	temp = scp_wait_on_list;
	scp_wait_on_list = pe_table[scp_wait_on_list].wait_on_ix;   
	pe_table[temp].wait_on_ix = 0;
    }
}

/***********************************************************************/

static release_waiting_local(pno,t)
int  pno;
SIM_TIME t;
{
    /* only cpu or src can call this function */
    RESET_WAIT_ON_COND(pno);
    RESET_IDLE(pno);
    assign_etime(pno,t);
}


/* ****************************************************************** */
/* upgrade the event time to the lower bound                          */
/* ****************************************************************** */
static update_etime(pno)
int pno;
{
    if (less_time(pe_table[pno].etime,pe_table[pno].lower_bound))
    {
	time_assign(&pe_table[pno].etime,&pe_table[pno].lower_bound);
	relocate_event(pe_table[pno].index);
    }
}

/***********************************************************************/

static assign_etime(pno,t)
int  pno;
SIM_TIME t;
{
    time_assign(&pe_table[pno].etime,&t);
    relocate_event(pe_table[pno].index);
}


/* ****************************************************************** */
/* increase the lower bound                                           */
/* ****************************************************************** */
static update_lower_bound(pno,elapsed_time)
int pno;
REL_TIME elapsed_time;
{
    time_add(&pe_table[pno].lower_bound,elapsed_time);
}

/* ****************************************************************** */
/* advance the global time                                            */
/* ****************************************************************** */
static advance_clock(pno)
int pno;
{ 
    if (greater_time(gclock,pe_table[pno].etime) )
	error_msg(ERR_MSG10,100);
    time_assign(&gclock,&pe_table[pno].etime);
    time_assign(&pe_table[pno].lower_bound,&gclock);
}

/* ****************************************************************** */
/* select the processor which has the most recent event               */
/* ****************************************************************** */
static select_processor()
{
    int pno;
    
    pno = next_event();
    if (!less_time(pe_table[pno].etime,max_time_const) )
	pno = 0;
    return pno;
}

/***********************************************************************/

static sim_send_message(pno,env,length,broadcast,destPE)
int         pno;       /* source processor */
void        *env;
int         length;
int         broadcast; /* TRUE if broadcast */
int         destPE;
{
    int scp;
    REL_TIME elapsed_time;
    MSG *msg;

    scp = CPU_TO_SCP(pno+1); 
    if ( (*scp_accepts)(scp,length) ) {
	msg = (MSG *) SMALLOC(sizeof(MSG),2);
	msg->length    = length;
	msg->broadcast = broadcast; 
	msg->dest      = destPE+1;
	msg->envelope  = env;
	msg->next      = NULL;
        /* message must be sended after elapsed_time + gclock */
        /* save elpased time in the message, actual send will add */
        /* gclock and latency                                     */
        elapsed_time = (REL_TIME) (CsiTimer() - Csi_start_time);
        msg->arrival_time = elapsed_time;
	fifo_enqueue(ADDR_Q_MSG(scp),msg);
	if ( !IS_WAITING_ON_COND(scp) && IS_IDLE(scp) ){
	    make_active(scp);
	}
    }
    else {
	/* communication processor buffers are full, try again */
	CmiPrintf("Error:Communication buffer is full: blocking : not implemented\n");
        exit(1);
	/* This needs to be implemented */
    }
}

/***********************************************************************/

static become_idle(pno)
int pno;
{
    SIM_TIME maxtime;
    
    SET_IDLE(pno); 
    time_assign_max(&pe_table[pno].etime);
    relocate_event(pe_table[pno].index);
}


/***********************************************************************/

static make_active(pno)
int pno;
{
    RESET_IDLE(pno);
    time_assign(&pe_table[pno].etime,&gclock);
    if (less_time(gclock,pe_table[pno].lower_bound) ) error_msg(ERR_MSG14,1020);
    relocate_event(pe_table[pno].index); 
}








/*  *************************************************************** */
/*  initialize the data structure                                   */
/*  *************************************************************** */
static int sim_initialize(paramFile,numpe)
char *paramFile;
int numpe;
{
    int i,j,k;
#ifdef DEBUG
    malloc_debug(2);
#endif
    /* srand(seed); */
    CrnSrand(seed);
    
    init_max_time_const();
    time_zero(&gclock);
    
    num.pe = numpe;

    /* assign the default values for various simulator parameters */

    num.cost.cpu_recv.alpha  = 0.0;
    num.cost.cpu_recv.beta   = 0.0;
    num.cost.cpu_send.alpha  = 0.0;
    num.cost.cpu_send.beta   = 0.0;
    num.cost.rcp.alpha       = 0.0;
    num.cost.rcp.beta        = 0.0;
    num.cost.scp.alpha       = 0.0;
    num.cost.scp.beta        = 0.0;
    num.cost.net.alpha       = 0.0;
    num.cost.net.beta        = 0.0;
    num.broadcast_flag       = 0;
    num.latency_flag         = 0;
    num.latency_argv1        = 0.0;
    num.latency_argv2        = 0.0; 
    num.processor_scale      = 1.0;
    num.periodic_interval    = 0.1;

    threshold.number.cpu = 0;
    threshold.size.cpu   = 0;
    threshold.number.rcp = 0;
    threshold.size.rcp   = 0;
    threshold.number.scp = 0;
    threshold.size.scp   = 0;
    threshold.number.net = 0;
    threshold.size.net   = 0;

    threshold.cpu_limit_type = 0;  
    threshold.scp_limit_type = 0;
    threshold.net_limit_type = 0;
    threshold.network_wide   = 0;

    
    if ( (sim_read_parameters(paramFile, &num,&threshold) == 0) ) 
    {
         printf("Warning: cannot find the parameter file: %s\n",paramFile);
         printf("         continuing with default simulator parameters\n");
    }
    
    pe_table = (PROCESSOR *) SMALLOC(sizeof(PROCESSOR)*(3*num.pe+1),15);
    create_event_heap(num.pe);
    
    for(i=0; i<=3*num.pe; i++) {
        pe_table[i].flag          = 0;
        pe_table[i].msg.front     = NULL;
        pe_table[i].msg.rear      = NULL;
        pe_table[i].msg.size      = 0;
        pe_table[i].msg.num_of_elem = 0;
        pe_table[i].rcp_buffer    = NULL;
        pe_table[i].wait_on_ix    = 0;
        time_zero(&pe_table[i].lower_bound);
        if ( i>0 && i <= num.pe )                    SET_TYPE(i,CPU_TYPE);
        else if ( i >num.pe && i <= (num.pe+num.pe)) SET_TYPE(i,SCP_TYPE);
        else if ( i >(num.pe+num.pe) )               SET_TYPE(i,RCP_TYPE);
    }
  
    cpu_accepts = always_accept;
    scp_accepts = always_accept;
    network_accepts = always_accept;

 
    if (threshold.cpu_limit_type != 0) 
    {
       cpu_accepts = 
         (threshold.cpu_limit_type == 1) ? cpu_accepts_l : cpu_accepts_s;
    }
  
    if (threshold.scp_limit_type != 0) { 
       scp_accepts = 
         (threshold.scp_limit_type == 1) ? scp_accepts_l : scp_accepts_s;
    }

    if (threshold.net_limit_type != 0)
    { 
        if (threshold.network_wide == 1) 
	    network_accepts = 
	      (threshold.net_limit_type == 1) ? network_ok_l_g:network_ok_s_g;
        else
  	    network_accepts = 
	      (threshold.net_limit_type == 1) ? network_ok_l : network_ok_s;
    }

 
    comm_init();

    return 1;
}


static init_max_time_const()
{
   max_time_const = 1e+20;
}


/*  *************************************************************** */
/*  print  error message                                            */ 
/*  *************************************************************** */
static int error_msg(err_msg,err_no)
int  *err_no;
char *err_msg;
{
    printf("Sim 1 [%d]: %s\n",err_no,err_msg);
    exit(1);
}
