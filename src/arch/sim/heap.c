#define PARENT(i) i>>1;
#define EVENT_TIME(i) pe_table[event[i]].etime
#define INDEX(i)      pe_table[i].index


/* extern PROCESSOR *pe_table; */

typedef int EVENT;

static EVENT *event;

static adjust_eq()
{
    unsigned int  i,j,p,n;
    EVENT         R;
    SIM_TIME      K;
    p = i = 1;
    n = event[0];
    j = 2*i;
    time_assign(&K,&EVENT_TIME(i));
    R = event[i];
    while (j<=n) {

      if (j<n)
          if ( less_time(EVENT_TIME(j+1),EVENT_TIME(j)) ) j++;

      if ( less_time(K,EVENT_TIME(j)) ) 
         {
            event[p] = R;
            INDEX(event[p]) = p; 
            return;
         }

      event[p]  = event[j];
      INDEX(event[p]) = p;
      j = 2*j;
      p = PARENT(j);

    }

    event[p]  = R;
    INDEX(event[p]) = p;
}


static relocate_event(i)
int i;
{
    unsigned int p;
    EVENT        R; 
   
    R = event[i];
    while( i != 1) {
       p = PARENT(i);
       event[i]  = event[p];
       INDEX(event[i]) = i;
       i = p;
    }
    event[i] = R;
    INDEX(event[i]) = i;
    adjust_eq();
}



static void *create_event_heap(n)
int n;
{
    int i;
  
    event = (EVENT *) SMALLOC(sizeof(EVENT)*(3*n+1),100);
    for(i=0; i<= 3*n; i++) {
        event[i] = i;
        pe_table[i].index = i;
        time_zero(&(EVENT_TIME(i)));
    }
    event[0] = 3*n;
    return event;
}

static next_event()
{
    return event[1];
}



