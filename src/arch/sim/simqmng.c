
/* ************************************************************** */
/*  Insert the item into the queue *front_ptr. The queue is kept  */
/*  sorted with respect to the comparison function.               */
/* ************************************************************** */

static insert(front_ptr,comparison,item)
MSG **front_ptr;
int  (*comparison)();
MSG *item;
{ 
    MSG *prev,*current;
#if HELPME
    CmiPrintf("Inserting msg: len %d:  dest %d: \n",
	     *(item->length), *(item->dest)) ;
#endif
    if (*front_ptr == NULL) {
	*front_ptr = item;
	return;
    }
     prev = NULL;
    current = *front_ptr;
    while( current ) {
        if (  comparison(item,current) ) {
	    prev = current;
	    current = current->next;
	}
        else 
	    break;
    }
     if (current == NULL) {
        prev->next = item;
        return;
    }
    if (prev == NULL) 
        *front_ptr = item;
    else
	prev->next = item;
    item->next = current;
}

static not_empty(front)
void *front;
{
    return (front) ? TRUE : FALSE;
}

static empty(front)
void *front;
{
    return (front) ? FALSE : TRUE;
}

static is_first_element(front,element)
void *front, *element;
{
    return (front == element) ? TRUE : FALSE;
}

static SIM_TIME next_msg_arrival(front)
MSG *front;
{
    /* valid for MSG_QUEUE */
    return front->arrival_time;
}

/* ************************************************************** */
/* remove the front element                                       */
/* ************************************************************** */

static void *remove_front(front_ptr)
MSG **front_ptr;
{    
    void *front_element;
    
    if (*front_ptr == NULL) return NULL;
    front_element = *front_ptr;
    *front_ptr = (*front_ptr)->next;
    return front_element;
}

static int ge(arg1,arg2)
MSG *arg1,*arg2;
{
    return( !less_time(arg1->arrival_time,arg2->arrival_time));
}

/* temporary FIFO implementation */

static fifo_enqueue(q,msg)
PE_MSG_QUEUE *q;
MSG          *msg;
{
    msg->next = NULL; 
    if (q->front == NULL) 
	q->front = q->rear = msg;
    else {
	q->rear->next  = msg;
	q->rear        = msg;
    }
    q->num_of_elem++;
    q->size += msg->length;
}

static MSG *fifo_dequeue(q)
PE_MSG_QUEUE *q;
{
    MSG *msg;   
    msg = q->front;
    if (q->front == q->rear)
	q->front = q->rear = NULL;
    else 
	q->front = msg->next;
    msg->next = NULL;
    q->num_of_elem--;
    q->size -= msg->length;
    return msg;
}

static fifo_empty(q)
PE_MSG_QUEUE *q;
{
    return ( (q->front) ? FALSE:TRUE);
}
