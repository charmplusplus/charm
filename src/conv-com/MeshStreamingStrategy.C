/**
   @addtogroup ComlibConverseStrategy   
   @{
   @file 
*/

#include "MeshStreamingStrategy.h"
#include "pup_cmialloc.h"
//#include "MsgPacker.h"

/**** not needed any-more after pup_CmiAlloc 
// These macros are taken directly from convcore.c.
#define SIZEFIELD(m) (((CmiChunkHeader *)(m))[-1].size)
#define REFFIELD(m) (((CmiChunkHeader *)(m))[-1].ref)
#define BLKSTART(m) (((CmiChunkHeader *)(m))-1)
***/

// These externs are defined inside ComlibManager.C.
//CkpvExtern(CkGroupID, cmgrID);
//CkpvExtern(int, RecvmsgHandle);

CkpvDeclare(int, streaming_column_handler_id);

/**************************************************************************
** This handler is invoked automatically when the processor goes idle.
**
** The idle handler automatically re-registers itself, so there is no need
** to re-register it from here.
**
** If nothing else is going on anyway, we might as well flush the buffers
** now instead of waiting for the flush period.
*/
void idle_flush_handler (void *ptr, double curT)
{
  ComlibPrintf ("[%d] idle_flush_handler() invoked.\n", CkMyPe());

  MeshStreamingStrategy *classptr = (MeshStreamingStrategy *) ptr;
  classptr->FlushBuffers ();
}



/**************************************************************************
** This handler is invoked automatically after a timeout occurs.
**
** The periodic handler does not automatically re-register itself, so it
** calls RegisterPeriodicFlush() to do so after it finishes flushing
** buffers.
*/
void periodic_flush_handler (void *ptr, double curT)
{
  ComlibPrintf ("[%d] periodic_flush_handler() invoked.\n", CkMyPe());

  MeshStreamingStrategy *classptr = (MeshStreamingStrategy *) ptr;
  classptr->FlushBuffers ();
  classptr->RegisterPeriodicFlush ();
}



/**************************************************************************
** This handler is invoked automatically when a packed message for a column
** is received.
**
** The layout of the message received is shown in the diagram below.
**
**             \  /
** +------------||---------------------------------------------+
** |Conv| I | # || dest || size | ref || Converse  || user | ...
** |hdr | D |   ||  PE  ||      | cnt ||  header   || data | ...
** +------------||---------------------------------------------+
**             /  \
**
** The function first retrieves the strategy ID and the number of messages
** in the packed message and then uses the strategy ID to obtain a pointer
** to the MeshStreamingStrategy class.  It also obtains the row length by
** calling GetRowLength().
**
** The function then iterates through the messages in the packed message.
** For each message within, it allocates space by calling CmiAlloc() and
** then copies the message from the packed buffer into the new message
** buffer.  It is also able to obtain the destination PE for the message
** because this information is included in the packed message data for
** each packed message.  If the destination PE is the current PE, the
** message is delivered immediately via a call to CmiSyncSendAndFree().
** This routine calls CmiFree() on the message, which is appropriate
** since it was allocated with CmiAlloc().  Otherwise, the message is
** inserted into the row bucket for the necessary row by calling
** InsertIntoRowBucket().  When messages are delivered from the row
** bucket, they are freed by CmiFree().
*/
void streaming_column_handler (void *msg)
{
    int dest_row;
    int my_pe;
    //int num_msgs;
    int row_length;
    //int strategy_id;
    //char *msgptr;
    char *newmsg;
    MeshStreamingStrategy *classptr;
        
    ComlibPrintf ("[%d] column_handler() invoked.\n", CkMyPe());
    
    my_pe = CkMyPe ();

    //PUP_cmialloc mem lets us use the converse reference counting
    //black magic in a transparent way. PUP_fromCmiAllocMem lets sub
    //messages in a messages be used freely in the program as messages.     
    PUP_fromCmiAllocMem fp(msg);    
    MeshStreamingHeader mhdr;
    
    //Read the header from the message
    fp | mhdr;
    
    //strategy_id = ((int *) (msg + CmiMsgHeaderSizeBytes))[0];
    //num_msgs = ((int *) (msg + CmiMsgHeaderSizeBytes))[1];
    
    classptr = (MeshStreamingStrategy *)ConvComlibGetStrategy(mhdr.strategy_id);
    //        CProxy_ComlibManager (CkpvAccess (cmgrID)).
    //        ckLocalBranch()->getStrategy (mhdr.strategy_id);
    
    row_length = classptr->GetRowLength ();
    
    //msgptr = (char *) (msg + CmiMsgHeaderSizeBytes + 2 * sizeof(int));
    
    for (int i = 0; i < mhdr.num_msgs; i++) {
        /*
        dest_pe = ((int *) msgptr)[0];
        msgsize = ((int *) msgptr)[1];
        
        newmsg = (char *) CmiAlloc (msgsize);
        
        memcpy (newmsg, (msgptr + 3 * sizeof(int)), msgsize);
        
        if (dest_pe == my_pe) {
            CmiSyncSendAndFree (my_pe, msgsize, newmsg);
        } else {
            dest_row = dest_pe / row_length;
            classptr->InsertIntoRowBucket (dest_row, newmsg);
        }
        
        msgptr += msgsize + 3 * sizeof(int);
        */

        int dest_pe;
        fp | dest_pe;

        //Returns a part of a message as an independent message and
        //updates the reference count of the container message.
        fp.pupCmiAllocBuf((void **)&newmsg);
        int msgsize = SIZEFIELD(newmsg);// ((envelope*)newmsg)->getTotalsize();

        if (dest_pe == my_pe) {
            CmiSyncSendAndFree (my_pe, msgsize, newmsg);
        } else {
            dest_row = dest_pe / row_length;
            classptr->InsertIntoRowBucket (dest_row, newmsg);
        }
    }
    
    CmiFree (msg);
}



/**************************************************************************
** This is the MeshStreamingStrategy constructor.
**
** The period and bucket_size have default values specified in the .h file.
**
** The constructor is invoked when the client code instantiates this
** strategy.  The constructor executes on a SINGLE PROCESS in the
** computation, so it cannot do things like determine an individual
** process's position within the mesh.
**
** After the constructor is invoked, the communications library creates
** instances that get pup'ed and shipped to each processor in the
** computation.  To that end, the process that instantiates this strategy
** (most likely PE 0) will then use pup to pack copies of the strategy
** and then ship them off to other processes.  They will be un-pup'ed
** there.  Finally, beginProcessing() will be called on EACH instance on
** its target processor.
*/
MeshStreamingStrategy::MeshStreamingStrategy (int period, int bucket_size) 
    : Strategy() 
{
    ComlibPrintf ("[%d] MeshStreamingStrategy::MeshStreamingStrategy() invoked.\n", CkMyPe());

    num_pe = CkNumPes ();
    
    num_columns = (int) (ceil (sqrt ((double) num_pe)));
    num_rows = num_columns;
    row_length = num_columns;
    
    flush_period = period;
    max_bucket_size = bucket_size;
    
    column_bucket = new CkQ<char *>[num_columns];
    column_destQ = new CkQ<int>[num_columns];
    column_bytes = new int[num_columns];
    row_bucket = new CkQ<char *>[num_rows];

    //shortMsgPackingFlag = false;
}



/**************************************************************************
** This method is called when the communications library sends a message
** from one PE to another PE.  This could be due to a direct message being
** sent, or due to a method invocation with marshalled parameters.
**
** The method begins by getting the destination PE from the
** CharmMessageHolder that is passed in (and from this, computing the
** destination column) and getting a pointer to the User data for the
** message (and from this, computing the Envelope pointer and the Block
** pointer).  The following diagram shows the layout of the message.
**
**     +----------------------------------------------------+
**     | size | refcount || Converse  ||        user        |
**     |      |          ||  header   ||        data        |
**     +----------------------------------------------------+
**     ^                  ^            ^
**     |                  |            |
**     blk (Block)        msg          usr (User)
**
** All Converse messages are allocated by CmiAlloc() which prepends two ints to
** all memory regions to hold a size field and a refcount field. BLKSTART() is a
** macro that gets the start of a block from the envelope pointer.
**
** If the destination PE is our current PE, we just deliver the message
** immediately.
**
** Otherwise, if the destination PE is in the same column as our PE, we
** allocate a new region of memory with CmiAlloc() and copy from the
** envelope pointer into the new region, and then deposit this new message
** into the appropriate row bucket for our column.  (The row buckets are
** queues of pointers to memory regions exactly like the diagram above.
** All entries in the row bucket are allocated with CmiAlloc() and must
** be deallocated with CmiFree()!)
**
** Otherwise, the destination PE must be in a different column from our
** PE.  We allocate a new region of memory with "new" that looks like
** the diagram below.
**
** +------------------------------------------------------------+
** | dest || size | refcount || Converse  ||        user        |
** |  PE  ||      |          ||  header   ||        data        |
** +------------------------------------------------------------+
** ^       ^                  ^            ^
** |       |                  |            |
** newmsg  blk (Block)        msg          usr (User)
**
** We then deposit this new message into the appropriate column bucket.
** (The column buckets are queues of pointers that are allocated with
** "new" and must be deallocated with "delete"!)
*/

void MeshStreamingStrategy::insertMessage (MessageHolder *cmsg)
{
    int dest_pe;
    int dest_row;
    int dest_col;
    int msg_size;
    int total_size;
    char *msg;
    //char *env;
    //char *blk;
    //char *newmsg;
    
    ComlibPrintf ("[%d] MeshStreamingStrategy::insertMessage() invoked.\n", 
                  CkMyPe());
    
    dest_pe = cmsg->dest_proc;
    dest_col = dest_pe % num_columns;
    msg = cmsg->getMessage ();
    //env = (char *) UsrToEnv (usr);
    
    //blk = (char *) BLKSTART (env);
    msg_size = SIZEFIELD(msg);//((envelope *)env)->getTotalsize();

    //misc_size = (env - blk);
    total_size = sizeof (int) + sizeof(CmiChunkHeader) + msg_size;
    
    if (dest_pe == my_pe) {
        CmiSyncSend (my_pe, msg_size, msg);
    } else if (dest_col == my_column) {
        //newmsg = (char *) CmiAlloc (env_size);
        //memcpy (newmsg, env, env_size);
        //newmsg = env;
        
        dest_row = dest_pe / row_length;
        
        InsertIntoRowBucket (dest_row, msg);
    } else {
        //newmsg = new char[total_size];
        //((int *) newmsg)[0] = dest_pe;
        //memcpy ( (void *) &(((int *) newmsg)[1]), blk, misc_size + env_size);
        
        column_bucket[dest_col].enq (msg);
        column_destQ[dest_col].enq(dest_pe);
        column_bytes[dest_col] += total_size;
        
        if (column_bucket[dest_col].length() > max_bucket_size) {
            FlushColumn (dest_col);
        }
    }
    
    delete cmsg;
}



/**************************************************************************
** This method is not used for streaming strategies.
*/
void MeshStreamingStrategy::doneInserting ()
{
    ComlibPrintf ("[%d] MeshStreamingStrategy::doneInserting() invoked.\n", CkMyPe());    
    // Empty for this strategy.

    //FlushBuffers();
    //Only want to flush local outgoing messages
    for (int column = 0; column < num_columns; column++) {
      FlushColumn ((column+my_column)%num_columns);
    }
}


/* *************************************************************************
** This method is invoked prior to any processing taking place in the
** class.  Various initializations take place here that cannot take place
** in the class constructor due to the communications library itself not
** being totally initialized.
**
** See MeshStreamingStrategy::MeshStreamingStrategy() for more details.
*/
/*
void MeshStreamingStrategy::beginProcessing (int ignored) {
    ComlibPrintf ("[%d] MeshStreamingStrategy::beginProcessing() invoked.\n", CkMyPe());
    
    //strategy_id = myInstanceID;
    
    my_pe = CkMyPe ();

    my_column = my_pe % num_columns;
    my_row = my_pe / row_length;
    
    //column_bucket = new CkQ<char *>[num_columns];
    //column_bytes = new int[num_columns];
    
    for (int i = 0; i < num_columns; i++) {
        column_bytes[i] = 0;
    }
    
    row_bucket = new CkQ<char *>[num_rows];
    
    column_handler_id = CkRegisterHandler ((CmiHandler) column_handler);
    
    CcdCallOnConditionKeepOnPE(CcdPROCESSOR_BEGIN_IDLE, idle_flush_handler,
                               (void *) this, CkMyPe());
    RegisterPeriodicFlush ();
}
*/


/**************************************************************************
** This method exists so periodic_flush_handler() can re-register itself to
** be invoked periodically to flush buffers.
*/
void MeshStreamingStrategy::RegisterPeriodicFlush (void)
{
  ComlibPrintf ("[%d] MeshStreamingStrategy::RegisterPeriodicFlush() invoked.\n", CkMyPe());

  CcdCallFnAfterOnPE(periodic_flush_handler, (void *) this, flush_period, CkMyPe());
}



/**************************************************************************
** This method is used to flush a specified column bucket, either as the
** result of the column bucket reaching its maximum capacity, as a result
** of the periodic flush handler being invoked, or as a result of the
** processor going idle.
**
** The method first finds the destination PE for the column.  This is the
** PE in the target column that is within the same row as the current PE.
**
** If there are actually messages in the bucket, then space is allocated
** to hold the new message which will pack all of the messages in the
** column bucket together.  The layout of this message is shown below:
**
**             \  /
** +------------||-------------------------------------------+
** |Conv| I | # || dest || size | ref || Converse  || user | ...
** |hdr | D |   ||  PE  ||      | cnt ||  header   || data | ...
** +------------||-------------------------------------------+
**             /  \
**
** Since the buffer represents a Converse message, it must begin with a
** Converse header.  After the header is an int representing the Commlib
** strategy ID for this strategy.  This is needed only so that the
** column_handler() can get a pointer to the MeshStreamingStrategy class
** later.  Next, comes an int containing the number of messages within
** the packed message.  Finally, the messages are removed from the column
** bucket and appended one after another into the buffer.
**
** After packing, a handler is set on the message to cause it to invoke
** column_handler() on the destination PE and the message is finally
** sent with CmiSyncSendAndFree().
**
** The buffer that is allocated in this message is used as a Converse
** message, so it is allocated with CmiAlloc() so the send routine can
** properly free it with CmiFree().  Therefore it has two ints for size
** and ref count at the beginning of the buffer.  These are not shown in
** the diagram above since they are basically irrelevant to this software.
*/

void MeshStreamingStrategy::FlushColumn (int column)
{
    int dest_column_pe;
    int num_msgs;
    int newmsgsize;
    char *newmsg;

    CmiAssert (column < num_columns);
    
    dest_column_pe = column + (my_row * row_length);
    if (dest_column_pe >= num_pe) {
      // This means that there is a hole in the mesh.
      //dest_column_pe = column + ((my_row % (num_rows - 1) - 1) * row_length);
      int new_row = my_column % (my_row + 1);
      if(new_row >= my_row)
	new_row = 0;

      dest_column_pe = column + new_row * row_length;
    }
    
    num_msgs = column_bucket[column].length ();
    
    if(num_msgs == 0)
      return;
    
    ComlibPrintf ("[%d] MeshStreamingStrategy::FlushColumn() invoked. to %d\n", 
		  CkMyPe(), dest_column_pe);    

    PUP_cmiAllocSizer sp;        
    int i = 0;
    MeshStreamingHeader mhdr;
    
    mhdr.strategy_id = getInstance();
    mhdr.num_msgs = num_msgs;
    sp | mhdr;
    
    for (i = 0; i < num_msgs; i++) {
      void *msg = column_bucket[column][i];
      int size = SIZEFIELD(msg);//((envelope *)msg)->getTotalsize();
      
      int destpe = column_destQ[column][i];
      sp | destpe;
      sp.pupCmiAllocBuf((void **)&msg, size);
    }
    
    newmsgsize = sp.size();
    newmsg = (char *) CmiAlloc (newmsgsize);
    
    //((int *) (newmsg + CmiMsgHeaderSizeBytes))[0] = strategy_id;
    //((int *) (newmsg + CmiMsgHeaderSizeBytes))[1] = num_msgs;
    
    PUP_toCmiAllocMem mp(newmsg);
    //make a structure header
    mp | mhdr;
    
    /*
      newmsgptr = (char *) (newmsg + CmiMsgHeaderSizeBytes + 2 * sizeof (int));               
      for (int i = 0; i < num_msgs; i++) {
      msgptr = column_bucket[column].deq ();            
      msgsize = ((int *) msgptr)[1] + (3 * sizeof (int));
      memcpy (newmsgptr, msgptr, msgsize);
      
      newmsgptr += msgsize;
      
      delete [] msgptr;
      }
    */
    
    for (i = 0; i < num_msgs; i++) {
      void *msg = column_bucket[column][i];
      int destpe = column_destQ[column][i];
      int size = SIZEFIELD(msg);//((envelope*)msg)->getTotalsize();
      
      mp | destpe;
      mp.pupCmiAllocBuf((void **)&msg, size);
    }
    
    for (i = 0; i < num_msgs; i++) {
      void *msg = column_bucket[column].deq();
      CmiFree(msg);
      
      column_destQ[column].deq();
    }
    
    column_bytes[column] = 0;        
    CmiSetHandler (newmsg, CkpvAccess(streaming_column_handler_id));        
    CmiSyncSendAndFree (dest_column_pe, newmsgsize, newmsg);
}


/**************************************************************************
** This method is used to flush a specified row bucket, either as the
** result of the row bucket reaching its maximum capacity, as a result
** of the periodic flush handler being invoked, or as a result of the
** processor going idle.
**
** The method first finds the destination PE for the row.  The method then
** iterates through the messages in the row bucket and constructs an array
** for sizes[] of the message sizes and an array for msgComps[] of
** pointers to the messages in the row bucket.  The method also sets the
** handler for each message to be "RecvmsgHandle" which is the handler
** for multi-message sends.  Finally, the method calls CmiMultiSend() to
** send all messages to the destination PE in one go.
**
** After the row bucket is emptied, the method calls CmiFree() to
** deallocate space for the individual messages.  Since each message was
** allocated via CmiAlloc() this is appropriate.
**
** Each message in the row bucket has the layout shown in the diagram
** below.
**
**     +----------------------------------------------------+
**     | size | refcount || Converse  ||        user        |
**     |      |          ||  header   ||        data        |
**     +----------------------------------------------------+
**                        ^
**                        |
**                        msg
**
*/

void MeshStreamingStrategy::FlushRow (int row)
{
    int dest_pe;
    int num_msgs;
    int *sizes;
    char *msg;
    char **msgComps;
    int i;
    
    ComlibPrintf ("[%d] MeshStreamingStrategy::FlushRow() invoked.\n", 
                  CkMyPe());
    
    CmiAssert (row < num_rows);
    
    dest_pe = my_column + (row * row_length);
    
    num_msgs = row_bucket[row].length ();
    if (num_msgs > 0) {
        
        //Strip charm++ envelopes from messages
      /*
        if(shortMsgPackingFlag) {
	    MsgPacker mpack(row_bucket[row], num_msgs);
            CombinedMessage *msg; 
            int size;
            mpack.getMessage(msg, size);
            
            CmiSyncSendAndFree(dest_pe, size, (char *)msg);
            return;
        }
      */
        //Send messages without short message packing
        sizes = new int[num_msgs];
        msgComps = new char *[num_msgs];
        
        for (i = 0; i < num_msgs; i++) {
            msg = row_bucket[row].deq ();
            //CmiSetHandler (msg, CkpvAccess(RecvmsgHandle));
            sizes[i] = SIZEFIELD(msg);//((envelope *)msg)->getTotalsize();
            msgComps[i] = msg;
        }
        
        CmiMultipleSend (dest_pe, num_msgs, sizes, msgComps);
        
        for (i = 0; i < num_msgs; i++) {
            CmiFree (msgComps[i]);
        }
        
        delete [] sizes;
        delete [] msgComps;
    }
}



/**************************************************************************
** This method exists so various handlers can easily trigger all column
** buckets and row buckets to flush.
*/
void MeshStreamingStrategy::FlushBuffers (void)
{
    ComlibPrintf ("[%d] MeshStreamingStrategy::PeriodicFlush() invoked.\n", 
                  CkMyPe());

    for (int column = 0; column < num_columns; column++) {
      FlushColumn ((column+my_column)%num_columns);
    }
    
    for (int row = 0; row < num_rows; row++) {
      FlushRow ((row+my_row)%num_rows);
    }
}



/**************************************************************************
** This method exists primarily so column_handler() can insert messages
** into a specified row bucket.
*/
void MeshStreamingStrategy::InsertIntoRowBucket (int row, char *msg)
{
  ComlibPrintf ("[%d] MeshStreamingStrategy::InsertIntoRowBucket() invoked.\n", CkMyPe());

  CmiAssert (row < num_rows);

  row_bucket[row].enq (msg);
  if (row_bucket[row].length() > max_bucket_size) {
    FlushRow (row);
  }
}



/**************************************************************************
** This method exists only so column_handler() can get the length of a row
** in the mesh.  Since it is outside of the MeshStreamingStrategy class, it
** does not have direct access to the class variables.
*/
int MeshStreamingStrategy::GetRowLength (void)
{
  ComlibPrintf ("[%d] MeshStreamingStrategy::GetRowLength() invoked.\n", CkMyPe());

  return (row_length);
}



/**************************************************************************
** This is a very complicated pack/unpack method.
**
** This method must handle the column_bucket[] and row_bucket[] data
** structures.  These are arrays of queues of (char *).  To pack these,
** we must iterate through the data structures and pack the sizes of
** each message (char *) pointed to by each queue entry.
*/
void MeshStreamingStrategy::pup (PUP::er &p)
{

  ComlibPrintf ("[%d] MeshStreamingStrategy::pup() invoked.\n", CkMyPe());

  // Call the superclass method -- easy.
  Strategy::pup (p);

  // Pup the instance variables -- easy.
  p | num_pe;
  p | num_columns;
  p | num_rows;
  p | row_length;

  //p | my_pe;
  //p | my_column;
  //p | my_row;

  p | max_bucket_size;
  p | flush_period;
  //p | strategy_id;
  //p | column_handler_id;

  //p | shortMsgPackingFlag;

  // Handle the column_bucket[] data structure.
  // For each element in column_bucket[], pup the length of the queue
  // at that element followed by the contents of that queue.  For each
  // queue, pup the size of the message pointed to by the (char *)
  // entry, followed by the memory for the (char *) entry.
  if (p.isUnpacking ()) {
      column_bucket = new CkQ<char *>[num_columns];
      column_destQ = new CkQ<int>[num_columns];
  }

  /*In correct code, will only be useful for checkpointing though
  for (i = 0; i < num_columns; i++) {
    int length = column_bucket[i].length ();

    p | length;

    for (int j = 0; j < length; j++) {
        char *msg = column_bucket[i].deq ();
        int size = sizeof (int) + ((int *) msg)[1];
        p | size;
        p(msg, size);
    }
  }
  */

  // Handle the column_bytes[] data structure.
  // This is a straightforward packing of an int array.
  if (p.isUnpacking ()) {
      column_bytes = new int[num_columns];
  }

  p(column_bytes, num_columns);

  // Handle the row_bucket[] data structure.
  // This works exactly like the column_bucket[] above.
  if (p.isUnpacking ()) {
    row_bucket = new CkQ<char *>[num_rows];
  }
  
  /* In correct code, will only be useful for checkpointing though
  for (i = 0; i < num_rows; i++) {
    int length = row_bucket[i].length ();

    p | length;

    for (int j = 0; j < length; j++) {
      char *msg = row_bucket[i].deq ();
      int size = ((int *) msg)[0];
      p | size;
      p(msg, size);
    }
  }
  */

    my_pe = CkMyPe ();

    my_column = my_pe % num_columns;
    my_row = my_pe / row_length;
    
    //column_bucket = new CkQ<char *>[num_columns];
    //column_bytes = new int[num_columns];
    
    for (int i = 0; i < num_columns; i++) {
        column_bytes[i] = 0;
    }
    
    // packing called once on processor 0, unpacking called once on all processors except 0
    if (p.isPacking() || p.isUnpacking()) {
      //column_handler_id = CkRegisterHandler ((CmiHandler) column_handler);
    
      CcdCallOnConditionKeepOnPE(CcdPROCESSOR_BEGIN_IDLE, idle_flush_handler,
				 (void *) this, CkMyPe());
      RegisterPeriodicFlush ();
    }
}

PUPable_def(MeshStreamingStrategy)

/*@}*/
