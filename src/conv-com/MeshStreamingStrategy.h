/**
   @addtogroup ComlibConverseStrategy
   @{
   @file
   This is MeshStreamingStrategy, a strategy in the Charm++ communications
   library.  See MeshStreamingStrategy.C for detailed comments.

   @author Greg Koenig
   @author Moved into converse comlib by Filippo Gioachin 02/2006
*/

#ifndef MESH_STREAMING_STRATEGY
#define MESH_STREAMING_STRATEGY

#include <math.h>

#include "convcomlibmanager.h"

#define DEFAULT_FLUSH_PERIOD 10        // milliseconds
#define DEFAULT_MAX_BUCKET_SIZE 1000   // number of messages

CkpvExtern(int, streaming_column_handler_id);
extern void streaming_column_handler(void *msg);

/// Passed along with every row message header in the first iteration of
/// the MesgStreamingStrategy
struct MeshStreamingHeader {
    char conv_hdr[CmiMsgHeaderSizeBytes];
    int strategy_id;
    int num_msgs;
};

PUPbytes(MeshStreamingHeader)

/**
 * This is MeshStreamingStrategy, a strategy in the Charm++ communications
 * library.  In this strategy, processes are organized into a mesh as
 * depicted in the following diagram:
 *
 *  1    2    3    4
 *
 *  5    6    7    8
 *
 *  9   10   11   12
 * 
 * 13   14   15   16
 *
 * If, for example, PE 6 sends a message to PE 4 and a message to PE 12,
 * both messages will be stored in a column bucket on PE 6 for destination
 * column 3.  After DEFAULT_FLUSH_PERIOD milliseconds elapse or
 * DEFAULT_MAX_BUCKET_SIZE messages accumulate, all messages in the bucket
 * for column 3 are flushed by bundling them together and sending them at
 * once to PE 8.  When they arrive on PE 8 and are delivered to its
 * column_handler(), PE 8 breaks the messages in the bundle apart and
 * stores messages destined for individual rows in separate row buckets.
 * In the case of our example, the message destined for PE 4 would be
 * stored in the row bucket for row 0 and the message destined for PE 12
 * would be stored in the row bucket for row 2.  Again, after
 * DEFAULT_FLUSH_PERIOD milliseconds elapse or DEFAULT_MAX_BUCKET_SIZE
 * messages accumulate, all messages in the row buckets are flushed by
 * bundling them together and sending them at once to their respective
 * destination PEs by calling CmiMultipleSend().
 *
 * The advantage of bundling messages together is that to send to N
 * PEs in a computation, sqrt(N) actual messages need to be sent on the
 * network.  This trades computational overhead of bundling messages
 * for communication overhead; if the processors are fast relative to
 * the network, we win.
 *
 * To understand the bundling/unbundling operations, knowledge of Converse
 * and Charm++ message header formats is required.  I have attempted to
 * provide documentation within this code to describe what is going on.
 */
class MeshStreamingStrategy : public Strategy
{
  //bool shortMsgPackingFlag;
 public:
    MeshStreamingStrategy (int period=DEFAULT_FLUSH_PERIOD,
			   int bucket_size=DEFAULT_MAX_BUCKET_SIZE);
    MeshStreamingStrategy (CkMigrateMessage *m) : Strategy(m){ }
        
    void insertMessage (MessageHolder *msg);
    void doneInserting ();
    //void beginProcessing (int ignored);
    void RegisterPeriodicFlush (void);
    void FlushColumn (int column);
    void FlushRow (int row);
    void FlushBuffers (void);
    void InsertIntoRowBucket (int row, char *msg);
    int GetRowLength (void);
    virtual void pup (PUP::er &p);
    PUPable_decl (MeshStreamingStrategy);

    virtual void handleMessage(void *msg) {
      CmiAbort("[%d] MeshStreamingStrategy::handleMessage should never be called\n");
    }

    //Should be used only for array messages
    //virtual void enableShortArrayMessagePacking()
    //{shortMsgPackingFlag=true;} 

  private:

    int num_pe;
    int num_columns;
    int num_rows;
    int row_length;

    int my_pe;
    int my_column;
    int my_row;

    int flush_period;
    int max_bucket_size;

    //int strategy_id;

    //int column_handler_id;

    CkQ<char *> *column_bucket;
    CkQ<int> *column_destQ;

    int *column_bytes;
    CkQ<char *> *row_bucket;
};
#endif

/*@}*/
