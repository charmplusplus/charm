/**************************************************************************
** Greg Koenig (koenig@uiuc.edu)
**
** This is MeshStreamingStrategy, a strategy in the Charm++ communications
** library.  See MeshStreamingStrategy.C for detailed comments.
*/

#ifndef MESH_STREAMING_STRATEGY
#define MESH_STREAMING_STRATEGY

#include <assert.h>
#include <math.h>

#include "ComlibManager.h"

#define DEFAULT_FLUSH_PERIOD 10        // milliseconds
#define DEFAULT_MAX_BUCKET_SIZE 1000   // number of messages

//Passed along with every row message header in the first iteration of
//the MesgStreamingStrategy
struct MeshStreamingHeader {
    char conv_hdr[CmiMsgHeaderSizeBytes];
    int strategy_id;
    int num_msgs;
};

PUPbytes(MeshStreamingHeader);

class MeshStreamingStrategy : public CharmStrategy
{
    CmiBool shortMsgPackingFlag;
 public:
    MeshStreamingStrategy (int period=DEFAULT_FLUSH_PERIOD,
			   int bucket_size=DEFAULT_MAX_BUCKET_SIZE);
    MeshStreamingStrategy (CkMigrateMessage *m) : CharmStrategy(m){ }
        
    void insertMessage (CharmMessageHolder *msg);
    void doneInserting ();
    void beginProcessing (int ignored);
    void RegisterPeriodicFlush (void);
    void FlushColumn (int column);
    void FlushRow (int row);
    void FlushBuffers (void);
    void InsertIntoRowBucket (int row, char *msg);
    int GetRowLength (void);
    virtual void pup (PUP::er &p);
    PUPable_decl (MeshStreamingStrategy);

    //Should be used only for array messages
    virtual void enableShortArrayMessagePacking()
    {shortMsgPackingFlag=CmiTrue;} 

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

    int strategy_id;

    int column_handler_id;

    CkQ<char *> *column_bucket;
    CkQ<int> *column_destQ;

    int *column_bytes;
    CkQ<char *> *row_bucket;
};
#endif
