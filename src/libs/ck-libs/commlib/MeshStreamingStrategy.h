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
//#include "commlib.h"

#define DEFAULT_MAX_BUCKET_SIZE 1000
#define DEFAULT_FLUSH_PERIOD 10

class MeshStreamingStrategy : public Strategy {
  public:
    MeshStreamingStrategy ();
    MeshStreamingStrategy (int period, int bucket_size);
    MeshStreamingStrategy (CkMigrateMessage *) { }
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

  private:
    int num_pe;
    int num_columns;
    int num_rows;
    int row_length;

    int my_pe;
    int my_column;
    int my_row;

    int max_bucket_size;
    int flush_period;

    int strategy_id;

    int column_handler_id;

    CkQ<char *> *column_bucket;
    int *column_bytes;
    CkQ<char *> *row_bucket;
};
#endif
