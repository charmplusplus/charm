#include "charm-api.h"

/* CmiIsomallocContext comes from converse.h, which charm-api.h includes. */

CLINKAGE void AMPI_Rank_Setup(int myrank, int numranks, CmiIsomallocContext ctx);
void AMPI_Rank_Setup(int myrank, int numranks, CmiIsomallocContext ctx)
{
}
