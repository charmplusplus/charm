#include "charm.h"

#include "trace.h"


/**********************************************************************/
/* This is perhaps the most crucial function in the entire system.    
Everything that a message does depends on these. To avoid computing
various offsets again and again, they have been made into variables,
computed only once at initialization. Any changes made to the layout
of the message must be reflected here. */
/**********************************************************************/
void InitializeMessageMacros(void)
{
}	
