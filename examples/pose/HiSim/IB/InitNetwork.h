#ifndef __FATTREE
#define __FATTREE
#define INDIRECT_NETWORK TRUE
#define ROUTING_ALGORITHM UpDown
#define OUTPUT_VC_SELECTION maxAvailBufferSwitch
#define INPUT_VC_SELECTION SLQ_Switch

#define ROUTING_FILE "../Routing/UpDown.h"
#define OUTPUT_VC_FILE "../OutputVcSelection/maxAvailBufferSwitch.h"
#define INPUT_VC_FILE "../InputVcSelection/SLQ_Switch.h"

#endif
