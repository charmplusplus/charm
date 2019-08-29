
#ifndef __TRACE_CORE_COMMON_H__
#define __TRACE_CORE_COMMON_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Core Trace Module */
void initTraceCore(char** argv);

/* End Core Trace Module */
void closeTraceCore(void);

/* Resume Core Trace Module */
void resumeTraceCore(void);

/* Suspend Core Trace Module */
void suspendTraceCore(void);

/*Install the beginIdle/endIdle condition handlers.*/
void beginTraceCore(void);

/*Cancel the beginIdle/endIdle condition handlers.*/
void endTraceCore(void);

#ifdef __cplusplus
}
#endif

#endif
