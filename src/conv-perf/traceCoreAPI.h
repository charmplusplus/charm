
#ifndef __TRACE_CORE_API_H__
#define __TRACE_CORE_API_H__

#ifdef __cplusplus
extern "C" {
#endif

/* Tracing API */
void RegisterLanguage(int lID, char* ln);
void RegisterEvent(int lID, int eID);
/* TODO some cleanup required below */
void LogEvent(int lID, int eID);
void LogEvent1(int lID, int eID, int iLen, int* iData);
void LogEvent2(int lID, int eID, int sLen, char* sData);
void LogEvent3(int lID, int eID, int iLen, int* iData, int sLen, char* sData);

#ifdef __cplusplus
}
#endif

#endif
