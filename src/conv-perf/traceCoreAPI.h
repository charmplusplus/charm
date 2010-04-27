
#ifndef __TRACE_CORE_API_H__
#define __TRACE_CORE_API_H__

CpvCExtern(int,_traceCoreOn);
#ifdef __cplusplus
extern "C" {
#endif
/* Tracing API */
#if CMK_TRACE_IN_CHARM || ! CMK_TRACE_ENABLED
#define LOGCONDITIONAL(x) 
#else 
#define LOGCONDITIONAL(x) do { if(CpvAccess(_traceCoreOn)!=0){ \
			x;\
		}  } while(0);
#endif

void RegisterLanguage(int lID, const char* ln);
void RegisterEvent(int lID, int eID);
/* TODO some cleanup required below */
void LogEvent(int lID, int eID);
void LogEvent1(int lID, int eID, int iLen, const int* iData);
void LogEvent2(int lID, int eID, int sLen, const char* sData);
void LogEvent3(int lID, int eID, int iLen, const int* iData, int sLen, const char* sData);
void LogEvent4(int lID, int eID, int iLen, const int* iData, double t);
#ifdef __cplusplus
}
#endif

#endif
