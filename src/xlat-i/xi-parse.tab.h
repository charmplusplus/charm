#define BOC 257
#define CHARE 258
#define ENTRY 259
#define MESSAGE 260
#define PACKMESSAGE 261
#define READONLY 262
#define TABLE 263
#define THREADED 264
#define EXTERN 265
#define IDENTIFIER 266
typedef union {
	char *strval;
	int intval;
} YYSTYPE;
extern YYSTYPE yylval;
