
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  EnType *rtype;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Template *templat;
  TypeList *typelist;
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  char *strval;
  int intval;
} YYSTYPE;
extern YYSTYPE yylval;
# define MODULE 257
# define MAINMODULE 258
# define EXTERN 259
# define READONLY 260
# define CHARE 261
# define GROUP 262
# define NODEGROUP 263
# define ARRAY 264
# define MESSAGE 265
# define CLASS 266
# define STACKSIZE 267
# define THREADED 268
# define TEMPLATE 269
# define SYNC 270
# define EXCLUSIVE 271
# define VOID 272
# define PACKED 273
# define VARSIZE 274
# define ENTRY 275
# define MAINCHARE 276
# define IDENT 277
# define NUMBER 278
# define LITERAL 279
# define INT 280
# define LONG 281
# define SHORT 282
# define CHAR 283
# define FLOAT 284
# define DOUBLE 285
# define UNSIGNED 286
