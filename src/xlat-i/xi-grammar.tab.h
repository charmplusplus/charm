typedef union {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Parameter *pname;
  ParamList *plist;
  Template *templat;
  TypeList *typelist;
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  PUPableClass *pupable;
  char *strval;
  int intval;
  Chare::attrib_t cattr;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	PUPABLE	262
#define	CHARE	263
#define	MAINCHARE	264
#define	GROUP	265
#define	NODEGROUP	266
#define	ARRAY	267
#define	MESSAGE	268
#define	CLASS	269
#define	STACKSIZE	270
#define	THREADED	271
#define	TEMPLATE	272
#define	SYNC	273
#define	EXCLUSIVE	274
#define	IMMEDIATE	275
#define	VIRTUAL	276
#define	MIGRATABLE	277
#define	CREATEHERE	278
#define	CREATEHOME	279
#define	VOID	280
#define	CONST	281
#define	PACKED	282
#define	VARSIZE	283
#define	ENTRY	284
#define	IDENT	285
#define	NUMBER	286
#define	LITERAL	287
#define	CPROGRAM	288
#define	INT	289
#define	LONG	290
#define	SHORT	291
#define	CHAR	292
#define	FLOAT	293
#define	DOUBLE	294
#define	UNSIGNED	295


extern YYSTYPE yylval;
