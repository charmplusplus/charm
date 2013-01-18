/**
 * ANTLR (v3) grammar for the Charj Language
 *
 * The other .g files are tree parsers that can read and modify an AST 
 * using the output of this grammar.
 */


grammar Charj;

options {
  backtrack = true; 
  memoize = true;
  output = AST;
  ASTLabelType = CharjAST;
}

tokens {

    ASSERT                  = 'assert'          ;
    ENTRY                   = 'entry'           ;
    SDAGENTRY               = 'sdagentry'       ;
    TRACED                  = 'traced'          ;
    PUBLIC                  = 'public'          ;
    PROTECTED               = 'protected'       ;
    PRIVATE                 = 'private'         ;
    ABSTRACT                = 'abstract'        ;
    NATIVE                  = 'native'          ;
    FINAL                   = 'final'           ;
    STATIC                  = 'static'          ;
    VOLATILE                = 'volatile'        ;
    VOID                    = 'void'            ;
    BOOLEAN                 = 'boolean'         ;
    CHAR                    = 'char'            ;
    BYTE                    = 'byte'            ;
    SHORT                   = 'short'           ;
    IN                      = 'in'              ;
    INT                     = 'int'             ;
    LET                     = 'let'             ;
    LONG                    = 'long'            ;
    FLOAT                   = 'float'           ;
    DOUBLE                  = 'double'          ;
    TRUE                    = 'true'            ;
    FALSE                   = 'false'           ;
    NULL                    = 'null'            ;
    THIS                    = 'this'            ;
    SUPER                   = 'super'           ;
    CHARE                   = 'chare'           ;
    CHARE_ARRAY             = 'chare_array'     ;
    MAINCHARE               = 'mainchare'       ;
    PACKAGE                 = 'package'         ;
    IMPORT                  = 'import'          ;
    CLASS                   = 'class'           ;
    EXTENDS                 = 'extends'         ;
    GROUP                   = 'group'           ;
    NODEGROUP               = 'nodegroup'       ;
    ENUM                    = 'enum'            ;
    READONLY                = 'readonly'        ;
    ACCELERATED             = 'accelerated'     ;
    THREADED                = 'threaded'        ;
    REDUCTIONTARGET         = 'reductiontarget' ;
	CONTRIBUTE				= 'contribute'		;

    OVERLAP                 = 'overlap'         ;
    WHEN                    = 'when'            ;

    THISINDEX		        = 'thisIndex'	;
    THISPROXY		        = 'thisProxy'	;

    MESSAGE                 = 'message'          ;
    MULTICAST_MESSAGE       = 'multicast_message';

    FOR                     = 'for'             ;
    WHILE                   = 'while'           ;
    IF                      = 'if'              ;
    CASE                    = 'case'            ;
    SWITCH                  = 'switch'          ;
    RETURN                  = 'return'          ;
    ELSE                    = 'else'            ;
    CONTINUE                = 'continue'        ;
    DO                      = 'do'              ;
    DEFAULT                 = 'default'         ;
    WHILE                   = 'while'           ;
    THROW                   = 'throw'           ;
    BREAK                   = 'break'           ;

    DOT                     = '.'               ;
    NEW                     = 'new'             ;
    BITWISE_OR              = '|'               ;
    BITWISE_AND             = '&'               ;
    ASSIGNMENT              = '='               ;
    EQUALS                  = '=='              ;
    NOT_EQUALS              = '!='              ;
    PLUS_EQUALS             = '+='              ;
    MINUS_EQUALS            = '-='              ;
    TIMES_EQUALS            = '*='              ;
    DIVIDE_EQUALS           = '/='              ;
    AND_EQUALS              = '&='              ;
    OR_EQUALS               = '|='              ;
    POWER_EQUALS            = '^='              ;
    MOD_EQUALS              = '%='              ;
    OR                      = '||'              ;
    AND                     = '&&'              ;
    POWER                   = '^'               ;
    GT                      = '>'               ;
    GTE                     = '>='              ;
    LT                      = '<'               ;
    LTE                     = '<='              ;
    PLUS                    = '+'               ;
    MINUS                   = '-'               ;
    TIMES                   = '*'               ;
    DIVIDE                  = '/'               ;
    MOD                     = '%'               ;
    UNARY_PLUS              = '++'              ;
    UNARY_MINUS             = '--'              ;
    NOT                     = '!'               ;
    TILDE                   = '~'               ;
    AT                      = '@'               ;
    INSTANCEOF              = 'instanceof'      ;
    SIZEOF                  = 'sizeof'          ;

    // Charj keywords for things that are automatically generated
    // and we don't want the user to use them as identifiers

    PUP                     = 'pup'             ;
    INITMETHOD              = 'initMethod'      ;
    CTORHELPER              = 'ctorHelper'      ;
    CHELPER                 = 'constructorHelper';


    // C++ keywords that aren't used in charj. 
    // We don't use these ourselves, but they're still reserved
    ASM                     = 'asm'             ;
    AUTO                    = 'auto'            ;
    BOOL                    = 'bool'            ;
    CONST_CAST              = 'const_cast'      ;
    DYNAMIC_CAST            = 'dynamic_cast'    ;
    EXPLICIT                = 'explicit'        ;
    EXPORT                  = 'export'          ;
    EXTERN                  = 'extern'          ;
    FRIEND                  = 'friend'          ;
    GOTO                    = 'goto'            ;
    INLINE                  = 'inline'          ;
    MUTABLE                 = 'mutable'         ;
    NAMESPACE               = 'namespace'       ;
    OPERATOR                = 'operator'        ;
    REGISTER                = 'register'        ;
    REINTERPRET_CAST        = 'reinterpret_cast';
    SIGNED                  = 'signed'          ;
    STATIC_CAST             = 'static_cast'     ;
    STRUCT                  = 'struct'          ;
    TEMPLATE                = 'template'        ;
    TYPEDEF                 = 'typedef'         ;
    TYPEID                  = 'typeid'          ;
    TYPENAME                = 'typename'        ;
    UNION                   = 'union'           ;
    UNSIGNED                = 'unsigned'        ;
    USING                   = 'using'           ;
    VIRTUAL                 = 'virtual'         ;
    WCHAR_T                 = 'wchar_t'         ;

    // tokens for imaginary nodes
    ARGUMENT_LIST;
    ARRAY_DECLARATOR;
    ARRAY_DECLARATOR_LIST;
    ARRAY_ELEMENT_ACCESS;
    ARRAY_INITIALIZER;
    BLOCK;
    DIVCON_BLOCK;
    CAST_EXPR;
    CATCH_CLAUSE_LIST;
    CLASS_CONSTRUCTOR_CALL;
    CLASS_INSTANCE_INITIALIZER;
    CLASS_STATIC_INITIALIZER;
    CLASS_TOP_LEVEL_SCOPE;
    CONSTRUCTOR_DECL;
    CONTRIBUTE_1;
	CONTRIBUTE_2;
	DOMAIN_EXPRESSION;
    END;
	ENUM_TOP_LEVEL_SCOPE;
    EXPR;
    EXTENDS_BOUND_LIST;
    EXTENDS_CLAUSE;
    FOR_EACH;
    FOR_EXPR;
    FOR_UPDATE;
    FORMAL_PARAM_LIST;
    FORMAL_PARAM_STD_DECL;
    FUNCTION_METHOD_DECL;
    DIVCON_METHOD_DECL;
    GENERIC_TYPE_ARG_LIST;
    GENERIC_TYPE_PARAM_LIST;
    IMPLEMENTS_CLAUSE;
    LABELED_STATEMENT;
    LET_ASSIGNMENT;
    CHARJ_SOURCE;
    METHOD_CALL;
    ENTRY_METHOD_CALL;
    MODIFIER_LIST;
    NEW_EXPRESSION;
    PAREN_EXPR;
    POST_DEC;
    POST_INC;
    PRE_DEC;
    PRE_INC;
    QUALIFIED_TYPE_IDENT;
    RANGE_EXPRESSION;
    SDAG_IF;
    SDAG_DO;
    SDAG_FOR;
    SDAG_WHILE;
    STATIC_ARRAY_CREATOR;
    SUPER_CONSTRUCTOR_CALL;
    TEMPLATE_INST;
    THIS_CONSTRUCTOR_CALL;
    TYPE;
    SIMPLE_TYPE;
    OBJECT_TYPE;
    REFERENCE_TYPE;
    POINTER_TYPE;
    PROXY_TYPE;
    ARRAY_SECTION_TYPE;
    ARRAY_SECTION;
    ARRAY_SECTION_INIT;
    MESSAGE_TYPE;
    PRIMITIVE_VAR_DECLARATION;
    OBJECT_VAR_DECLARATION;
    VAR_DECLARATOR;
    VAR_DECLARATOR_LIST;
    ARROW;
    LOCAL_MODIFIER_LIST;
    ACCESS_MODIFIER_LIST;
    CHARJ_MODIFIER_LIST;
    OTHER_MODIFIER_LIST;
    POINTER_DEREFERENCE;
    ENTRY_FUNCTION_DECL;
    SDAG_FUNCTION_DECL;
    ENTRY_CONSTRUCTOR_DECL;
}

@header {
    package charj.translator;
}

@members {
}

@lexer::header {
    package charj.translator; 
}

@lexer::members {
}

// Starting point for parsing a Charj file.
charjSource
    :   compilationUnit EOF
        ->  ^(CHARJ_SOURCE compilationUnit)
    ;

compilationUnit
    :   packageDeclaration? 
        topLevelDeclaration+ 
    ;

topLevelDeclaration
    :   importDeclaration
    |   readonlyDeclaration
    |   externDeclaration
    |   typeDeclaration
    ;

packageDeclaration
    :   PACKAGE IDENT (DOT IDENT)* ';'
        ->  ^(PACKAGE IDENT+)
    ;

importDeclaration
    :   IMPORT^ qualifiedIdentifier '.*'? ';'!
    ;

readonlyDeclaration
    :   READONLY^ localVariableDeclaration ';'!
    ;

externDeclaration
    :   EXTERN^ qualifiedIdentifier ';'!
    ;

typeDeclaration
    :   classDefinition
    |   templateDeclaration
    |   enumDefinition
    |   chareDefinition
    |   messageDefinition
    ;

templateList
    : 'class'! IDENT (','! 'class'! IDENT)*
    ;

templateDeclaration
    : 'template' '<' templateList '>' classDefinition
        -> ^('template' templateList classDefinition)
    ;

classDefinition
    :   PUBLIC? CLASS IDENT (EXTENDS type)? ('implements' typeList)? '{'
            classScopeDeclaration* '}' ';'?
        -> ^(TYPE CLASS IDENT ^(EXTENDS type)? ^('implements' typeList)? classScopeDeclaration*)
    ;

chareType
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   CHARE_ARRAY '[' ARRAY_DIMENSION ']' -> ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

chareDefinition
    :   PUBLIC? chareType IDENT (EXTENDS type)? ('implements' typeList)? '{'
            classScopeDeclaration*
        '}' ';'?
        -> ^(TYPE chareType IDENT ^(EXTENDS type)? ^('implements' typeList)? classScopeDeclaration*)
    ;

enumDefinition
    :   ENUM IDENT ('implements' typeList)? '{'
            enumConstants ','? ';' classScopeDeclaration*
        '}' ';'?
        -> ^(ENUM IDENT ^('implements' typeList)? enumConstants classScopeDeclaration*)
    ;

messageDefinition
    :   MESSAGE IDENT '{' messageScopeDeclaration* '}' ';'?
        -> ^(MESSAGE IDENT messageScopeDeclaration*)
    |   MULTICAST_MESSAGE IDENT '{' messageScopeDeclaration* '}' ';'?
        -> ^(MULTICAST_MESSAGE IDENT messageScopeDeclaration*)
    ;

enumConstants
    :   enumConstant (','! enumConstant)*
    ;

enumConstant
    :   IDENT^ arguments?
    ;

typeList
    :   type (','! type)*
    ;

messageScopeDeclaration
    :	primitiveVariableDeclaration
	|	objectVariableDeclaration
    ;

classScopeDeclaration
    :   functionMethodDeclaration
    |	constructorDeclaration
    |	divconMethodDeclaration
    |	primitiveVariableDeclaration
    |	objectVariableDeclaration
    ;

functionMethodDeclaration
	:	modifierList? genericTypeParameterList? type IDENT formalParameterList (';' | block)
		->  ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? type IDENT formalParameterList block?)
	;

constructorDeclaration
	:	modifierList? genericTypeParameterList? ident=IDENT formalParameterList block
		->  ^(CONSTRUCTOR_DECL[$ident, "CONSTRUCTOR_DECL"] modifierList? genericTypeParameterList? IDENT formalParameterList block)
	;

divconMethodDeclaration
	:	modifierList? type IDENT formalParameterList divconBlock
		->  ^(DIVCON_METHOD_DECL modifierList? type IDENT formalParameterList divconBlock)
	;

primitiveVariableDeclaration
  : modifierList? simpleType classFieldDeclaratorList ';'
    ->  ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType classFieldDeclaratorList)
  ;

objectVariableDeclaration
  : modifierList? objectType classFieldDeclaratorList ';'
    ->  ^(OBJECT_VAR_DECLARATION modifierList? objectType classFieldDeclaratorList)
  ;

classFieldDeclaratorList
    :   classFieldDeclarator (',' classFieldDeclarator)*
        ->  ^(VAR_DECLARATOR_LIST classFieldDeclarator+)
    ;

classFieldDeclarator
    :   variableDeclaratorId (ASSIGNMENT variableInitializer)?
        ->  ^(VAR_DECLARATOR variableDeclaratorId variableInitializer?)
    ;

variableDeclaratorId
    :   IDENT^ domainExpression?
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayInitializer
    :   lc='{' (variableInitializer (',' variableInitializer)* ','?)? '}'
        ->  ^(ARRAY_INITIALIZER[$lc, "ARRAY_INITIALIZER"] variableInitializer*)
    ;

templateArg
    : genericTypeArgument
    | literal
    ;

templateArgList
    :   templateArg (','! templateArg)*
    ;

templateInstantiation
    :    '<' templateArgList '>'
        -> ^(TEMPLATE_INST templateArgList)
    |    '<' templateInstantiation '>'
        -> ^(TEMPLATE_INST templateInstantiation)
    ;

genericTypeParameterList
    :   lt='<' genericTypeParameter (',' genericTypeParameter)* genericTypeListClosing
        ->  ^(GENERIC_TYPE_PARAM_LIST[$lt, "GENERIC_TYPE_PARAM_LIST"] genericTypeParameter+)
    ;

// This hack is fairly dirty - we just bite off some angle brackets and don't
// actually match up opening and closing brackets.
genericTypeListClosing  
    :   '>'
    |   '>>'
    |   '>>>'
    |
    ;

genericTypeParameter
    :   IDENT bound?
        ->  ^(IDENT bound?)
    ;

bound
    :   e=EXTENDS type ('&' type)*
        ->  ^(EXTENDS_BOUND_LIST[$e, "EXTENDS_BOUND_LIST"] type+)
    ;

modifierList
    :   modifier+
        ->  ^(MODIFIER_LIST modifier+)
    ;

modifier
    :   PUBLIC
    |   PROTECTED
    |   ENTRY
    |   SDAGENTRY
    |   TRACED
    |   ACCELERATED
    |   THREADED
    |   PRIVATE
    |   ABSTRACT
    |   NATIVE
    |   localModifier
    ;

localModifierList
    :   localModifier+
        -> ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

localModifier
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

type
    :   simpleType
    |   objectType
    |   VOID
    ;

constructorType
    :   qualifiedTypeIdent AT domainExpression?
        ->  ^(PROXY_TYPE qualifiedTypeIdent domainExpression?)
    |   qualifiedTypeIdent domainExpression?
        ->  ^(OBJECT_TYPE qualifiedTypeIdent domainExpression?)
	|	MOD qualifiedTypeIdent AT domainExpression
		->	^(ARRAY_SECTION_TYPE qualifiedTypeIdent domainExpression)
    |   qualifiedTypeIdent TILDE 
        ->  ^(MESSAGE_TYPE qualifiedTypeIdent)
    ;

simpleType
    :   primitiveType domainExpression?
        ->  ^(SIMPLE_TYPE primitiveType domainExpression?)  
    ;

objectType
    :   qualifiedTypeIdent AT domainExpression?
        ->  ^(PROXY_TYPE qualifiedTypeIdent domainExpression?)
    |   qualifiedTypeIdent domainExpression?
        ->  ^(POINTER_TYPE qualifiedTypeIdent domainExpression?)
	|	qualifiedTypeIdent '[' MOD ']' AT
		->	^(ARRAY_SECTION_TYPE qualifiedTypeIdent)
	|	qualifiedTypeIdent '[' TILDE ']' AT
		->	^(MESSAGE_TYPE qualifiedTypeIdent)
    ;

qualifiedTypeIdent
    :   typeIdent (DOT typeIdent)*
        ->  ^(QUALIFIED_TYPE_IDENT typeIdent+) 
    ;

typeIdent
    :   IDENT^ templateInstantiation?
    ;

primitiveType
    :   BOOLEAN
    |   CHAR
    |   BYTE
    |   SHORT
    |   INT
    |   LONG
    |   FLOAT
    |   DOUBLE
    ;

genericTypeArgument
    :   type
    |   '?'
    ;

qualifiedIdentList
    :   qualifiedIdentifier (','! qualifiedIdentifier)*
    ;

formalParameterList
    :   lp='('
        (   // Contains at least one standard argument declaration and optionally a variable argument declaration.
            formalParameterStandardDecl (',' formalParameterStandardDecl)*
            ->  ^(FORMAL_PARAM_LIST[$lp, "FORMAL_PARAM_LIST"] formalParameterStandardDecl+)
            // Contains nothing.
        |   ->  ^(FORMAL_PARAM_LIST[$lp, "FORMAL_PARAM_LIST"]) 
        )
        ')'
    ;

formalParameterStandardDecl
    :   localModifierList? type variableDeclaratorId
        ->  ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId)
    ;

qualifiedIdentifier
    :   (   IDENT
            ->  IDENT
        )
        (   DOT ident=IDENT
            ->  ^(DOT $qualifiedIdentifier $ident)
        )*
    ;

block
    :   lc='{' blockStatement* '}'
        ->  ^(BLOCK[$lc, "BLOCK"] blockStatement*)
    |   nonBlockStatement
        -> ^(BLOCK nonBlockStatement)
    ;

blockStatement
    :   localVariableDeclaration ';'!
    |   statement
    ;

localVariableDeclaration
    :	primitiveVarDeclaration
    |   objectVarDeclaration
    ;

primitiveVarDeclaration
    :	localModifierList? simpleType classFieldDeclaratorList
        ->  ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType classFieldDeclaratorList)
    ;

objectVarDeclaration
    :	localModifierList? objectType classFieldDeclaratorList
        ->  ^(OBJECT_VAR_DECLARATION localModifierList? objectType classFieldDeclaratorList)
    ;

statement
    :   nonBlockStatement
    |   sdagStatement
    |   block
    ;

sdagTrigger
    : IDENT ('['! expression ']'!)? formalParameterList
    ;

sdagStatement
    :   OVERLAP block
        -> ^(OVERLAP block)
    |   WHEN (sdagTrigger (',' sdagTrigger)*)? block
        -> ^(WHEN sdagTrigger* block)
    ;

divconBlock
    :   divconExpr
        ->  ^(DIVCON_BLOCK divconExpr)
    ;

divconAssignment
    :   IDENT '=' expression
        -> ^(LET_ASSIGNMENT IDENT expression)
    ;

divconAssignmentList
    :   divconAssignment (','! divconAssignment)*
    ;

divconExpr
    :   IF parenthesizedExpression ifExpr=divconExpr
        (   ELSE elseExpr=divconExpr
            ->  ^(IF parenthesizedExpression $ifExpr $elseExpr)
        |
            ->  ^(IF parenthesizedExpression $ifExpr)
        )
    |   LET ^divconAssignmentList IN divconExpr
    |   expression ';'!
    |   '{'! divconExpr '}'!
    ;

nonBlockStatement
    :   ASSERT expr1=expression 
        (   ':' expr2=expression ';'
            ->  ^(ASSERT $expr1 $expr2)
        |   ';'
            ->  ^(ASSERT $expr1)
        )
    |   IF parenthesizedExpression ifStat=block
        (   ELSE elseStat=block
            ->  ^(IF parenthesizedExpression $ifStat $elseStat)
        |
            ->  ^(IF parenthesizedExpression $ifStat)
        )   
    |   f=FOR '('
        (   forInit? ';' expression? ';' expressionList? ')' block
            -> ^($f forInit? FOR_EXPR expression? FOR_UPDATE expressionList? block)
        |   localModifierList? type IDENT ':' expression ')' block
            -> ^(FOR_EACH[$f, "FOR_EACH"] localModifierList? type IDENT expression block)
        )
    |   WHILE parenthesizedExpression block
        ->  ^(WHILE parenthesizedExpression block)
    |   DO block WHILE parenthesizedExpression ';'
        ->  ^(DO block parenthesizedExpression)
    |   SWITCH parenthesizedExpression '{' switchCaseLabel* '}'
        ->  ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   RETURN expression? ';'
        ->  ^(RETURN expression?)
    |   THROW expression ';'
        ->  ^(THROW expression)
    |   BREAK IDENT? ';'
        ->  ^(BREAK IDENT?)
    |   CONTINUE IDENT? ';'
        ->  ^(CONTINUE IDENT?)
    |   IDENT ':' statement
        ->  ^(LABELED_STATEMENT IDENT statement)
    |   'delete' expression ';'
        -> ^('delete' expression)
    |   'embed' STRING_LITERAL EMBED_BLOCK
        ->  ^('embed' STRING_LITERAL EMBED_BLOCK)
	|	CONTRIBUTE '(' expression ')' ';'
		-> ^(CONTRIBUTE_1 expression)
	|	CONTRIBUTE '(' expression ',' expression ',' qualifiedIdentifier ',' expression ')' ';'
		-> ^(CONTRIBUTE_2 expression expression qualifiedIdentifier expression)
    |   expression ';'!
    |   ';' // Preserve empty statements.
    ;           
        

switchCaseLabel
    :   CASE^ expression ':'! blockStatement*
    |   DEFAULT^ ':'! blockStatement*
    ;
    
forInit
    :   localVariableDeclaration
    |   expressionList
    ;
    
// EXPRESSIONS

parenthesizedExpression
    :   lp='(' expression ')'
        ->  ^(PAREN_EXPR[$lp, "PAREN_EXPR"] expression)
    ;
    
rangeItem
    :   expression
    ;

rangeExpression
    :   rangeItem
        -> ^(RANGE_EXPRESSION rangeItem)
    |   rangeItem ':' rangeItem
        -> ^(RANGE_EXPRESSION rangeItem rangeItem)
    |   rangeItem ':' rangeItem ':' rangeItem
        -> ^(RANGE_EXPRESSION rangeItem rangeItem rangeItem)
    ;

rangeList
    :   rangeExpression (','! rangeExpression)*
    ;

domainExpression
    :   '[' rangeList ']'
        -> ^(DOMAIN_EXPRESSION rangeList)
    ;

expressionList
    :   expression (','! expression)*
    ;

expression
    :   assignmentExpression
        ->  ^(EXPR assignmentExpression)
    ;

assignmentExpression
    :   conditionalExpression 
        (   (   ASSIGNMENT^
            |   '+='^
            |   '-='^
            |   '*='^
            |   '/='^
            |   '&='^
            |   '|='^
            |   '^='^
            |   '%='^
            |   '<<='^
            |   '>>='^
            |   '>>>='^
            ) 
            assignmentExpression)?
    ;
    
conditionalExpression
    :   logicalOrExpression ('?'^ assignmentExpression ':'! conditionalExpression)?
    ;

logicalOrExpression
    :   logicalAndExpression ('||'^ logicalAndExpression)*
    ;

logicalAndExpression
    :   inclusiveOrExpression ('&&'^ inclusiveOrExpression)*
    ;

inclusiveOrExpression
    :   exclusiveOrExpression ('|'^ exclusiveOrExpression)*
    ;

exclusiveOrExpression
    :   andExpression ('^'^ andExpression)*
    ;

andExpression
    :   equalityExpression ('&'^ equalityExpression)*
    ;

equalityExpression
    :   instanceOfExpression 
        (   (   EQUALS^
            |   '!='^
            ) 
            instanceOfExpression
        )*
    ;

instanceOfExpression
    :   relationalExpression ('instanceof'^ type)?
    ;

relationalExpression
    :   shiftExpression 
        (   (   '<='^
            |   '>='^
            |   '<'^
            |   '>'^
            )
            shiftExpression
        )*
    ;
    
shiftExpression
    :   additiveExpression
        (   (   '>>>'^
            |   '>>'^
            |   '<<'^
            )
            additiveExpression
        )*
    ;

additiveExpression
    :   multiplicativeExpression
        (   (   '+'^
            |   '-'^
            )
            multiplicativeExpression
        )*
    ;

multiplicativeExpression
    :   unaryExpression 
        (   (   '*'^
            |   '/'^
            |   '%'^
            )
            unaryExpression
        )*
    ;
    
unaryExpression
    :   op='+' unaryExpression
        ->  ^(UNARY_PLUS[$op, "UNARY_PLUS"] unaryExpression)
    |   op='-' unaryExpression
        ->  ^(UNARY_MINUS[$op, "UNARY_MINUS"] unaryExpression)
    |   op='++' postfixedExpression
        ->  ^(PRE_INC[$op, "PRE_INC"] postfixedExpression)
    |   op='--' postfixedExpression
        ->  ^(PRE_DEC[$op, "PRE_DEC"] postfixedExpression)
    |   unaryExpressionNotPlusMinus
    ;

unaryExpressionNotPlusMinus
    :   '!' unaryExpression
        ->  ^('!' unaryExpression)
    |   '~' unaryExpression
        ->  ^('~' unaryExpression)
    |   lp='(' type ')' unaryExpression
        ->  ^(CAST_EXPR[$lp, "CAST_EXPR"] type unaryExpression)
    |   postfixedExpression
    ;
    
postfixedExpression
        // At first resolve the primary expression ...
    :   (   primaryExpression
            ->  primaryExpression
        )
        // ... and than the optional things that may follow a primary expression 0 or more times.
        (   outerDot=DOT                 
            // Note: generic type arguments are only valid for method calls,
            // i.e. if there is an argument list
            (   (   templateInstantiation?  
                    IDENT
                    ->  ^($outerDot $postfixedExpression IDENT)
                ) 
                (   arguments
                    ->  ^(METHOD_CALL $postfixedExpression templateInstantiation? arguments)
                )?
            |   THIS
                ->  ^($outerDot $postfixedExpression THIS)
            |   s=SUPER arguments
                ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] $postfixedExpression arguments)
            |   (   SUPER innerDot=DOT IDENT
                    ->  ^($innerDot ^($outerDot $postfixedExpression SUPER) IDENT)
                )
                (   arguments
                    ->  ^(METHOD_CALL $postfixedExpression arguments)
                )?
            )
        |   (AT templateInstantiation? IDENT arguments)
            ->  ^(ENTRY_METHOD_CALL ^(AT $postfixedExpression IDENT) templateInstantiation? arguments)
        |   domainExpression
            ->  ^(ARRAY_ELEMENT_ACCESS $postfixedExpression domainExpression)
        )*
        // At the end there may follow a post increment/decrement.
        (   op='++'-> ^(POST_INC[$op, "POST_INC"] $postfixedExpression)
        |   op='--'-> ^(POST_DEC[$op, "POST_DEC"] $postfixedExpression)
        )?
    ;    
    
primaryExpression
    :   parenthesizedExpression
    |   literal
    |   newExpression
    |   qualifiedIdentExpression
    |   domainExpression
    |   templateInstantiation
        (   s=SUPER
            (   arguments
                ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] templateInstantiation arguments)
            |   IDENT arguments
                ->  ^(METHOD_CALL ^(DOT SUPER IDENT) templateInstantiation arguments)
            )
        |   IDENT arguments
            ->  ^(METHOD_CALL IDENT templateInstantiation arguments)
        |   t=THIS arguments
            ->  ^(THIS_CONSTRUCTOR_CALL[$t, "THIS_CONSTRUCTOR_CALL"] templateInstantiation arguments)
        )
    |   (   THIS
            ->  THIS
        )
        (   arguments
            ->  ^(THIS_CONSTRUCTOR_CALL[$t, "THIS_CONSTRUCTOR_CALL"] arguments)
        )?
    |   s=SUPER arguments
        ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] arguments)
    |   (   SUPER DOT IDENT
        )
        (   arguments
            ->  ^(METHOD_CALL ^(DOT SUPER IDENT) arguments)
        |   ->  ^(DOT SUPER IDENT)
        )
	|	THISINDEX
	|	THISPROXY
    |   SIZEOF '(' expression ')'
        -> ^(SIZEOF expression)
    |   SIZEOF '(' type ')'
        -> ^(SIZEOF type)

    ;
    
qualifiedIdentExpression
        // The qualified identifier itself is the starting point for this rule.
    :   (   qualifiedIdentifier
            ->  qualifiedIdentifier
        )
        // And now comes the stuff that may follow the qualified identifier.
        (   arguments
            ->  ^(METHOD_CALL qualifiedIdentifier arguments)
        |   outerDot=DOT
            (   templateInstantiation
                (   s=SUPER arguments
                    ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"]
                            qualifiedIdentifier templateInstantiation arguments)
                |   SUPER innerDot=DOT IDENT arguments
                    ->  ^(METHOD_CALL ^($innerDot ^($outerDot qualifiedIdentifier SUPER) IDENT)
                            templateInstantiation arguments)
                |   IDENT arguments
                    ->  ^(METHOD_CALL ^($outerDot qualifiedIdentifier IDENT) templateInstantiation arguments)
                )
            |   THIS
                ->  ^($outerDot qualifiedIdentifier THIS)
            |   s=SUPER arguments
                ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] qualifiedIdentifier arguments)
            )
        )?
    ;

newExpression
    :   NEW
        (
            domainExpression arguments?
            ->  ^(NEW_EXPRESSION arguments? domainExpression)
        |   constructorType arguments
            -> ^(NEW constructorType arguments)
		)
    ;
    
/*newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
    |   '['! expression ']'! ('['! expression ']'!)* arrayDeclaratorList?
    ;*/

arguments
    :   lp='(' expressionList? ')'
        ->  ^(ARGUMENT_LIST[$lp, "ARGUMENT_LIST"] expressionList?)
    ;

literal 
    :   HEX_LITERAL
    |   OCTAL_LITERAL
    |   DECIMAL_LITERAL
    |   FLOATING_POINT_LITERAL
    |   CHARACTER_LITERAL
    |   STRING_LITERAL
    |   TRUE
    |   FALSE
    |   NULL
    ;

// LEXER

HEX_LITERAL : '0' ('x'|'X') HEX_DIGIT+ INTEGER_TYPE_SUFFIX? ;

DECIMAL_LITERAL : ('0' | '1'..'9' '0'..'9'*) INTEGER_TYPE_SUFFIX? ;

OCTAL_LITERAL : '0' ('0'..'7')+ INTEGER_TYPE_SUFFIX? ;

ARRAY_DIMENSION :  ('1'..'6')('d'|'D') ;

fragment
HEX_DIGIT : ('0'..'9'|'a'..'f'|'A'..'F') ;

fragment
INTEGER_TYPE_SUFFIX : ('l'|'L') ;

FLOATING_POINT_LITERAL
    :   ('0'..'9')+ 
        (
            DOT ('0'..'9')* EXPONENT? FLOAT_TYPE_SUFFIX?
        |   EXPONENT FLOAT_TYPE_SUFFIX?
        |   FLOAT_TYPE_SUFFIX
        )
    |   DOT ('0'..'9')+ EXPONENT? FLOAT_TYPE_SUFFIX?
    ;

fragment
EXPONENT : ('e'|'E') ('+'|'-')? ('0'..'9')+ ;

fragment
FLOAT_TYPE_SUFFIX : ('f'|'F'|'d'|'D') ;

CHARACTER_LITERAL
    :   '\'' ( ESCAPE_SEQUENCE | ~('\''|'\\') ) '\''
    ;

STRING_LITERAL
    :  '"' ( ESCAPE_SEQUENCE | ~('\\'|'"') )* '"'
    ;

fragment
ESCAPE_SEQUENCE
    :   '\\' ('b'|'t'|'n'|'f'|'r'|'\"'|'\''|'\\')
    |   UNICODE_ESCAPE
    |   OCTAL_ESCAPE
    ;

fragment
OCTAL_ESCAPE
    :   '\\' ('0'..'3') ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7') ('0'..'7')
    |   '\\' ('0'..'7')
    ;

fragment
UNICODE_ESCAPE
    :   '\\' 'u' HEX_DIGIT HEX_DIGIT HEX_DIGIT HEX_DIGIT
    ;

IDENT
    :   CHARJ_ID_START (CHARJ_ID_PART)*
    ;

fragment
CHARJ_ID_START
    :  '\u0024'
    |  '\u0041'..'\u005a'
    |  '\u005f'
    |  '\u0061'..'\u007a'
    |  '\u00c0'..'\u00d6'
    |  '\u00d8'..'\u00f6'
    |  '\u00f8'..'\u00ff'
    |  '\u0100'..'\u1fff'
    |  '\u3040'..'\u318f'
    |  '\u3300'..'\u337f'
    |  '\u3400'..'\u3d2d'
    |  '\u4e00'..'\u9fff'
    |  '\uf900'..'\ufaff'
    ;

fragment
CHARJ_ID_PART
    :  CHARJ_ID_START
    |  '\u0030'..'\u0039'
    ;

WS  :  (' '|'\r'|'\t'|'\u000C'|'\n') 
    {   
        $channel = HIDDEN;
    }
    ;

fragment
EMBED_BLOCK
    :   '{' ( options {greedy=false;} : EMBED_BLOCK | . )* '}'
    ;

COMMENT
    :   '/*' ( options {greedy=false;} : . )* '*/'
    {   
        $channel = HIDDEN;
    }
    ;

LINE_COMMENT
    : ('//'|'#') ~('\n'|'\r')* '\r'? '\n'
    {   
        $channel = HIDDEN;
    }
    ;

