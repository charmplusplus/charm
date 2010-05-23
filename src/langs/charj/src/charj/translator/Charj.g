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
    ENTRY                   = 'entry'           ;

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
    SIZEOF                  = 'sizeof'          ;
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
    CAST_EXPR;
    CATCH_CLAUSE_LIST;
    CLASS_CONSTRUCTOR_CALL;
    CLASS_INSTANCE_INITIALIZER;
    CLASS_STATIC_INITIALIZER;
    CLASS_TOP_LEVEL_SCOPE;
    CONSTRUCTOR_DECL;
    ENUM_TOP_LEVEL_SCOPE;
    EXPR;
    EXTENDS_BOUND_LIST;
    EXTENDS_CLAUSE;
    FOR_EACH;
    FOR_EXPR;
    FOR_UPDATE;
    FORMAL_PARAM_LIST;
    FORMAL_PARAM_STD_DECL;
    FORMAL_PARAM_VARARG_DECL;
    FUNCTION_METHOD_DECL;
    GENERIC_TYPE_ARG_LIST;
    GENERIC_TYPE_PARAM_LIST;
    INTERFACE_TOP_LEVEL_SCOPE;
    IMPLEMENTS_CLAUSE;
    LABELED_STATEMENT;
    LOCAL_MODIFIER_LIST;
    CHARJ_SOURCE;
    METHOD_CALL;
    MODIFIER_LIST;
    PAREN_EXPR;
    POST_DEC;
    POST_INC;
    PRE_DEC;
    PRE_INC;
    QUALIFIED_TYPE_IDENT;
    STATIC_ARRAY_CREATOR;
    SUPER_CONSTRUCTOR_CALL;
    THIS_CONSTRUCTOR_CALL;
    TYPE;
    UNARY_MINUS;
    UNARY_PLUS;
    PRIMITIVE_VAR_DECLARATION;
    OBJECT_VAR_DECLARATION;
    VAR_DECLARATOR;
    VAR_DECLARATOR_LIST;
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
        importDeclaration* 
        typeDeclaration
    ;

packageDeclaration
    :   'package'^ qualifiedIdentifier ';'!  
    ;

importDeclaration
    :   'import'^ qualifiedIdentifier '.*'? ';'!
    ;

typeDeclaration
    :   classDefinition
    |   interfaceDefinition
    |   enumDefinition
    |   chareDefinition
    ;

classDefinition
    :   'public'? 'class' IDENT ('extends' type)? ('implements' typeList)? '{'
            classScopeDeclaration*
        '}' ';'?
        -> ^(TYPE 'class' IDENT ^('extends' type)? ^('implements' typeList)? classScopeDeclaration*)
    ;

chareType
    :   'chare'
    |   'group'
    |   'nodegroup'
    |   'chare_array' '[' ARRAY_DIMENSION ']' -> ^('chare_array' ARRAY_DIMENSION)
    ;

chareDefinition
    :   'public'? chareType IDENT ('extends' type)? ('implements' typeList)? '{'
            classScopeDeclaration*
        '}' ';'?
        -> ^(TYPE chareType IDENT ^('extends' type)? ^('implements' typeList)? classScopeDeclaration*)
    ;

interfaceDefinition
    :   'interface' IDENT ('extends' typeList)?  '{'
            interfaceScopeDeclaration*
        '}' ';'?
        -> ^('interface' IDENT ^('extends' typeList)? interfaceScopeDeclaration*)
    ;

enumDefinition
    :   'enum' IDENT ('implements' typeList)? '{'
            enumConstants ','? ';' classScopeDeclaration*
        '}' ';'?
        -> ^('enum' IDENT ^('implements' typeList)? enumConstants classScopeDeclaration*)
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

classScopeDeclaration
    :   modifierList?
        (   genericTypeParameterList?
            (   type IDENT formalParameterList arrayDeclaratorList? (block | ';')
                ->  ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? type IDENT
                    formalParameterList arrayDeclaratorList? block?)
            |   ident=IDENT formalParameterList block
                ->  ^(CONSTRUCTOR_DECL[$ident, "CONSTRUCTOR_DECL"] modifierList? genericTypeParameterList? IDENT
                        formalParameterList block)
            )
        |   simpleType classFieldDeclaratorList ';'
            ->  ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType classFieldDeclaratorList)
        |   objectType classFieldDeclaratorList ';'
            ->  ^(OBJECT_VAR_DECLARATION modifierList? objectType classFieldDeclaratorList)
        )
    ;

interfaceScopeDeclaration
    :   modifierList?
        (   genericTypeParameterList?
            (   type IDENT formalParameterList arrayDeclaratorList? ';'
                ->  ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList?
                        type IDENT formalParameterList arrayDeclaratorList?)
            )
        |   simpleType interfaceFieldDeclaratorList ';'
            ->  ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType interfaceFieldDeclaratorList)
        |   objectType interfaceFieldDeclaratorList ';'
            ->  ^(OBJECT_VAR_DECLARATION modifierList? objectType interfaceFieldDeclaratorList)        
        )
    ;

classFieldDeclaratorList
    :   classFieldDeclarator (',' classFieldDeclarator)*
        ->  ^(VAR_DECLARATOR_LIST classFieldDeclarator+)
    ;

classFieldDeclarator
    :   variableDeclaratorId ('=' variableInitializer)?
        ->  ^(VAR_DECLARATOR variableDeclaratorId variableInitializer?)
    ;

interfaceFieldDeclaratorList
    :   interfaceFieldDeclarator (',' interfaceFieldDeclarator)*
        ->  ^(VAR_DECLARATOR_LIST interfaceFieldDeclarator+)
    ;

interfaceFieldDeclarator
    :   variableDeclaratorId '=' variableInitializer
        ->  ^(VAR_DECLARATOR variableDeclaratorId variableInitializer)
    ;


variableDeclaratorId
    :   IDENT^ arrayDeclaratorList?
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayDeclarator
    :   '[' ']'
        ->  ARRAY_DECLARATOR
    ;

arrayDeclaratorList
    :   arrayDeclarator+
        ->  ^(ARRAY_DECLARATOR_LIST arrayDeclarator+)   
    ;

arrayInitializer
    :   lc='{' (variableInitializer (',' variableInitializer)* ','?)? '}'
        ->  ^(ARRAY_INITIALIZER[$lc, "ARRAY_INITIALIZER"] variableInitializer*)
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
    :   e='extends' type ('&' type)*
        ->  ^(EXTENDS_BOUND_LIST[$e, "EXTENDS_BOUND_LIST"] type+)
    ;

modifierList
    :   modifier+
        ->  ^(MODIFIER_LIST modifier+)
    ;

modifier
    :   'public'
    |   'protected'
    |   'entry'
    |   'private'
    |   'abstract'
    |   'native'
    |   localModifier
    ;

localModifierList
    :   localModifier+
        -> ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

localModifier
    :   'final'
    |   'static'
    |   'volatile'
    ;

type
    :   simpleType
    |   objectType
    |   'void'
    ;

simpleType
    :   primitiveType arrayDeclaratorList?
        ->  ^(TYPE primitiveType arrayDeclaratorList?)  
    ;

objectType
    :   qualifiedTypeIdent arrayDeclaratorList?
        ->  ^(TYPE qualifiedTypeIdent arrayDeclaratorList?)
    ;

qualifiedTypeIdent
    :   typeIdent ('.' typeIdent)*
        ->  ^(QUALIFIED_TYPE_IDENT typeIdent+) 
    ;

typeIdent
    :   IDENT^ genericTypeArgumentList?
    ;

primitiveType
    :   'boolean'
    |   'char'
    |   'byte'
    |   'short'
    |   'int'
    |   'long'
    |   'float'
    |   'double'
    ;

genericTypeArgumentList
    :   lt='<' genericTypeArgument (',' genericTypeArgument)* genericTypeListClosing
        ->  ^(GENERIC_TYPE_ARG_LIST[$lt, "GENERIC_TYPE_ARG_LIST"] genericTypeArgument+)
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
            formalParameterStandardDecl (',' formalParameterStandardDecl)* (',' formalParameterVarArgDecl)? 
            ->  ^(FORMAL_PARAM_LIST[$lp, "FORMAL_PARAM_LIST"] formalParameterStandardDecl+ formalParameterVarArgDecl?) 
            // Contains a variable argument declaration only.
        |   formalParameterVarArgDecl
            ->  ^(FORMAL_PARAM_LIST[$lp, "FORMAL_PARAM_LIST"] formalParameterVarArgDecl) 
            // Contains nothing.
        |   ->  ^(FORMAL_PARAM_LIST[$lp, "FORMAL_PARAM_LIST"]) 
        )
        ')'
    ;

formalParameterStandardDecl
    :   localModifierList? type variableDeclaratorId
        ->  ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId)
    ;

formalParameterVarArgDecl
    :   localModifierList? type '...' variableDeclaratorId
        ->  ^(FORMAL_PARAM_VARARG_DECL localModifierList? type variableDeclaratorId)
    ;

qualifiedIdentifier
    :   (   IDENT
            ->  IDENT
        )
        (   '.' ident=IDENT
            ->  ^('.' $qualifiedIdentifier $ident)
        )*
    ;

block
    :   lc='{' blockStatement* '}'
        ->  ^(BLOCK[$lc, "BLOCK"] blockStatement*)
    ;

blockStatement
    :   localVariableDeclaration ';'!
    |   statement
    ;

localVariableDeclaration
    :   localModifierList? simpleType classFieldDeclaratorList
        ->  ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType classFieldDeclaratorList)
    |   localModifierList? objectType classFieldDeclaratorList
        ->  ^(OBJECT_VAR_DECLARATION localModifierList? objectType classFieldDeclaratorList)
    ;
        
statement
    :   block
    |   'assert' expr1=expression 
        (   ':' expr2=expression ';'
            ->  ^('assert' $expr1 $expr2)
        |   ';'
            ->  ^('assert' $expr1)
        )
    |   'if' parenthesizedExpression ifStat=statement 
        (   'else' elseStat=statement
            ->  ^('if' parenthesizedExpression $ifStat $elseStat)
        |
            ->  ^('if' parenthesizedExpression $ifStat)
        )   
    |   f='for' '('
        (   forInit? ';' expression? ';' expressionList? ')' statement
            -> ^($f forInit? FOR_EXPR expression? FOR_UPDATE expressionList? statement)
        |   localModifierList? type IDENT ':' expression ')' statement
            -> ^(FOR_EACH[$f, "FOR_EACH"] localModifierList? type IDENT expression statement)
        )
    |   'while' parenthesizedExpression statement
        ->  ^('while' parenthesizedExpression statement)
    |   'do' statement 'while' parenthesizedExpression ';'
        ->  ^('do' statement parenthesizedExpression)
    |   'switch' parenthesizedExpression '{' switchCaseLabel* '}'
        ->  ^('switch' parenthesizedExpression switchCaseLabel*)
    |   'return' expression? ';'
        ->  ^('return' expression?)
    |   'throw' expression ';'
        ->  ^('throw' expression)
    |   'break' IDENT? ';'
        ->  ^('break' IDENT?)
    |   'continue' IDENT? ';'
        ->  ^('continue' IDENT?)
    |   IDENT ':' statement
        ->  ^(LABELED_STATEMENT IDENT statement)
    |   'embed' STRING_LITERAL EMBED_BLOCK
        ->  ^('embed' STRING_LITERAL EMBED_BLOCK)
    |   expression ';'!
    |   ';' // Preserve empty statements.
    ;           
        

switchCaseLabel
    :   'case'^ expression ':'! blockStatement*
    |   'default'^ ':'! blockStatement*
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
    
expressionList
    :   expression (','! expression)*
    ;

expression
    :   assignmentExpression
        ->  ^(EXPR assignmentExpression)
    ;

assignmentExpression
    :   conditionalExpression 
        (   (   '='^
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
        (   (   '=='^
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
        // ... and than the optional things that may follow a primary
        // expression 0 or more times.
        (   outerDot='.'                 
            // Note: generic type arguments are only valid for method calls,
            // i.e. if there is an argument list
            (   (   genericTypeArgumentList?  
                    IDENT
                    ->  ^($outerDot $postfixedExpression IDENT)
                ) 
                (   arguments
                    ->  ^(METHOD_CALL $postfixedExpression genericTypeArgumentList? arguments)
                )?
            |   'this'
                ->  ^($outerDot $postfixedExpression 'this')
            |   s='super' arguments
                ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] $postfixedExpression arguments)
            |   (   'super' innerDot='.' IDENT
                    ->  ^($innerDot ^($outerDot $postfixedExpression 'super') IDENT)
                )
                (   arguments
                    ->  ^(METHOD_CALL $postfixedExpression arguments)
                )?
            )
        |   '[' expression ']'
            ->  ^(ARRAY_ELEMENT_ACCESS $postfixedExpression expression)
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
    |   genericTypeArgumentList 
        (   s='super'
            (   arguments
                ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] genericTypeArgumentList arguments)
            |   IDENT arguments
                ->  ^(METHOD_CALL ^('.' 'super' IDENT) genericTypeArgumentList arguments)
            )
        |   IDENT arguments
            ->  ^(METHOD_CALL IDENT genericTypeArgumentList arguments)
        |   t='this' arguments
            ->  ^(THIS_CONSTRUCTOR_CALL[$t, "THIS_CONSTRUCTOR_CALL"] genericTypeArgumentList arguments)
        )
    |   (   'this'
            ->  'this'
        )
        (   arguments
            ->  ^(THIS_CONSTRUCTOR_CALL[$t, "THIS_CONSTRUCTOR_CALL"] arguments)
        )?
    |   s='super' arguments
        ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] arguments)
    |   (   'super' '.' IDENT
        )
        (   arguments
            ->  ^(METHOD_CALL ^('.' 'super' IDENT) arguments)
        |   ->  ^('.' 'super' IDENT)
        )
    ;
    
qualifiedIdentExpression
        // The qualified identifier itself is the starting point for this rule.
    :   (   qualifiedIdentifier
            ->  qualifiedIdentifier
        )
        // And now comes the stuff that may follow the qualified identifier.
        (   arguments
            ->  ^(METHOD_CALL qualifiedIdentifier arguments)
        |   outerDot='.'
            (   genericTypeArgumentList 
                (   s='super' arguments
                    ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"]
                            qualifiedIdentifier genericTypeArgumentList arguments)
                |   'super' innerDot='.' IDENT arguments
                    ->  ^(METHOD_CALL ^($innerDot ^($outerDot qualifiedIdentifier 'super') IDENT)
                            genericTypeArgumentList arguments)
                |   IDENT arguments
                    ->  ^(METHOD_CALL ^($outerDot qualifiedIdentifier IDENT) genericTypeArgumentList arguments)
                )
            |   'this'
                ->  ^($outerDot qualifiedIdentifier 'this')
            |   s='super' arguments
                ->  ^(SUPER_CONSTRUCTOR_CALL[$s, "SUPER_CONSTRUCTOR_CALL"] qualifiedIdentifier arguments)
            )
        )?
    ;

newExpression
    :   n='new'
        (   primitiveType newArrayConstruction          // new static array of primitive type elements
            ->  ^(STATIC_ARRAY_CREATOR[$n, "STATIC_ARRAY_CREATOR"] primitiveType newArrayConstruction)
        |   genericTypeArgumentList? qualifiedTypeIdent
                newArrayConstruction                // new static array of object type reference elements
            ->  ^(STATIC_ARRAY_CREATOR[$n, "STATIC_ARRAY_CREATOR"] genericTypeArgumentList? qualifiedTypeIdent newArrayConstruction)
        )
    ;
    
newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
    |   '['! expression ']'! ('['! expression ']'!)* arrayDeclaratorList?
    ;

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
    |   'true'
    |   'false'
    |   'null'
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
            '.' ('0'..'9')* EXPONENT? FLOAT_TYPE_SUFFIX?
        |   EXPONENT FLOAT_TYPE_SUFFIX?
        |   FLOAT_TYPE_SUFFIX
        )
    |   '.' ('0'..'9')+ EXPONENT? FLOAT_TYPE_SUFFIX?
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

