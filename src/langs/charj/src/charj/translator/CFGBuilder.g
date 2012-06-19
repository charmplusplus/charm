

tree grammar CFGBuilder;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
package charj.translator;
import java.util.Iterator;
}

@members {
    SymbolTable symtab;
    Scope currentScope;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;

    public CFGBuilder(TreeNodeStream input, SymbolTable symtab) {
        this(input);
        this.symtab = symtab;
        this.currentScope = symtab.getDefaultPkg();
    }
}


topdown
    :   enterClass
    |   enterMethod
    |   expression
    |   assignment
    ;

bottomup
    :   exitClass
    |   exitMethod
    ;

enterMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL)
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            type IDENT .*) { currentMethod = (MethodSymbol)$IDENT.def; }
    |   ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL)
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*) { currentMethod = (MethodSymbol)$IDENT.def; }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL)
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            type IDENT .*) { currentMethod = null; }
    |   ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL)
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*) { currentMethod = null; }
    ;

enterClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | DIVCON_METHOD_DECL |  PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        { currentClass = (ClassSymbol)$IDENT.def.type; }
    ;

exitClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | DIVCON_METHOD_DECL |  PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        { currentClass = null; }
    ;

expression returns [Type type]
    :   ^(EXPR expr)
    ;

assignment
    :   ^(ASSIGNMENT IDENT expr)
    ;

binary_op
    : ASSIGNMENT
    | PLUS_EQUALS
    | MINUS_EQUALS
    | TIMES_EQUALS
    | DIVIDE_EQUALS
    | AND_EQUALS
    | OR_EQUALS
    | POWER_EQUALS
    | MOD_EQUALS
    | '>>>='
    | '>>='
    | '<<='
    | '?'
    | OR
    | AND
    | BITWISE_OR
    | POWER
    | BITWISE_AND
    | EQUALS
    | NOT_EQUALS
    | INSTANCEOF
    | LTE
    | GTE
    | '>>>'
    | '>>'
    | GT
    | '<<'
    | LT
    | PLUS
    | MINUS
    | TIMES
    | DIVIDE
    | MOD
    ;

unary_op
    : UNARY_PLUS
    | UNARY_MINUS
    | PRE_INC
    | PRE_DEC
    | POST_INC
    | POST_DEC
    | TILDE
    | NOT
    | CAST_EXPR
    ;


expr returns [Type type]
    :   ^(binary_op e1=expr e2=expr)
    |   ^(unary_op e1=expr) 
    |   primaryExpression 
    ;

primaryExpression
    :   IDENT 
    |   THIS 
    |   SUPER 
    |   ^((DOT | ARROW) e=expr
            (   IDENT 
            |   THIS 
            |   SUPER 
            ))
    |   ^(PAREN_EXPR expression)
    |   ^(METHOD_CALL e=expr .*)
    |   ^(ENTRY_METHOD_CALL e=expr .*)
    |   ^(THIS_CONSTRUCTOR_CALL .*)
    |   ^(SUPER_CONSTRUCTOR_CALL .*)
    |   ^(ARRAY_ELEMENT_ACCESS expr expression)
    |   literal
    |   ^(NEW t=type .*)
    |   GETNUMPES
    |   GETNUMNODES
    |   GETMYPE
    |   GETMYNODE
    |   GETMYRANK
	|	THISINDEX
	|	THISPROXY
    ;

literal
    :   (HEX_LITERAL | OCTAL_LITERAL | DECIMAL_LITERAL)
    |   FLOATING_POINT_LITERAL
    |   CHARACTER_LITERAL
    |   STRING_LITERAL
    |   (TRUE | FALSE)
    |   NULL
    ;

literalVal
    :   (HEX_LITERAL | OCTAL_LITERAL | DECIMAL_LITERAL)
    |   FLOATING_POINT_LITERAL
    |   CHARACTER_LITERAL
    |   STRING_LITERAL
    |   (TRUE | FALSE)
    |   NULL
    ;

type
    :   VOID
    |   ^(SIMPLE_TYPE t=. .*)
    |   ^(OBJECT_TYPE 
            ^(QUALIFIED_TYPE_IDENT (^(IDENT (^(TEMPLATE_INST
                (t1=type | lit1=literalVal )*))?))+) .*)
    |   ^(REFERENCE_TYPE 
            ^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)
    |   ^(PROXY_TYPE 
            ^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)
	|	^(ARRAY_SECTION_TYPE 
			^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)
	|	^(MESSAGE_TYPE 
			^(QUALIFIED_TYPE_IDENT (^(IDENT .*))+) .*)
    |   ^(POINTER_TYPE 
            ^(QUALIFIED_TYPE_IDENT (^(IDENT (^(TEMPLATE_INST
                        (t1=type | lit1=literalVal)*))?))+) .*)
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

