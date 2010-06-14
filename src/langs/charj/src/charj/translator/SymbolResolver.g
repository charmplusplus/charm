
tree grammar SymbolResolver;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
package charj.translator;
}

@members {
    SymbolTable symtab;
    Scope currentScope;
    ClassSymbol currentClass;

    public SymbolResolver(TreeNodeStream input, SymbolTable symtab) {
        this(input);
        this.symtab = symtab;
        this.currentScope = symtab.getDefaultPkg();
    }
}


topdown
    :   enterClass
    |   enterMethod
    |   varDeclaration
    |   expression
    ;

enterMethod
@init {
boolean entry = false;
}
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            type IDENT .*)
        {
            $IDENT.symbol.type = $type.sym;
        }
    |   ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*)
        {
            $IDENT.symbol.type = (ClassSymbol)$IDENT.symbol.scope;
        }
    ;

enterClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))? .*)
        {
            $IDENT.symbol.type = (ClassSymbol)$IDENT.symbol;
        }
    ;

varDeclaration
    :   ^((PRIMITIVE_VAR_DECLARATION | OBJECT_VAR_DECLARATION)
            (^(MODIFIER_LIST .*))? type
            ^(VAR_DECLARATOR_LIST (^(VAR_DECLARATOR ^(IDENT .*) .*)
            {
                $IDENT.symbol.type = $type.sym;
                //System.out.println("Resolved type of variable " + $IDENT.text + ": " + $IDENT.symbol.type + ", symbol is " + $IDENT.symbol);
            }
            )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .*))? type ^(IDENT .*))
        {
            $IDENT.symbol.type = $type.sym;
        }
    ;

expression returns [ClassSymbol type]
    :   ^(EXPR expr) {
            $type = $expr.type;
            $EXPR.symbol = new Symbol(symtab, "EXPR", $type);
        }
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


expr returns [ClassSymbol type]
    :   ^(binary_op e1=expr e2=expr) {
            // TODO: proper type promotion rules
            $type = $e1.type;
            $binary_op.start.symbol = new Symbol(symtab, $binary_op.text, $type);
        }
    |   ^(unary_op e1=expr) {
            $type = $e1.type;
            $unary_op.start.symbol = new Symbol(symtab, $unary_op.text, $type);
        }
    |   primaryExpression {
            $type = $primaryExpression.type;
        }
    ;

// TODO: fill out all the different cases here
primaryExpression returns [ClassSymbol type]
@init{
    String memberText = "";
    CharjAST memberNode = null;
    CharjAST parentNode = null;
}
    :   IDENT {
            $IDENT.symbol = $IDENT.scope.resolve($IDENT.text);
            if ($IDENT.symbol != null) {
                $type = $IDENT.symbol.type;
                //System.out.println("Resolved type of " + $IDENT.text + ": " + $type + ", symbol is " + $IDENT.symbol);
            } else {
                System.out.println("Couldn't resolve IDENT type: " + $IDENT.text);
            }
        }
    |   THIS {
            $THIS.symbol = symtab.getEnclosingClass($THIS.scope);
            $type = (ClassSymbol)$THIS.symbol;
        }
    |   SUPER {
            $SUPER.symbol = symtab.getEnclosingClass($SUPER.scope).superClass;
            $type = (ClassSymbol)$SUPER.symbol;
        }
    |   ^((DOT { parentNode = $DOT; } | ARROW { parentNode = $ARROW; } ) e=expr
            (   IDENT { memberNode = $IDENT; memberText = $IDENT.text; }
            |   THIS { memberNode = $THIS; memberText = "this"; }
            |   SUPER { memberNode = $SUPER; memberText = "super"; }
            ))
        {
            ClassSymbol cxt = $e.type;
            Symbol s;
            if (cxt == null) {
                s = null;
                /*System.out.println("No expression context: " + $e.text);*/
            } else {
                //System.out.println("Expression context is: " + cxt + " for symbol named " + memberText);
                if (memberText.equals("this")) s = cxt;
                else if (memberText.equals("super")) s = cxt.superClass;
                else s = cxt.resolve(memberText);
            }
            if (s != null) {
                $type = s.type;
                memberNode.symbol = s;
                parentNode.symbol = s;
            } else {
                System.out.println("Couldn't resolve access " + memberText);
            }
        }
    |   ^(PAREN_EXPR expression) {
            $type = $expression.type;
        }
    |   ^((METHOD_CALL|ENTRY_METHOD_CALL) e=expr .*) {
            $type = $e.type; // Type of a method is its return type.
        }
    |   ^(THIS_CONSTRUCTOR_CALL .*) {
            // TODO: fill in
        }
    |   ^(SUPER_CONSTRUCTOR_CALL .*) {
            // TODO: fill in
        }
    |   ^(ARRAY_ELEMENT_ACCESS expr expression)
    |   literal {
            $type = $literal.type;
        }
    |   ^(NEW t=type .*) {
            $type = $t.sym;
        }
    |   GETNUMPES
    |   GETNUMNODES
    |   GETMYPE
    |   GETMYNODE
    |   GETMYRANK
    ;

literal returns [ClassSymbol type]
    :   (HEX_LITERAL | OCTAL_LITERAL | DECIMAL_LITERAL) {
            $type = symtab.resolveBuiltinType("int");
        }
    |   FLOATING_POINT_LITERAL {
            $type = symtab.resolveBuiltinType("double");
        }
    |   CHARACTER_LITERAL {
            $type = symtab.resolveBuiltinType("char");
        }
    |   STRING_LITERAL {
            $type = symtab.resolveBuiltinType("string");
        }
    |   (TRUE | FALSE) {
            $type = symtab.resolveBuiltinType("boolean");
        }
    |   NULL {
            $type = symtab.resolveBuiltinType("null");
        }
    ;

type returns [ClassSymbol sym]
@init {
    String typeText = "";
    CharjAST head = null;
    Scope scope = null;
}
@after {
    typeText = typeText.substring(1);
    //System.out.println("type string: " + typeText);
    //System.out.println("direct scope: " + scope);
    $start.symbol = scope.resolveType(typeText);
    $sym = (ClassSymbol)$start.symbol;
    if ($sym == null) System.out.println("Couldn't resolve type: " + typeText);
}
    :   VOID {
            scope = $VOID.scope;
            typeText = ".void";
        }
    |   ^(SIMPLE_TYPE t=. {
            scope = $SIMPLE_TYPE.scope;
            typeText += "." + $t.getText();
        } .*)
    |   ^(OBJECT_TYPE { scope = $OBJECT_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeText += "." + $IDENT.text;} .*))+) .*)
    |   ^(REFERENCE_TYPE { scope = $REFERENCE_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT  {typeText += "." + $IDENT.text;} .*))+) .*)
    |   ^(PROXY_TYPE { scope = $PROXY_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeText += "." + $IDENT.text;} .*))+) .*)
    |   ^(POINTER_TYPE { scope = $POINTER_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeText += "." + $IDENT.text;} .*))+) .*)
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;
