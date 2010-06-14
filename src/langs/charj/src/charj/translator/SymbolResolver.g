
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
            }
            )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .*))? type ^(IDENT .*))
        {
            $IDENT.symbol.type = $type.sym;
        }
    ;

expression returns [ClassSymbol type]
    :   ^(EXPR expr) { $type = $expr.type; }
    ;

// TODO: fill out all the different cases here
expr returns [ClassSymbol type]
    :   IDENT {
            $IDENT.symbol = $IDENT.scope.resolveVariable($IDENT.text);
            if ($IDENT.symbol != null) {
                $type = $IDENT.symbol.type;
            } else {
                System.out.println("Couldn't resolve type: " + $IDENT.text);
            }
        }
    |   THIS {
            $THIS.symbol = symtab.getEnclosingClass($THIS.symbol.scope);
            $type = (ClassSymbol)$THIS.symbol;
        }
    |   SUPER {
            $SUPER.symbol = symtab.getEnclosingClass($SUPER.symbol.scope).superClass;
            $type = (ClassSymbol)$SUPER.symbol;
        }
    |   ^((DOT|ARROW) e=expr id=.) {
            ClassSymbol cxt = $e.type;
            Symbol s;
            if (cxt == null) {
                s = null;
                System.out.println("No expression context: " + $e.text);
            } else {
                s = cxt.resolveVariable($id.getText());
            }
            if (s != null) {
                $type = s.type;
            } else {
                System.out.println("Couldn't resolve access " + $id.getText());
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
    System.out.println("direct scope: " + scope);
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
