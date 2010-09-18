
tree grammar SymbolResolver;

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
    |   assignment
    ;

bottomup
    :   exitClass
    ;

enterMethod
@init {
boolean entry = false;
}
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            type IDENT .*)
    |   ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*)
    ;

enterClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | DIVCON_METHOD_DECL |  PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = (ClassSymbol)$IDENT.def.type;
        }
    ;

exitClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | DIVCON_METHOD_DECL |  PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        { currentClass = null; }
    ;

varDeclaration
    :   ^((PRIMITIVE_VAR_DECLARATION | OBJECT_VAR_DECLARATION)
            (^(MODIFIER_LIST .*))? type
            ^(VAR_DECLARATOR_LIST (^(VAR_DECLARATOR ^(IDENT .*) .*)
            {
                //System.out.println("Resolved type of variable " + $IDENT.text + ": " +
                //    $IDENT.def.type + ", symbol is " + $IDENT.def);
                if (currentClass != null) {
                    ClassSymbol declType = null;
                    if ($type.sym instanceof ClassSymbol) {
                        declType = (ClassSymbol)$type.sym;
                    } else if ($type.sym instanceof ProxyType) {
                        declType = (ClassSymbol)((ProxyType)$type.sym).baseType;
                    }

                    if (declType != null) {
                        //System.out.println("Looking to extern " + $IDENT.text + " as " + declType);
                        if (declType.isChare && declType != currentClass) {
                            //System.out.println("extern added");
                            currentClass.addExtern(declType.getTypeName());
                        }
                    }
                }
            }
            )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .*))? type ^(IDENT .*))
    ;

expression returns [Type type]
    :   ^(EXPR expr) {
            $type = $expr.type;
            $EXPR.def = new Symbol(symtab, "EXPR", $type);
            $EXPR.symbolType = $type;
        }
    ;

assignment
    :   ^(ASSIGNMENT IDENT expr) {
            //System.out.println("Found candidate assignment to " + $IDENT.text);
            if ($IDENT.def instanceof VariableSymbol) {
                VariableSymbol vs = (VariableSymbol)$IDENT.def;
                if (vs.isReadOnly && !(currentClass.isMainChare &&
                    $IDENT.hasParentOfType(CharjParser.ENTRY_CONSTRUCTOR_DECL))) {
                    System.out.println("Warning: assignment to readonly variable " +
                        $IDENT.text + " on line " + $IDENT.getLine());
               }
            }
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


expr returns [Type type]
    :   ^(binary_op e1=expr e2=expr) {
            // TODO: proper type promotion rules
            $type = $e1.type;
            $binary_op.start.def = new Symbol(symtab, $binary_op.text, $type);
            $binary_op.start.symbolType = $type;
        }
    |   ^(unary_op e1=expr) {
            $type = $e1.type;
            $unary_op.start.def = new Symbol(symtab, $unary_op.text, $type);
            $unary_op.start.symbolType = $type;
        }
    |   primaryExpression {
            $type = $primaryExpression.type;
        }
    ;

// TODO: fill out all the different cases here
// TODO: warn on readonly assigment outside of ctor
primaryExpression returns [Type type]
@init{
    String memberText = "";
    CharjAST memberNode = null;
    CharjAST parentNode = null;
}
    :   IDENT {
            $IDENT.def = $IDENT.scope.resolve($IDENT.text);
            if ($IDENT.def != null) {
                $type = $IDENT.def.type;
                $IDENT.symbolType = $type;
                //System.out.println("Resolved type of " + $IDENT.text + ": " + $type + ", symbol is " + $IDENT.def);
            } else {
                System.out.println("Couldn't resolve IDENT type: " + $IDENT.text);
            }
        }
    |   THIS {
            $THIS.def = symtab.getEnclosingClass($THIS.scope);
            $type = $THIS.def.type;
            $THIS.symbolType = $type;
        }
    |   SUPER {
            $SUPER.def = symtab.getEnclosingClass($SUPER.scope).superClass;
            $type = $SUPER.def.type;
            $SUPER.symbolType = $type;
        }
    |   ^((DOT { parentNode = $DOT; } | ARROW { parentNode = $ARROW; } ) e=expr
            (   IDENT { memberNode = $IDENT; memberText = $IDENT.text; }
            |   THIS { memberNode = $THIS; memberText = "this"; }
            |   SUPER { memberNode = $SUPER; memberText = "super"; }
            ))
        {
            Type et = $e.type;
            if (et instanceof ProxyType) et = ((ProxyType)et).baseType;
            if (et instanceof PointerType) et = ((PointerType)et).baseType;
            ClassSymbol cxt = (ClassSymbol)et;
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
                memberNode.def = s;
                memberNode.symbolType = $type;
                parentNode.def = s;
                parentNode.symbolType = $type;
            } else {
                System.out.println("Couldn't resolve access " + memberText);
            }
        }
    |   ^(PAREN_EXPR expression) {
            $type = $expression.type;
        }
    |   ^(METHOD_CALL e=expr .*) {
            $type = $e.type; // Type of a method is its return type.
            $METHOD_CALL.symbolType = $type;
        }
    |   ^(ENTRY_METHOD_CALL e=expr .*) {
            $type = $e.type; // Type of a method is its return type.
            $ENTRY_METHOD_CALL.symbolType = $type;
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
    |   GETNUMPES {
            $type = symtab.resolveBuiltinType("int");
        }
    |   GETNUMNODES {
            $type = symtab.resolveBuiltinType("int");
        }
    |   GETMYPE {
            $type = symtab.resolveBuiltinType("int");
        }
    |   GETMYNODE {
            $type = symtab.resolveBuiltinType("int");
        }
    |   GETMYRANK {
            $type = symtab.resolveBuiltinType("int");
        }
	|	THISINDEX {
			$type = symtab.resolveBuiltinType("int");
		}
	|	THISPROXY {
			// TODO
		}
    ;

literal returns [Type type]
@after { $start.symbolType = $type; }
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

literalVal returns [Type type]
@init { String lit = $start.getText().toString(); }
@after { $start.symbolType = $type; }
    :   (HEX_LITERAL | OCTAL_LITERAL | DECIMAL_LITERAL) {
            $type = symtab.resolveBuiltinLitType("int", lit);
        }
    |   FLOATING_POINT_LITERAL {
            $type = symtab.resolveBuiltinLitType("double", lit);
        }
    |   CHARACTER_LITERAL {
            $type = symtab.resolveBuiltinLitType("char", lit);
        }
    |   STRING_LITERAL {
            $type = symtab.resolveBuiltinLitType("string", lit);
        }
    |   (TRUE | FALSE) {
            $type = symtab.resolveBuiltinLitType("boolean", lit);
        }
    |   NULL {
            $type = symtab.resolveBuiltinLitType("null", lit);
        }
    ;

type returns [Type sym]
@init {
    List<TypeName> typeText = new ArrayList<TypeName>();
    List<Type> tparams = new ArrayList<Type>();
    CharjAST head = null;
    Scope scope = null;
    boolean proxy = false;
    boolean pointer = false;
}
@after {
    //System.out.println("\ntype string: " + typeText);
    //System.out.println("direct scope: " + scope);
    $start.symbolType = scope.resolveType(typeText);
    //System.out.println("symbolType: " + $start.symbolType);
    if (proxy && $start.symbolType != null) $start.symbolType = new ProxyType(symtab, $start.symbolType);
    if (pointer && $start.symbolType != null) $start.symbolType = new PointerType(symtab, $start.symbolType);

    // TODO: Special case for Arrays, change this?
    if (typeText != null && typeText.size() > 0 &&
        typeText.get(0).name.equals("Array") && $start.symbolType == null) {

        int numDims = 1;

        ClassSymbol cs = new ClassSymbol(symtab, "Array");
        cs.templateArgs = typeText.get(0).parameters;

        if (cs.templateArgs != null &&
            cs.templateArgs.size() > 1) {
            if (cs.templateArgs.get(1) instanceof LiteralType) {
                numDims = Integer.valueOf(((LiteralType)cs.templateArgs.get(1)).literal);
            }
        }
        $start.symbolType = new PointerType(symtab, cs);
    }

    $sym = $start.symbolType;
    if ($sym == null) System.out.println("Couldn't resolve type: " + typeText);
}
    :   VOID {
            scope = $VOID.scope;
            typeText.add(new TypeName("void"));
        }
    |   ^(SIMPLE_TYPE t=. {
            scope = $SIMPLE_TYPE.scope;
            typeText.add(new TypeName($t.getText()));
        } .*)
    |   ^(OBJECT_TYPE { scope = $OBJECT_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT (^(TEMPLATE_INST
                (t1=type {tparams.add($t1.sym);} | lit1=literalVal {tparams.add($lit1.type);} )*))?
                {typeText.add(new TypeName($IDENT.text, tparams));}))+) .*)
    |   ^(REFERENCE_TYPE { scope = $REFERENCE_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT  {typeText.add(new TypeName($IDENT.text));} .*))+) .*)
    |   ^(PROXY_TYPE { scope = $PROXY_TYPE.scope; proxy = true; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeText.add(new TypeName($IDENT.text));} .*))+) .*)
    |   ^(POINTER_TYPE { scope = $POINTER_TYPE.scope; pointer = true; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT (^(TEMPLATE_INST
            (t1=type {tparams.add($t1.sym);} | lit1=literalVal {tparams.add($lit1.type);} )*))?
            {typeText.add(new TypeName($IDENT.text, tparams));}))+) .*)
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;
