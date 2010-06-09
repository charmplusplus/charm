/**
 * The semantic phase walks the tree and builds the symbol table, handles
 * all the imports, and does the semantic checks. The resulting tree and
 * symbol table are used by the emitter to generate the output. 
 */

tree grammar CharjSemantics;

options {
    backtrack = true; 
    memoize = true;
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
}

scope ScopeStack {
    Scope current;
}

@header {
package charj.translator;
}

@members {
    SymbolTable symtab = null;
    PackageScope currentPackage = null;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;
    LocalScope currentLocalScope = null;
    Translator translator;
    List<CharjAST> imports = new ArrayList<CharjAST>();
    AstModifier astmod = new AstModifier();

    /**
     *  Test a list of CharjAST nodes to see if any of them has the given token
     *  type.
     */
    public boolean listContainsToken(List<CharjAST> list, int tokenType) {
        if (list == null) return false;
        for (CharjAST node : list) {
            if (node.token.getType() == tokenType) {
                return true;
            }
        }
        return false;
    }

    public void importPackages(ClassSymbol cs, List<CharjAST> imports) {
        if (imports == null) {
            return;
        }

        for (CharjAST pkg : imports) {
            String pkgName = input.getTokenStream().toString(
                    pkg.getTokenStartIndex(),
                    pkg.getTokenStopIndex());
            // find imported class and add to cs.imports
            PackageScope p = cs.importPackage(pkgName);
            if (p == null) {
                translator.error(
                    this, 
                    "package " + pkgName + " not found.",
                    pkg);
            }
        }
    }

    public void addImport(CharjAST importNode) {
        imports.add(importNode);
    }
}


// Replace default ANTLR generated catch clauses with this action, allowing early failure.
@rulecatch {
    catch (RecognitionException re) {
        reportError(re);
        throw re;
    }
}


// Starting point for parsing a Charj file.
charjSource[SymbolTable _symtab] returns [ClassSymbol cs]
scope ScopeStack; // default scope
@init {
    symtab = _symtab;
    $ScopeStack::current = symtab.getDefaultPkg();
}
    :   ^(CHARJ_SOURCE 
        (packageDeclaration)? 
        (importDeclaration
        | typeDeclaration { $cs = $typeDeclaration.sym; }
        | readonlyDeclaration)*)
    ;

// note: no new scope here--this replaces the default scope
packageDeclaration
@init { 
    List<String> names = null; 
    String packageName = "";
}
    :   ^(PACKAGE ((ids+=IDENT) { packageName += "." + $IDENT.text; })+)
        {
            packageName = packageName.substring(1);
            PackageScope ps = symtab.resolvePackage(packageName);
            if (ps == null) {
                ps = symtab.definePackage(packageName);
                symtab.addScope(ps);
            }
            currentPackage = ps;
            $ScopeStack::current = ps;
        }
    ;
    
importDeclaration
    :   ^(IMPORT qualifiedIdentifier '.*'?)
        { addImport($qualifiedIdentifier.start); }
    ;

readonlyDeclaration
    :   ^(READONLY localVariableDeclaration)
    ;

typeDeclaration returns [ClassSymbol sym]
scope ScopeStack; // top-level type scope
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))? (^('implements' type+))?
            {
                Scope outerScope = $ScopeStack[-1]::current;
                $sym = new ClassSymbol(symtab, $IDENT.text, outerScope.resolveType($parent.text), outerScope);
                outerScope.define($sym.name, $sym);
                currentClass = $sym;
                $sym.definition = $typeDeclaration.start;
                $sym.definitionTokenStream = input.getTokenStream();
                $IDENT.symbol = $sym;
                $ScopeStack::current = $sym;
                String classTypeName = $classType.text;
                if (classTypeName.equals("class")) {
                } else if (classTypeName.equals("chare")) {
                    currentClass.isChare = true;
                } else if (classTypeName.equals("group")) {
                    currentClass.isChare = true;
                } else if (classTypeName.equals("nodegroup")) {
                    currentClass.isChare = true;
                } else if (classTypeName.equals("chare_array")) {
                    currentClass.isChare = true;
                } else if (classTypeName.equals("mainchare")) {
                    currentClass.isChare = true;
                    currentClass.isMainChare = true;
                } else System.out.println("Error: type " + classTypeName + " not recognized.");
                importPackages($sym, imports);
            }
            classScopeDeclaration*)
            {
                //System.out.println("Members for type " + $sym.name + ":");
                //for (Map.Entry<String, Symbol> entry : $sym.members.entrySet()) {
                //    System.out.println(entry.getKey());
                //}
            }
    |   ^('interface' IDENT (^('extends' type+))?  interfaceScopeDeclaration*)
    |   ^('enum' IDENT (^('implements' type+))? enumConstant+ classScopeDeclaration*)
    ;

classType
    :   CLASS
    |   chareType
    ;

chareType
    :   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

enumConstant
    :   ^(IDENT arguments?)
    ;
    
classScopeDeclaration
scope ScopeStack;
    :   ^(FUNCTION_METHOD_DECL m=modifierList? g=genericTypeParameterList? 
            ty=type IDENT f=formalParameterList a=arrayDeclaratorList? 
            b=block?)
        {
            ClassSymbol returnType = currentClass.resolveType($ty.text);
            MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, returnType);
            currentMethod = sym;
            sym.definition = $classScopeDeclaration.start;
            sym.definitionTokenStream = input.getTokenStream();
            currentClass.define($IDENT.text, sym);
            $FUNCTION_METHOD_DECL.symbol = sym;
        }
    |   ^(ENTRY_FUNCTION_DECL m=modifierList? g=genericTypeParameterList?
            ty=type IDENT formalParameterList a=arrayDeclaratorList? b=block)
        {
            ClassSymbol returnType = currentClass.resolveType($ty.text);
            MethodSymbol sym = new MethodSymbol(symtab, $IDENT.text, currentClass, returnType);
            currentMethod = sym;
            sym.definition = $classScopeDeclaration.start;
            sym.definitionTokenStream = input.getTokenStream();
            currentClass.define($IDENT.text, sym);
            $ENTRY_FUNCTION_DECL.symbol = sym;
        }
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType
            ^(VAR_DECLARATOR_LIST field[$simpleType.type]+))
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType
            ^(VAR_DECLARATOR_LIST field[$objectType.type]+))
        {
            ClassSymbol type = $objectType.type;
            if (type != null && type.isChare) currentClass.addExtern(type.getName());
        }
    |   ^(CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            b=block)
        {
            if (astmod.isMigrationCtor($CONSTRUCTOR_DECL)) currentClass.migrationCtor = $CONSTRUCTOR_DECL;
        }
    |   ^(ENTRY_CONSTRUCTOR_DECL m=modifierList? g=genericTypeParameterList? IDENT f=formalParameterList 
            b=block)
        {
            if (astmod.isMigrationCtor($ENTRY_CONSTRUCTOR_DECL)) currentClass.migrationCtor = $ENTRY_CONSTRUCTOR_DECL;
        }
    ;

field [ClassSymbol type]
    :   ^(VAR_DECLARATOR ^(IDENT arrayDeclaratorList?) variableInitializer?)
    {
            VariableSymbol sym = new VariableSymbol(symtab, $IDENT.text, $type);
            sym.definition = $field.start;
            sym.definitionTokenStream = input.getTokenStream();
            $VAR_DECLARATOR.symbol = sym;
            currentClass.define($IDENT.text, sym);
    }
    ;
    
interfaceScopeDeclaration
    :   ^(FUNCTION_METHOD_DECL modifierList? genericTypeParameterList? 
            type IDENT formalParameterList arrayDeclaratorList?)
        // Interface constant declarations have been switched to variable
        // declarations by Charj.g; the parser has already checked that
        // there's an obligatory initializer.
    |   ^(PRIMITIVE_VAR_DECLARATION modifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION modifierList? objectType variableDeclaratorList)
    ;

variableDeclaratorList
    :   ^(VAR_DECLARATOR_LIST variableDeclarator+)
    ;

variableDeclarator
    :   ^(VAR_DECLARATOR ^(IDENT arrayDeclaratorList?) variableInitializer?)

    ;
    
variableDeclaratorId
    :   ^(IDENT arrayDeclaratorList?)
    ;

variableInitializer
    :   arrayInitializer
    |   expression
    ;

arrayDeclaratorList
    :   ^(ARRAY_DECLARATOR_LIST ARRAY_DECLARATOR*)  
    ;
    
arrayInitializer
    :   ^(ARRAY_INITIALIZER variableInitializer*)
    ;

genericTypeParameterList
    :   ^(GENERIC_TYPE_PARAM_LIST genericTypeParameter+)
    ;

genericTypeParameter
    :   ^(IDENT bound?)
    ;
        
bound
    :   ^(EXTENDS_BOUND_LIST type+)
    ;

modifierList
    :   ^(MODIFIER_LIST accessModifierList? localModifierList? charjModifierList? otherModifierList?)
    ;

modifier
    :   accessModifier
    |   localModifier
    |   charjModifier
    |   otherModifier
    ;

localModifierList
    :   ^(LOCAL_MODIFIER_LIST localModifier+)
    ;

accessModifierList
    :   ^(ACCESS_MODIFIER_LIST accessModifier+)
    ;

charjModifierList
    :   ^(CHARJ_MODIFIER_LIST charjModifier+)
    ;

otherModifierList
    :   ^(OTHER_MODIFIER_LIST otherModifier+)
    ;
    
localModifier
    :   FINAL
    |   STATIC
    |   VOLATILE
    ;

accessModifier
    :   PUBLIC
    |   PROTECTED
    |   PRIVATE
    ;

charjModifier
    :   ENTRY
    ;

otherModifier
    :   ABSTRACT
    |   NATIVE
    ;

type
    :   simpleType
    |   objectType
    |   VOID
    ;

simpleType returns [ClassSymbol type]
    :   ^(SIMPLE_TYPE primitiveType arrayDeclaratorList?)
        {
            $type = symtab.resolveBuiltinType($primitiveType.text);
        }
    ;
    
objectType returns [ClassSymbol type]
    :   ^(OBJECT_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(REFERENCE_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(PROXY_TYPE qualifiedTypeIdent arrayDeclaratorList?)
    |   ^(POINTER_TYPE qualifiedTypeIdent arrayDeclaratorList?)
        {
            $type = $qualifiedTypeIdent.type;
        }
    ;

qualifiedTypeIdent returns [ClassSymbol type]
@init {
String name = "";
}
    :   ^(QUALIFIED_TYPE_IDENT (typeIdent {name += $typeIdent.name;})+) 
        {
            $type = null;
            /*System.out.println("trying to resolve type " + name + " in type " + currentClass);*/
            if (currentClass != null) $type = currentClass.resolveType(name);
            /*System.out.println("got " + $type);*/
            if ($type == null) $type = symtab.resolveBuiltinType(name);
            $QUALIFIED_TYPE_IDENT.symbol = $type;
        }
    ;

typeIdent returns [String name]
    :   ^(IDENT genericTypeArgumentList?)
        { $name = $IDENT.text; }
    ;

primitiveType
    :   BOOLEAN     { $start.symbol = new Symbol(symtab, "bool_primitive", symtab.resolveBuiltinType("bool")); }
    |   CHAR        { $start.symbol = new Symbol(symtab, "char_primitive", symtab.resolveBuiltinType("char")); }
    |   BYTE        { $start.symbol = new Symbol(symtab, "byte_primitive", symtab.resolveBuiltinType("char")); }
    |   SHORT       { $start.symbol = new Symbol(symtab, "short_primitive", symtab.resolveBuiltinType("short")); }
    |   INT         { $start.symbol = new Symbol(symtab, "int_primitive", symtab.resolveBuiltinType("int")); }
    |   LONG        { $start.symbol = new Symbol(symtab, "long_primitive", symtab.resolveBuiltinType("long")); }
    |   FLOAT       { $start.symbol = new Symbol(symtab, "float_primitive", symtab.resolveBuiltinType("float")); }
    |   DOUBLE      { $start.symbol = new Symbol(symtab, "double_primitive", symtab.resolveBuiltinType("double")); }
    ;

genericTypeArgumentList
    :   ^(GENERIC_TYPE_ARG_LIST genericTypeArgument+)
    ;
    
genericTypeArgument
    :   type
    |   '?'
    ;

formalParameterList
    :   ^(FORMAL_PARAM_LIST formalParameterStandardDecl* formalParameterVarargDecl?) 
    ;
    
formalParameterStandardDecl
    :   ^(FORMAL_PARAM_STD_DECL localModifierList? type variableDeclaratorId)
    ;
    
formalParameterVarargDecl
    :   ^(FORMAL_PARAM_VARARG_DECL localModifierList? type variableDeclaratorId)
    ;
    
// FIXME: is this rule right? Verify that this is ok, I expected something like:
// IDENT (^(DOT qualifiedIdentifier IDENT))*
qualifiedIdentifier
    :   IDENT
    |   ^(DOT qualifiedIdentifier IDENT)
    ;
    
block
    :   ^(BLOCK (blockStatement)*)
    ;
    
blockStatement
    :   localVariableDeclaration
    |   statement
    ;
    
localVariableDeclaration
    :   ^(PRIMITIVE_VAR_DECLARATION localModifierList? simpleType variableDeclaratorList)
    |   ^(OBJECT_VAR_DECLARATION localModifierList? objectType variableDeclaratorList)
        {
            ClassSymbol type = $objectType.type;
            /*System.out.println("looked up type " + type + " for declaration " + $objectType.text);*/
            if (type != null && type.isChare && currentClass != null) currentClass.addExtern(type.getName());
        }
    ;

statement
    : nonBlockStatement
    | block
    ;

nonBlockStatement
    :   ^(ASSERT expression expression?)
    |   ^(IF parenthesizedExpression block block?)
    |   ^(FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block)
    |   ^(FOR_EACH localModifierList? type IDENT expression block) 
    |   ^(WHILE parenthesizedExpression block)
    |   ^(DO block parenthesizedExpression)
    |   ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled break not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(CONTINUE IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled continue not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(LABELED_STATEMENT IDENT statement)
    |   expression
    |   ^('delete' qualifiedIdentifier)
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
    |   ';' // Empty statement.
    |   ^(PRINT expression*)
    |   ^(PRINTLN expression*)
    |   ^(EXIT expression?)
    |   EXITALL
    ;
        
switchCaseLabel
    :   ^(CASE expression blockStatement*)
    |   ^(DEFAULT blockStatement*)
    ;
    
forInit
    :   localVariableDeclaration 
    |   expression+
    ;
    
// EXPRESSIONS

parenthesizedExpression
    :   ^(PAREN_EXPR expression)
    ;
    
expression
    :   ^(EXPR expr)
    ;

expr
    :   ^(ASSIGNMENT expr expr)
    |   ^(PLUS_EQUALS expr expr)
    |   ^(MINUS_EQUALS expr expr)
    |   ^(TIMES_EQUALS expr expr)
    |   ^(DIVIDE_EQUALS expr expr)
    |   ^(AND_EQUALS expr expr)
    |   ^(OR_EQUALS expr expr)
    |   ^(POWER_EQUALS expr expr)
    |   ^(MOD_EQUALS expr expr)
    |   ^('>>>=' expr expr)
    |   ^('>>=' expr expr)
    |   ^('<<=' expr expr)
    |   ^('?' expr expr expr)
    |   ^(OR expr expr)
    |   ^(AND expr expr)
    |   ^(BITWISE_OR expr expr)
    |   ^(POWER expr expr)
    |   ^(BITWISE_AND expr expr)
    |   ^(EQUALS expr expr)
    |   ^(NOT_EQUALS expr expr)
    |   ^(INSTANCEOF expr type)
    |   ^(LTE expr expr)
    |   ^(GTE expr expr)
    |   ^('>>>' expr expr)
    |   ^('>>' expr expr)
    |   ^(GT expr expr)
    |   ^('<<' expr expr)
    |   ^(LT expr expr)
    |   ^(PLUS expr expr)
    |   ^(MINUS expr expr)
    |   ^(TIMES expr expr)
    |   ^(DIVIDE expr expr)
    |   ^(MOD expr expr)
    |   ^(UNARY_PLUS expr)
    |   ^(UNARY_MINUS expr)
    |   ^(PRE_INC expr)
    |   ^(PRE_DEC expr)
    |   ^(POST_INC expr)
    |   ^(POST_DEC expr)
    |   ^(TILDE expr)
    |   ^(NOT expr)
    |   ^(CAST_EXPR type expr)
    |   primaryExpression
    ;
    
primaryExpression
    :   ^(DOT primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
    |   ^(ARROW primaryExpression
                (   IDENT
                |   THIS
                |   SUPER
                )
        )
    |   parenthesizedExpression
    |   IDENT
    |   ^(METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   ^(ENTRY_METHOD_CALL primaryExpression genericTypeArgumentList? arguments)
    |   explicitConstructorCall
    |   ^(ARRAY_ELEMENT_ACCESS primaryExpression expression)
    |   literal
    |   newExpression
    |   THIS
    |   arrayTypeDeclarator
    |   SUPER
    |   GETNUMPES
    |   GETNUMNODES
    |   GETMYPE
    |   GETMYNODE
    |   GETMYRANK
    ;
    
explicitConstructorCall
    :   ^(THIS_CONSTRUCTOR_CALL genericTypeArgumentList? arguments)
    |   ^(SUPER_CONSTRUCTOR_CALL primaryExpression? genericTypeArgumentList? arguments)
    ;

arrayTypeDeclarator
    :   ^(ARRAY_DECLARATOR (arrayTypeDeclarator | qualifiedIdentifier | primitiveType))
    ;

newExpression
    :   ^(  STATIC_ARRAY_CREATOR
            (   primitiveType newArrayConstruction
            |   genericTypeArgumentList? qualifiedTypeIdent newArrayConstruction
            )
        )
    |   ^(NEW type arguments)
    ;

newArrayConstruction
    :   arrayDeclaratorList arrayInitializer
    |   expression+ arrayDeclaratorList?
    ;

arguments
    :   ^(ARGUMENT_LIST expression*)
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

