package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;
import java.util.*;

class AstModifier
{
    private CharjAST pupNode;
    private CharjAST initNode;
    private CharjAST migrationCtor;
    private CharjAST constructorHelper;

    AstModifier()
    {
        createPupNode();
        createInitNode();
        createCtorHelperNode();
    }

    protected CharjAST getPupRoutineNode()
    {
        return pupNode;
    }

    protected CharjAST getInitRoutineNode()
    {
        return initNode;
    }

    protected CharjAST getCtorHelperNode()
    {
        return constructorHelper;
    }

    private CharjAST createNode(int type, String text)
    {
        return new CharjAST(new CommonToken(type, text));
    }   

    private void createCtorHelperNode()
    {
        constructorHelper = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");
        constructorHelper.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        constructorHelper.getChild(0).addChild(createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST"));
        constructorHelper.getChild(0).getChild(0).addChild(createNode(CharjParser.PROTECTED, "protected"));
        constructorHelper.addChild(createNode(CharjParser.VOID, "void"));
        constructorHelper.addChild(createNode(CharjParser.IDENT, "constructorHelper"));
        constructorHelper.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
        constructorHelper.addChild(createNode(CharjParser.BLOCK, "BLOCK"));
    }
    
    private void createInitNode()
    {
        initNode = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");

    	initNode.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        initNode.addChild(createNode(CharjParser.VOID, "void"));
        initNode.addChild(createNode(CharjParser.IDENT, "initMethod"));
        initNode.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
        initNode.addChild(createNode(CharjParser.BLOCK, "BLOCK"));

        initNode.getChild(0).addChild(createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST"));
        initNode.getChild(0).getChild(0).addChild(createNode(CharjParser.PRIVATE, "private"));
    }

    private void createPupNode()
    {
        pupNode = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");

    	pupNode.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        pupNode.addChild(createNode(CharjParser.VOID, "void"));
        pupNode.addChild(createNode(CharjParser.IDENT, "pup"));
        pupNode.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
        pupNode.addChild(createNode(CharjParser.BLOCK, "BLOCK"));

        pupNode.getChild(0).addChild(createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST"));
        pupNode.getChild(0).getChild(0).addChild(createNode(CharjParser.PUBLIC, "public"));

        pupNode.getChild(3).addChild(createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"));
        pupNode.getChild(3).getChild(0).addChild(createNode(CharjParser.REFERENCE_TYPE, "REFERENCE_TYPE"));
        pupNode.getChild(3).getChild(0).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
        pupNode.getChild(3).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "PUP::er"));

        pupNode.getChild(3).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
    }

    protected CharjAST getEnclosingType(CharjAST varDeclNode)
    {
        for(CharjAST p = varDeclNode.getParent(); p != null; p = p.getParent())
            if(p.getType() == CharjParser.TYPE)
                return p;
        return null;
    }         

    protected void varPup(CharjAST idNode)
    {
        int type = -1;

        for(CharjAST p = idNode.getParent(); p != null; p = p.getParent())
        {
            switch(p.getType())
            {
                case CharjParser.PRIMITIVE_VAR_DECLARATION:
                    //System.out.println("found primitive var: " + idNode.getText());
                    type = p.getType();
                    break;
                case CharjParser.OBJECT_VAR_DECLARATION:
                    //System.out.println("found object var: " + idNode.getText());
                    type = p.getChildAfterType(CharjParser.MODIFIER_LIST).getType();
                    break;
                case CharjParser.FUNCTION_METHOD_DECL:
                case CharjParser.BLOCK:
                case CharjParser.FORMAL_PARAM_LIST:
                    return;
                case CharjParser.TYPE:
                    switch(type)
                    {
                        case CharjParser.REFERENCE_TYPE:
                            //System.out.println("puping a reference type");
                            break;
                        case CharjParser.PRIMITIVE_VAR_DECLARATION:
                            //System.out.println("puping a primitive type");
                            primitiveVarPup(idNode);
                            break;
                        case CharjParser.POINTER_TYPE:
                            //System.out.println("puping a pointer type");
                            pointerVarPup(idNode);
                            break;
                        case CharjParser.PROXY_TYPE:
                            //System.out.println("puping a proxy type");
                            proxyVarPup(idNode);
                            break;
                        default:
                            System.out.println("AstModifier.varPup: unknown type " + idNode);
                            break;
                    }
                    return;
            }
        }
        System.out.println("AstModifier.varPup: could not pup variable " + idNode);
    }

    protected void primitiveVarPup(CharjAST idNode)
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));
        
        int index = pupNode.getChild(4).getChildren().size() - 1;

        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.BITWISE_OR, "|"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(idNode.dupNode());
    }

    protected void proxyVarPup(CharjAST idNode)
    {
        // For now, just do a basic PUP. More complex handling may be needed later.
        primitiveVarPup(idNode);
    }
    
    private boolean generatedIf = false;

    protected void pointerVarPup(CharjAST idNode)
    {
        if(!generatedIf)
        {
            generateIf();
            generatedIf = true;
        }

        // add stuff to the initMethod routine
        initNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));

        int index = initNode.getChild(4).getChildren().size() - 1;

        initNode.getChild(4).getChild(index).addChild(createNode(CharjParser.ASSIGNMENT, "="));
        initNode.getChild(4).getChild(index).getChild(0).addChild(idNode.dupNode());
        initNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.NEW, "new"));
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.OBJECT_TYPE, "OBJECT_TYPE"));
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).getChild(0).getChild(0).addChild(idNode.getParent().getParent().getParent().getChildOfType(CharjParser.POINTER_TYPE).getChild(0).getChild(0).dupTree());
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));

        // add stuff to the pup routine
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));

        index = pupNode.getChild(4).getChildren().size() - 1;

        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.ARROW, "ARROW"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(idNode.dupNode());
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "pup"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(1).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
    }

    protected void generateIf()
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.IF, "if"));
        
        int index = pupNode.getChild(4).getChildren().size() - 1;
       
        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.PAREN_EXPR, "PAREN_EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.DOT, "."));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "isUnpacking"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.BLOCK, "BLOCK"));
        pupNode.getChild(4).getChild(index).getChild(1).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(1).getChild(0).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(1).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "initMethod"));
        pupNode.getChild(4).getChild(index).getChild(1).getChild(0).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
    }

    protected void arrangeModifiers(CharjAST modlistNode)
    {
        CharjAST accessList = createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST");
        CharjAST localList = createNode(CharjParser.LOCAL_MODIFIER_LIST, "LOCAL_MODIFIER_LIST");
        CharjAST charjList = createNode(CharjParser.CHARJ_MODIFIER_LIST, "CHARJ_MODIFIER_LIST");
        CharjAST otherList = createNode(CharjParser.CHARJ_MODIFIER_LIST, "OTHER_MODIFIER_LIST");


        Iterator<CharjAST> iter = modlistNode.getChildren().iterator();
        
        while(iter.hasNext())
        {
            CharjAST mod = iter.next();
            iter.remove();

            switch(mod.getType())
            {
                case CharjParser.PUBLIC:
                case CharjParser.PRIVATE:
                case CharjParser.PROTECTED:
                    accessList.addChild(mod.dupNode());
                    break;
                case CharjParser.ENTRY:
                    charjList.addChild(mod.dupNode());
                    break;
                case CharjParser.FINAL:
                case CharjParser.STATIC:
                case CharjParser.VOLATILE:
                    localList.addChild(mod.dupNode());
                    break;
                case CharjParser.ABSTRACT:
                case CharjParser.NATIVE:
                    otherList.addChild(mod.dupNode());
                    break;
            }
        }

       if(accessList.getChildren() == null)
           try
           {
               if(charjList.getChildren().contains(createNode(CharjParser.ENTRY, "entry")))
                   accessList.addChild(createNode(CharjParser.PUBLIC, "public"));
               else
                   accessList.addChild(createNode(CharjParser.PRIVATE, "private"));
           }
           catch(NullPointerException npe)
           {
               // charjList == null && accessList is empty
               accessList.addChild(createNode(CharjParser.PRIVATE, "private"));
           }

       modlistNode.addChild(accessList);
       if(localList.getChildren() != null) modlistNode.addChild(localList);
       if(charjList.getChildren() != null) modlistNode.addChild(charjList);
       if(otherList.getChildren() != null) modlistNode.addChild(otherList);
    }

    protected void fillPrivateModifier(CharjAST declNode)
    {
        CharjAST modlist = createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST");
        modlist.addChild(createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST"));
        modlist.getChild(0).addChild(createNode(CharjParser.PRIVATE, "private"));

        declNode.insertChild(0, modlist);
    }

    protected void insertHelperRoutineCall(CharjAST ctordecl)
    {
        CharjAST expr = createNode(CharjParser.EXPR, "EXPR");
        expr.addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        expr.getChild(0).addChild(createNode(CharjParser.IDENT, "constructorHelper"));
        expr.getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));

        ctordecl.getChild(3).insertChild(0, expr);   
    }

    protected void dealWithInit(CharjAST vardecl)
    {
        if(vardecl.getChildCount() > 1) // has an initialization expression
        {
            boolean localScope = false;
            for(CharjAST temp = vardecl; temp != null; temp = temp.getParent())
                if(temp.getType() == CharjParser.FUNCTION_METHOD_DECL || temp.getType() == CharjParser.CONSTRUCTOR_DECL)
                {
                    localScope = true;
                    break;
                }
            if(!localScope)
            {
                // add current variable initialization to constructorHelper
                CharjAST expr = createNode(CharjParser.EXPR, "EXPR");
                expr.addChild(createNode(CharjParser.ASSIGNMENT, "="));
                expr.getChild(0).addChild(vardecl.getChild(0).dupNode());
                expr.getChild(0).addChild(vardecl.getChild(1).getChild(0).dupTree());
                constructorHelper.getChild(4).addChild(expr);
            }
        }
    }

    private boolean hasMigrationCtor = false;
    private CharjAST defaultCtor;

    protected void checkForDefaultCtor(CharjAST ctordecl, CharjAST ctordecltree)
    {
        if(defaultCtor != null)
            return;

        CharjAST params = null;
        for(CharjAST node : ctordecl.getChildren()) {
            if(node.getType() == CharjParser.FORMAL_PARAM_LIST)
            {
                params = node;
                break;
            }
        }
        if(params.getChildren() == null)
            defaultCtor = ctordecltree;
        else 
            try
            {
                if(params.getChildren().size() == 1 && params.getChild(0).getChild(0).getChild(0).getChild(0).getText().equals("CkArgMsg"))
                    for(CharjAST temp = ctordecl; temp != null; temp = temp.getParent())
                        if(temp.getType() == CharjParser.TYPE && temp.getChild(0).getType() == CharjParser.MAINCHARE)
                        {
                            defaultCtor = ctordecltree;
                            return;
                        }
            }
            catch(NullPointerException npe)
            {
                // argument is a simple type, so one of the getChild(0) calls above returns null
            }
    }

    protected boolean isMigrationCtor(CharjAST ctordecl)
    {
        CharjAST params = null;
        for(CharjAST node : ctordecl.getChildren()) {
            if(node.getType() == CharjParser.FORMAL_PARAM_LIST)
            {
                params = node;
                break;
            }
        }
        
        if (params == null || params.getChildren() == null) return false;
        if (params.getChildren().size() != 1) return false;
        params = params.getChild(0);
        if (params == null || params.getType() != CharjParser.FORMAL_PARAM_STD_DECL) return false;
        params = params.getChild(0);
        if (params == null || params.getType() != CharjParser.POINTER_TYPE) return false ;
        params = params.getChild(0);
        if (params == null || params.getType() != CharjParser.QUALIFIED_TYPE_IDENT) return false;
        params = params.getChild(0);
        if (params.toString().equals("CkMigrateMessage")) return true;
        return false;

    }

    protected void checkForMigrationCtor(CharjAST ctordecl)
    {
        if(hasMigrationCtor) return;
        if (isMigrationCtor(ctordecl)) {
            hasMigrationCtor = true;
            migrationCtor = ctordecl;
        }
    }

    protected void ensureDefaultCtor(CharjAST typenode)
    {
        if(defaultCtor != null && typenode.getChild(0).getType() == CharjParser.MAINCHARE && defaultCtor.getChild(2).getChildren() == null)
        {
            // fill CkMsgArg* argument
            defaultCtor.getChild(2).addChild(createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"));
            defaultCtor.getChild(2).getChild(0).addChild(createNode(CharjParser.POINTER_TYPE, "POINTER_TYPE"));
            defaultCtor.getChild(2).getChild(0).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
            defaultCtor.getChild(2).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "CkArgMsg"));
            defaultCtor.getChild(2).getChild(0).addChild(createNode(CharjParser.IDENT, "m"));
        }
        else if(defaultCtor == null)
        {
            defaultCtor = createNode(CharjParser.CONSTRUCTOR_DECL, "CONSTRUCTOR_DECL");
            defaultCtor.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
            defaultCtor.getChild(0).addChild(createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST"));
            defaultCtor.getChild(0).getChild(0).addChild(createNode(CharjParser.PUBLIC, "public"));
            defaultCtor.addChild(typenode.getChild(1).dupNode());
            defaultCtor.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
            defaultCtor.addChild(createNode(CharjParser.BLOCK, "BLOCK"));

            insertHelperRoutineCall(defaultCtor);

            if(typenode.getChild(0).getType() == CharjParser.MAINCHARE)
            {
                // fill CkMsgArg* argument
                defaultCtor.getChild(0).addChild(createNode(CharjParser.CHARJ_MODIFIER_LIST, "CHARJ_MODIFIER_LIST"));
                defaultCtor.getChild(0).getChild(1).addChild(createNode(CharjParser.ENTRY, "entry"));
                defaultCtor.getChild(2).addChild(createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"));
                defaultCtor.getChild(2).getChild(0).addChild(createNode(CharjParser.POINTER_TYPE, "POINTER_TYPE"));
                defaultCtor.getChild(2).getChild(0).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
                defaultCtor.getChild(2).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "CkArgMsg"));
                defaultCtor.getChild(2).getChild(0).addChild(createNode(CharjParser.IDENT, "m"));
            }

            typenode.addChild(defaultCtor);
        }
    }

    protected CharjAST ensureMigrationCtor(CharjAST typenode)
    {
        if(hasMigrationCtor)
            return migrationCtor;
        
        CharjAST ctor = createNode(CharjParser.CONSTRUCTOR_DECL, "CONSTRUCTOR_DECL");
        ctor.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        ctor.getChild(0).addChild(createNode(CharjParser.ACCESS_MODIFIER_LIST, "ACCESS_MODIFIER_LIST"));
        ctor.getChild(0).getChild(0).addChild(createNode(CharjParser.PUBLIC, "public"));
        ctor.getChild(0).addChild(createNode(CharjParser.CHARJ_MODIFIER_LIST, "CHARJ_MODIFIER_LIST"));
        ctor.getChild(0).getChild(1).addChild(createNode(CharjParser.ENTRY, "entry"));
        ctor.addChild(typenode.getChild(1).dupNode());
        CharjAST args = createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST");
        args.addChild(createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"));
        args.getChild(0).addChild(createNode(CharjParser.POINTER_TYPE, "POINTER_TYPE"));
        args.getChild(0).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
        args.getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "CkMigrateMessage"));
        args.getChild(0).addChild(createNode(CharjParser.IDENT, "m"));
        ctor.addChild(args);
        ctor.addChild(createNode(CharjParser.BLOCK, "BLOCK"));
        typenode.addChild(ctor);
        migrationCtor = ctor;
        return migrationCtor;
    }

    protected void makePointerDereference(CharjAST node)
    {
        CharjAST deref = createNode(CharjParser.POINTER_DEREFERENCE, "POINTER_DEREFERENCE");
        deref.addChild(node.dupNode());

        CharjAST parent = node.getParent();
        int index = node.getChildIndex();
        parent.deleteChild(index);
        parent.insertChild(index, deref);
    }

    protected boolean isEntry(CharjAST funcdecl)
    {
        CharjAST mods = funcdecl.getChildOfType(CharjParser.MODIFIER_LIST);
        if(mods.getChildOfType(CharjParser.ENTRY) != null)
            return true;
        CharjAST charjmods = mods.getChildOfType(CharjParser.CHARJ_MODIFIER_LIST);
        if(charjmods == null)
            return false;
        return charjmods.getChildOfType(CharjParser.ENTRY) != null;
    }

    protected void dealWithEntryMethodParam(CharjAST pointertype, CharjAST pointertypetree)
    {
        try
        {
            CharjAST funcdecl = pointertype.getParent().getParent().getParent();
            if(funcdecl.getType() == CharjParser.FUNCTION_METHOD_DECL && isEntry(funcdecl))
                pointertypetree.setType(CharjParser.OBJECT_TYPE, "OBJECT_TYPE");
        }
        catch(NullPointerException npe)
        {
            // do nothing, it's just not a method parameter
        }
    }

}
