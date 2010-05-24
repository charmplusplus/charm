package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;

class PupRoutineCreator
{
    private CharjAST pupNode;
    private CharjAST block;

    PupRoutineCreator()
    {
        createPupNode();
        block = pupNode.getChild(4);
    }

    private CharjAST createNode(int type, String text)
    {
        return new CharjAST(new CommonToken(type, text));
    }    

    private void createPupNode()
    {
        pupNode = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");
        
        CharjAST modlist = createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST");
        modlist.addChild(createNode(CharjParser.PUBLIC, "public"));
        pupNode.addChild(modlist);

        pupNode.addChild(createNode(CharjParser.VOID, "void"));

        pupNode.addChild(createNode(CharjParser.IDENT, "pup"));

        CharjAST paramlist = createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST");

        CharjAST param = createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL");

        CharjAST type = createNode(CharjParser.TYPE, "TYPE");
        
        CharjAST qualifiedtypeid = createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT");
        qualifiedtypeid.addChild(createNode(CharjParser.IDENT, "PUP::er&"));

        type.addChild(qualifiedtypeid);
        param.addChild(type);
        paramlist.addChild(param);
        
        pupNode.addChild(createNode(CharjParser.BLOCK, "BLOCK"));
    }

    protected CharjAST getEnclosingType(CharjAST varDeclNode)
    {
        for(CharjAST p = varDeclNode.getParent(); p != null; p = p.getParent())
            if(p.getType() == CharjParser.TYPE)
                return p;
        return null;
    }         

    protected void primitiveVarPup(CharjAST varDeclNode)
    {
        CharjAST enclTypeNode = getEnclosingType(varDeclNode);
    }
    
    protected void objectVarPup(CharjAST varDeclNode){}
}
