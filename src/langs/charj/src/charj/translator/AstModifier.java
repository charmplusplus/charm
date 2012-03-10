package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;
import java.util.*;

class AstModifier
{
    public AstModifier() { }

    private CharjAST createNode(int type, String text)
    {
        return new CharjAST(new CommonToken(type, text));
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
        if(mods.getChildOfType(CharjParser.ENTRY) != null) return true;
        CharjAST charjmods = mods.getChildOfType(CharjParser.CHARJ_MODIFIER_LIST);
        if(charjmods == null) return false;
        return charjmods.getChildOfType(CharjParser.ENTRY) != null;
    }

    protected void dealWithEntryMethodParam(CharjAST pointertype, CharjAST pointertypetree)
    {
        try {
            CharjAST funcdecl = pointertype.getParent().getParent().getParent();
            if(funcdecl.getType() == CharjParser.FUNCTION_METHOD_DECL && isEntry(funcdecl))
                pointertypetree.setType(CharjParser.OBJECT_TYPE, "OBJECT_TYPE");
        } catch(NullPointerException npe) {
            // do nothing, it's just not a method parameter
        }
    }

}

