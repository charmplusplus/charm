package charj.translator;

import java.util.LinkedHashMap;
import java.util.Map;

public class LocalScope extends SymbolWithScope implements Scope {
    Scope parent;
    Map members = new LinkedHashMap();

    public LocalScope(
            SymbolTable symtab, 
            Scope parent) 
    {
        super(symtab);
        this.parent = parent;
    }

    public Scope getEnclosingScope() 
    {
        return parent;
    }

    public Map getMembers() 
    {
        return members;
    }

    /** A local scope's name is the parent method's scope */
    public String getScopeName() 
    {
        return parent.getScopeName();
    }

    public String toString() 
    {
        return members.toString();
    }
}
