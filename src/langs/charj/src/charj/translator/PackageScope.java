
package charj.translator;

import java.util.HashMap;
import java.util.Map;

public class PackageScope extends SymbolWithScope {

    /** List of packages and classes in this package */
    Map<String, Symbol> members = new HashMap();
    Scope enclosingScope;

    public PackageScope(
            SymbolTable symtab, 
            String name, 
            Scope enclosingScope) {
        super(symtab, name);
        this.enclosingScope = enclosingScope;
    }

    public Scope getEnclosingScope() 
    {
        return enclosingScope;
    }

    /** See if type is already defined in this package.  If not, look
     *  for type on the disk in same package.  For example, first time
     *  charj.lang.Chare fails to resolve.  Load from disk and put File
     *  in package io which is in package charj.  Next time, File will
     *  be found.
     */
    public ClassSymbol resolveType(String type) {
        if (debug()) System.out.println(
                " PackageScope.resolveType(" + type + 
                "): examine " + toString());

        // look for type in this package's members (other packages, classes)
        if ( getMembers()!=null ) {
            Symbol s = getMembers().get(type);
            if ( s!=null && s instanceof ClassSymbol ) {
                if (debug()) System.out.println(
                        " PackageScope.resolveType(" + type + "): found in " + 
                        toString());
                return (ClassSymbol)s;
            }
        }
        return null;
    }

    public Map<String, Symbol> getMembers() {
        return members;
    }

    public String getFullyQualifiedName() {
        if ( name.equals(SymbolTable.DEFAULT_PACKAGE_NAME) ) {
            return null;
        }
        return super.getFullyQualifiedName();
    }

    public String toString() {
        return "PackageScope[" + name + "]: " + members.keySet();
    }
}
