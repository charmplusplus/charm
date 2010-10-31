
package charj.translator;

public class PointerType extends Symbol implements Type {
    public Type baseType;

    public PointerType(SymbolTable symtab, Type _baseType) {
        super(symtab, _baseType.getTypeName(), null);
        baseType = _baseType;
    }

    public String getTypeName() {
        return baseType.getTypeName();
    }

    public String getTranslatedTypeName() {
        return baseType.getTypeName() + "*";
    }
}
