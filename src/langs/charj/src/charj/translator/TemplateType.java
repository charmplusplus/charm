
package charj.translator;

import java.util.List;

public class TemplateType extends Symbol implements Type {
    public Type baseType;
    public List<Type> parameters;

    public TemplateType(SymbolTable symtab, Type _baseType, 
                        List<Type> _parameters) {
        super(symtab, _baseType.getTypeName(), null);
        baseType = _baseType;
        parameters = _parameters;
    }

    public String getTypeName() {
        StringBuilder sb = new StringBuilder();

        for (Type t : parameters) {
            sb.append(t.getTypeName());
        }

        return baseType.getTypeName() + "<" + sb.toString() + ">";
    }

	// TODO dummy implementation so as to compile
	public String getTranslatedTypeName()
	{
		return getTypeName();
	}
}
