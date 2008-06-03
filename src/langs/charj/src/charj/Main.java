package charj;

import charj.translator.Translator;
import java.io.FileInputStream;

public class Main 
{
    public static void main(String[] args) throws Exception
    {
        for (String filename : args) { 
            System.out.println(Translator.translate(filename));
        }
    }
}
