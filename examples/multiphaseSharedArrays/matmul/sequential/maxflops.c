
main()
{
    register double result = 1.0;
    // 10 times
    for(register unsigned long int i=0; i<10; i++)
        // 2 GF
        for(register unsigned long int j=0; j<1000000000; j++)
            result += result * 1.01;
}
