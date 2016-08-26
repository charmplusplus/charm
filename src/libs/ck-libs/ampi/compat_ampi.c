/*
 * C wrapper for calling AMPI_Main(), which is
 * declared without a strict prototype, from C++.
 */
void AMPI_Main_c(int argc,char **argv)
{
	AMPI_Main(argc,argv); /* call C main routine with implicit declaration */
}
