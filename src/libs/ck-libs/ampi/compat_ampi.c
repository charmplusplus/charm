/*
 * C wrapper for calling AMPI_Main().
 */

extern void AMPI_Main(int argc, char **argv);

void AMPI_Main_c(int argc, char **argv)
{
	AMPI_Main(argc, argv); /* call C main routine */
}
