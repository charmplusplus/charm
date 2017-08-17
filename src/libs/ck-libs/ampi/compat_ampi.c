/*
 * C wrapper for calling AMPI_Main().
 */

#ifdef __cplusplus
extern "C"
#endif
void AMPI_Main(int argc, char **argv);

#ifdef __cplusplus
extern "C"
#endif
void AMPI_Main_c(int argc, char **argv)
{
	AMPI_Main(argc, argv); /* call C main routine */
}
