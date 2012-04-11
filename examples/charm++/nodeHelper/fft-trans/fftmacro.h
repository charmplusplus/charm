//todo fix this is a more clever macro
#ifdef SINGLE_PRECISION
#define fft_complex fftwf_complex
#define fft_malloc fftwf_malloc
#define fft_free fftwf_free
#define fft_plan_many_dft fftwf_plan_many_dft
#define fft_destroy_plan fftwf_destroy_plan
#define fft_execute fftwf_execute
#define fft_plan fftwf_plan
#define realType float
#else
#define fft_complex fftw_complex
#define fft_malloc fftw_malloc
#define fft_free fftw_free
#define fft_plan_many_dft fftw_plan_many_dft
#define fft_destroy_plan fftw_destroy_plan
#define fft_execute fftw_execute
#define fft_plan fftw_plan
#define realType double
#endif
