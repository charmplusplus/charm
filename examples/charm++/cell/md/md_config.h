#ifndef __MD_CONFIG_H__
#define __MD_CONFIG_H__


////////////////////////////////////////////////////////////////////////////////
// Default Simulation Parameters


// convenience typedefs and consts to facilitate float vs double usage
#define USE_DOUBLE 0

#if USE_DOUBLE
  typedef double  MD_FLOAT;
  #define  MD_VEC simdia_veclf
  const double zero=0.0;
  const double one=1.0;
  const double two=2.0;
  #define myvec_numElems simdia_veclf_numElems;
  #define vextract_MDF simdia_vextractlf
  #define vsub_MDF simdia_vsublf
  #define vadd_MDF simdia_vaddlf
  #define vmadd_MDF simdia_vmaddlf
  #define vmul_MDF simdia_vmullf
  #define vspread_MDF simdia_vspreadlf
  #define vrecip_MDF simdia_vreciplf
  #define vsqrt_MDF simdia_vsqrtlf
  #define visfinite_MDF simdia_visfinitelf
#else
  typedef float  MD_FLOAT;
  #define  MD_VEC simdia_vecf
  #define myvec_numElems simdia_vecf_numElems;
  const float zero=0.0f;
  const float one=1.0f;
  const float two=2.0f;
  #define vextract_MDF simdia_vextractf
  #define vsub_MDF simdia_vsubf
  #define vadd_MDF simdia_vaddf
  #define vmadd_MDF simdia_vmaddf
  #define vmul_MDF simdia_vmulf
  #define vspread_MDF simdia_vspreadf
  #define vrecip_MDF simdia_vrecipf
  #define vsqrt_MDF simdia_vsqrtf
  #define visfinite_MDF simdia_visfinitef
#endif

#define DEFAULT_NUM_PARTICLES_PER_PATCH  (128)

#define DEFAULT_NUM_PATCHES_X              (2)
#define DEFAULT_NUM_PATCHES_Y              (2)
#define DEFAULT_NUM_PATCHES_Z              (2)

#define DEFAULT_NUM_STEPS                 (16)
#define STEPS_PER_PRINT                   (16)

#define USE_ARRAY_SECTIONS                 (1)

#define USE_PROXY_PATCHES                  (1)


////////////////////////////////////////////////////////////////////////////////
// Physics Constants

#if USE_DOUBLE
#define TIME_PER_STEP       (1.0e-15)           // Unit: s
#define SIM_BOX_SIDE_LEN    (1.0e-7)            // Unit: m (NOTE: 1 nm = 10A)
#define COULOMBS_CONSTANT   (8.987551787e-9)    // Unit: N*(m^2)*(C^-2)
#define ELECTRON_CHARGE     (-1.602176487e-19)  // Unit: C
#define ELECTRON_MASS       (9.109382154e-31)   // Unit: kg
#else
#define TIME_PER_STEP       (1.0e-15f)           // Unit: s
#define SIM_BOX_SIDE_LEN    (1.0e-7f)            // Unit: m (NOTE: 1 nm = 10A)
#define COULOMBS_CONSTANT   (8.987551787e-9f)    // Unit: N*(m^2)*(C^-2)
#define ELECTRON_CHARGE     (-1.602176487e-19f)  // Unit: C
#define ELECTRON_MASS       (9.109382154e-31f)   // Unit: kg
#endif

////////////////////////////////////////////////////////////////////////////////
// Misc. Helper Macros

#define PATCH_XYZ_TO_I(x,y,z)  (((z)*numPatchesX*numPatchesY)+((y)*numPatchesX)+(x))
#define PATCH_I_TO_X(i)        ((i)%numPatchesX)
#define PATCH_I_TO_Y(i)        (((i)/numPatchesX)%numPatchesY)
#define PATCH_I_TO_Z(i)        ((i)/(numPatchesX*numPatchesY))


////////////////////////////////////////////////////////////////////////////////
// Misc. Macros for Performance Testing

// DMK - DEBUG
#define ENABLE_STATIC_LOAD_BALANCING     (0)

// DMK - DEBUG
#define DUMP_INITIAL_PARTICLE_DATA       (0)

// DMK - DEBUG
#define COUNT_FLOPS                      (0)

// EJB - SANITY CHECK
#define SANITY_CHECK                     (0)

// DMK - DEBUG
#define ENABLE_USER_EVENTS               (0)
#define PROJ_USER_EVENT_PATCH_FORCECHECKIN_CALLBACK  (1120)
#define PROJ_USER_EVENT_PATCH_INTEGRATE_CALLBACK     (1121)
#define PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_CALLBACK  (1130)
#define PROJ_USER_EVENT_SELFCOMPUTE_DOCALC_WORK      (1131)
#define PROJ_USER_EVENT_PAIRCOMPUTE_DOCALC_CALLBACK  (1140)
#define PROJ_USER_EVENT_PAIRCOMPUTE_DOCALC_WORK      (1141)
#define PROJ_USER_EVENT_MACHINEPROGRESS              (1150)

// DMK - DEBUG
#define ENABLE_NETWORK_PROGRESS          (0)
#if ENABLE_NETWORK_PROGRESS != 0
  #if ENABLE_USER_EVENTS != 0
    #define NetworkProgress  \
      {  \
        double __start_time__ = CkWallTimer();  \
        CmiMachineProgressImpl();  \
        traceUserBracketEvent(PROJ_USER_EVENT_MACHINEPROGRESS, __start_time__, CkWallTimer());  \
      }
  #else
    #define NetworkProgress  CmiMachineProgressImpl();
  #endif
#else
  #define NetworkProgress
#endif


#endif //__MD_CONFIG_H__
