#include "conv-mach.h" 

#ifdef CMK_BLUEGENEP

#include <spi/UPC.h>
#include <spi/UPC_Events.h>
#include <iostream>


/// An initcall that registers the idle time reducer idleTimeReduction()
void initBGP_UPC_Counters(void) {

    // every process on the node calls BGP_UPC_Initialize()
    BGP_UPC_Initialize();


    // just one rank per node sets the counter config and zeros the counters
    
    // counter_mode = 0, 1, 2, 3 (plus some others … see UPC.h)
    // counter_trigger = BGP_UPC_CFG_LEVEL_HIGH, BGP_UPC_CFG_EDGE_DEFAULT
    
    BGP_UPC_Mode_t counter_mode = BGP_UPC_MODE_0;
    BGP_UPC_Event_Edge_t counter_trigger = BGP_UPC_CFG_LEVEL_HIGH;
    
    BGP_UPC_Initialize_Counter_Config(counter_mode, counter_trigger);

    BGP_UPC_Zero_Counter_Values();
    BGP_UPC_Start(0);

}





/// print out the counters
void printBGP_UPC_Counters(void) {
    
 
    BGP_UPC_Stop();
   
    // Should look at BGP_TORUS_XP_NO_TOKENS to determine if there is contention
   
    int64_t cxp = BGP_UPC_Read_Counter_Value(BGP_TORUS_XP_PACKETS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus xp = " << cxp << std::endl;

    int64_t cxm = BGP_UPC_Read_Counter_Value(BGP_TORUS_XM_PACKETS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus xm = " << cxm << std::endl;

    int64_t cyp = BGP_UPC_Read_Counter_Value(BGP_TORUS_YP_PACKETS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus yp = " << cyp << std::endl;

    int64_t cym = BGP_UPC_Read_Counter_Value(BGP_TORUS_YM_PACKETS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus ym = " << cym << std::endl;

    int64_t czp = BGP_UPC_Read_Counter_Value(BGP_TORUS_ZP_PACKETS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus zp = " << czp << std::endl;

    int64_t czm = BGP_UPC_Read_Counter_Value(BGP_TORUS_ZM_PACKETS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus zm = " << czm << std::endl;



    int64_t cxpc = BGP_UPC_Read_Counter_Value(BGP_TORUS_XP_32BCHUNKS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus xp chunks = " << cxpc << std::endl;

    int64_t cxmc = BGP_UPC_Read_Counter_Value(BGP_TORUS_XM_32BCHUNKS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus xm chunks = " << cxmc << std::endl;

    int64_t cypc = BGP_UPC_Read_Counter_Value(BGP_TORUS_YP_32BCHUNKS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus yp chunks = " << cypc << std::endl;

    int64_t cymc = BGP_UPC_Read_Counter_Value(BGP_TORUS_YM_32BCHUNKS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus ym chunks = " << cymc << std::endl;

    int64_t czpc = BGP_UPC_Read_Counter_Value(BGP_TORUS_ZP_32BCHUNKS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus zp chunks = " << czpc << std::endl;

    int64_t czmc = BGP_UPC_Read_Counter_Value(BGP_TORUS_ZM_32BCHUNKS, BGP_UPC_READ_EXCLUSIVE);
    std::cout << "BGP_UPC_Read_Counter_Value returned torus zm chunks = " << czmc << std::endl;


    //    Save the counter values from the counter_data structure …
    BGP_UPC_Start(0);
    
}
#else


void printBGP_UPC_Counters(void){
// do nothing since not on BGP
}



void initBGP_UPC_Counters(void){
// do nothing since not on BGP
}



#endif
