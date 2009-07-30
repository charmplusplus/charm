#ifndef __SSE_FLOAT_H__
#define __SSE_FLOAT_H__

#include<emmintrin.h>
#include<iostream>



class SSEFloat
{

   public: __m128 val; 
           

   public:
    
           SSEFloat() {} 
  
           SSEFloat(float f) { val= _mm_set1_ps(f);}

           SSEFloat(float f0, float f1,float f2, float f3) {val = _mm_setr_ps(f0,f1,f2,f3);}                     

           /* Arithmetic Operators*/ 

           friend inline SSEFloat operator -(const SSEFloat &a) {SSEFloat c;c.val=_mm_sub_ps(_mm_setzero_ps(),a.val);return c;}

           friend inline SSEFloat operator +(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_add_ps(a.val,b.val);return c;}
                
           friend inline SSEFloat operator -(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_sub_ps(a.val,b.val);return c;}

           friend inline SSEFloat operator *(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_mul_ps(a.val,b.val);return c;}

           friend inline SSEFloat operator /(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_div_ps(a.val,b.val);return c;}

           friend inline SSEFloat sqrt      (const SSEFloat &a)                  { SSEFloat c;c.val= _mm_sqrt_ps(a.val);return c;} 


          friend inline SSEFloat operator +(float a, const SSEFloat &b) {SSEFloat c;c.val= _mm_add_ps(_mm_set1_ps(a),b.val);return c;}


          friend inline SSEFloat operator -(float a, const SSEFloat &b) {SSEFloat c;c.val= _mm_sub_ps(_mm_set1_ps(a),b.val);return c;}

          friend inline SSEFloat operator *(float a, const SSEFloat &b) {SSEFloat c;c.val= _mm_mul_ps(_mm_set1_ps(a),b.val);return c;}   
       
          friend inline SSEFloat operator /(float a, const SSEFloat &b) {SSEFloat c;c.val= _mm_div_ps(_mm_set1_ps(a),b.val);return c;}

           inline SSEFloat& operator +=(const SSEFloat &a) {val= _mm_add_ps(val,a.val);return *this;}
                
           inline SSEFloat& operator -=(const SSEFloat &a) {val= _mm_sub_ps(val,a.val);return *this;}

           inline SSEFloat& operator *=(const SSEFloat &a) {val= _mm_mul_ps(val,a.val);return *this;}

           inline SSEFloat& operator /=(const SSEFloat &a) {val= _mm_div_ps(val,a.val);return *this;}

          /*Logical Operators*/

           friend inline SSEFloat operator &(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_and_ps(a.val,b.val);return c;}

           friend inline SSEFloat operator |(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_or_ps(a.val,b.val);return c;}

           friend inline SSEFloat operator ^(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_xor_ps(a.val,b.val);return c;}

           friend inline SSEFloat andnot (const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_andnot_ps(a.val,b.val);return c;}

         /*Comparison Operators*/


            friend inline SSEFloat operator <(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_cmplt_ps(a.val,b.val);return c;}

            friend inline SSEFloat operator >(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_cmpgt_ps(a.val,b.val);return c;}

            friend inline SSEFloat operator ==(const SSEFloat &a, const SSEFloat &b) {SSEFloat c;c.val= _mm_cmpeq_ps(a.val,b.val);return c;}  
            
            friend inline SSEFloat operator <(const SSEFloat &a, float b) {SSEFloat c;c.val= _mm_cmplt_ps(a.val,_mm_set1_ps(b));return c;} 

            friend inline SSEFloat operator >(const SSEFloat &a, float b) {SSEFloat c;c.val= _mm_cmpgt_ps(a.val,_mm_set1_ps(b));return c;}

            friend inline SSEFloat max (const SSEFloat &a, SSEFloat &b) { SSEFloat c; c.val= _mm_max_ps(a.val,b.val);return c;}
 

        /*Masking Operations */

           friend inline int movemask( const SSEFloat &a) {return _mm_movemask_ps(a.val);}


        /*Store Operations*/

          friend inline void storeu(float *p, const SSEFloat &a) { _mm_storeu_ps(p,a.val);}

      //    friend void storeh(float *p, const SSEFloat &a) { _mm_storeh_pd(p,a.val);}


        //   void display();


 

};


/*
void Double::display()
{

storel(z,val);
//_mm_storeh_pd(z,val);
cout<<*z;
}
*/

/*
int main()
{

  float i[4];
  float *p=i;
// __m128d t1=_mm_setr_pd(3.0,0.0); __m128d t2 = _mm_setr_pd(5.0,0.0); 

  SSEFloat f1(2.0,1.0,4.0,5.0),f2(4.0,7.0,2.0,5.0),f4(25.0);

  SSEFloat d3 =   (f1 * f2)  ;  

  
  storeu(p,d3);
   
  cout<<*p;

  p++;cout<<*p;

  p++;cout<<*p;

  p++;cout<<*p;

//      d3 = d1 ^ d2;

 // d4 = sqrt(d2);

 // __m128d t =  _mm_and_pd(t1,t2);

//  cout << movemask(d3);
//   d3.display();

  //int i = movemask(d4);

  //cout<<i;

}


*/

#endif // __SSE_FLOAT_H__
