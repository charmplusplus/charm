      integer, parameter :: NetFEM_POINTAT=1
      integer, parameter :: NetFEM_WRITE=2
      integer, parameter :: NetFEM_COPY=10
      external NetFEM_Nodes
      external NetFEM_Elements
      external NetFEM_Vector_field
      external NetFEM_Scalar_field
      external NetFEM_Vector
      external NetFEM_Scalar
      interface
       function NetFEM_Begin(source,timestep,dim,flavor)
         integer,intent (in) :: source,timestep,dim,flavor
         integer :: NetFEM_Begin
       end function
       subroutine NetFEM_End(netfem)
         integer, intent(in) :: netfem
       end subroutine
      end interface
       
