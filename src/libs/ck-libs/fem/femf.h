
       include 'idxlf.h'
       
       integer, parameter :: FEM_BYTE=IDXL_BYTE
       integer, parameter :: FEM_INT=IDXL_INT
       integer, parameter :: FEM_REAL=IDXL_REAL
       integer, parameter :: FEM_DOUBLE=IDXL_DOUBLE

       integer, parameter :: FEM_SUM=IDXL_SUM
       integer, parameter :: FEM_PROD=IDXL_PROD
       integer, parameter :: FEM_MAX=IDXL_MAX
       integer, parameter :: FEM_MIN=IDXL_MIN

       integer, parameter :: FEM_MESH_OUTPUT=0
       integer, parameter :: FEM_MESH_UPDATE=1
       integer, parameter :: FEM_MESH_FINALIZE=2

       integer, parameter :: FEM_TRIANGULAR=3
       integer, parameter :: FEM_TETRAHEDRAL=4
       integer, parameter :: FEM_HEXAHEDRAL=8
       integer, parameter :: FEM_QUADRILATERAL=4
       
       integer, parameter :: FEM_ENTITY_FIRST=1610000000
       integer, parameter :: FEM_NODE=(FEM_ENTITY_FIRST+0)
       integer, parameter :: FEM_ELEM=(FEM_ENTITY_FIRST+1000)
       integer, parameter :: FEM_ELEMENT=FEM_ELEM
       integer, parameter :: FEM_SPARSE=(FEM_ENTITY_FIRST+2000)
       integer, parameter :: FEM_EDGE=FEM_SPARSE
       integer, parameter :: FEM_FACE=FEM_SPARSE
       integer, parameter :: FEM_GHOST=10000
       
       integer, parameter :: FEM_DATA=0
       integer, parameter :: FEM_ATTRIB_TAG_MAX=1000000000
       integer, parameter :: FEM_ATTRIB_FIRST=1620000000
       integer, parameter :: FEM_CONN=(FEM_ATTRIB_FIRST+1)
       integer, parameter :: FEM_CONNECTIVITY=FEM_CONN
       integer, parameter :: FEM_SPARSE_ELEM=(FEM_ATTRIB_FIRST+2)
       integer, parameter :: FEM_COOR=(FEM_ATTRIB_FIRST+3)
       integer, parameter :: FEM_COORD=FEM_COOR
       integer, parameter :: FEM_COORDINATES=FEM_COOR
       integer, parameter :: FEM_GLOBALNO=(FEM_ATTRIB_FIRST+4)
       integer, parameter :: FEM_PARTITION=(FEM_ATTRIB_FIRST+5)
       integer, parameter :: FEM_SYMMETRIES=(FEM_ATTRIB_FIRST+6)
       integer, parameter :: FEM_NODE_PRIMARY=(FEM_ATTRIB_FIRST+7)

       external FEM_Print
       
       external FEM_Mesh_set_conn
       external FEM_Mesh_get_conn
       external FEM_Mesh_conn
       external FEM_Mesh_set_data
       external FEM_Mesh_get_data
       external FEM_Mesh_data
       external FEM_Mesh_data_layout
       external FEM_Mesh_data_offset
       external FEM_Mesh_pup
       
       external FEM_Set_Mesh

       external FEM_Set_Node
       external FEM_Get_Node
       external FEM_Set_Elem
       external FEM_Get_Elem
       external FEM_Set_Elem_Conn_r
       external FEM_Get_Elem_Conn_r
       external FEM_Set_Node_Data_r
       external FEM_Get_Node_Data_r
       external FEM_Set_Elem_Data_r
       external FEM_Get_Elem_Data_r
       
       external FEM_Set_Sparse
       external FEM_Set_Sparse_Elem
       external FEM_Get_Sparse

       external FEM_Add_Linear_Periodicity
       external FEM_Sym_Coordinates

       external FEM_Set_Sym_Nodes
       external FEM_Get_Sym

       external FEM_Set_Partition

       external FEM_Add_Ghost_Layer
       external FEM_Add_Ghost_Elem

       external FEM_Get_Comm_Nodes

       external FEM_Reduce_Field
       external FEM_Reduce
       external FEM_Update_Field
       external FEM_Update_Ghost_Field
       external FEM_Read_Field
       integer, external :: foffsetof

       external FEM_Barrier
       external FEM_Get_Ghost_List

       integer, external :: FEM_Register
       external FEM_Migrate 

       external FEM_Update_mesh

       interface
       
       subroutine FEM_Init(comm) 
           integer,intent(in) :: comm
       end subroutine
       integer function FEM_My_partition()
       end function
       integer function FEM_Num_partitions()
       end function 
       double precision function FEM_Timer()
       end function
       subroutine FEM_Done()
       end subroutine
       subroutine FEM_Print_partition()
       end subroutine
       subroutine FEM_Mesh_print(mesh)
          integer, intent(in) :: mesh
       end subroutine
       
       integer function FEM_Mesh_allocate()
       end function
       integer function FEM_Mesh_copy(mesh)
          integer, intent(in) :: mesh
       end function
       subroutine FEM_Mesh_deallocate(mesh) 
          integer, intent(in) :: mesh
       end subroutine
       
       integer function FEM_Mesh_read(prefix,partNo,nParts)
          integer, intent(in) :: partNo,nParts
          character (LEN=*), intent(in) :: prefix
       end function
       subroutine FEM_Mesh_write(mesh,prefix,partNo,nParts) 
          integer, intent(in) :: mesh
          integer, intent(in) :: partNo,nParts
          character (LEN=*), intent(in) :: prefix
       end subroutine
       
       integer function FEM_Mesh_assemble(nParts,parts)
          integer, intent(in) :: nParts
          integer, intent(in) :: parts(nParts)
       end function
       subroutine FEM_Mesh_partition(mesh,nParts,parts) 
          integer, intent(in) :: mesh
          integer, intent(in) :: nParts
          integer, intent(out) :: parts(nParts)
       end subroutine
       
       integer function FEM_Mesh_recv(source,tag,comm)
          integer, intent(in) :: source,tag,comm
       end function
       subroutine FEM_Mesh_send(mesh,source,tag,comm) 
          integer, intent(in) :: mesh
          integer, intent(in) :: source,tag,comm
       end subroutine
       
       integer function FEM_Mesh_reduce(mesh,master,comm)
          integer, intent(in) :: mesh
          integer, intent(in) :: master,comm
       end function
       integer function FEM_Mesh_broadcast(mesh,master,comm) 
          integer, intent(in) :: mesh
          integer, intent(in) :: master,comm
       end function
       
       
       integer function FEM_Mesh_default_read()
       end function
       integer function FEM_Mesh_default_write()
       end function
       subroutine FEM_Mesh_set_default_read(mesh)
         integer, intent(in) :: mesh
       end subroutine
       subroutine FEM_Mesh_set_default_write(mesh)
         integer, intent(in) :: mesh
       end subroutine
       
       function FEM_Mesh_get_length(mesh,ent)
         integer, intent(in) :: mesh,ent
         integer :: FEM_Mesh_get_length
       end function
       subroutine FEM_Mesh_set_length(mesh,ent,newLength)
         integer, intent(in) :: mesh,ent,newLength
       end subroutine
       function FEM_Mesh_get_width(mesh,ent)
         integer, intent(in) :: mesh,ent
         integer :: FEM_Mesh_get_width
       end function
       subroutine FEM_Mesh_set_width(mesh,ent,attr,newWidth)
         integer, intent(in) :: mesh,ent,attr,newWidth
       end subroutine
       function FEM_Mesh_get_datatype(mesh,ent,attr)
         integer, intent(in) :: mesh,ent,attr
         integer :: FEM_Mesh_get_datatype
       end function
       
       integer function FEM_Mesh_get_entities(mesh,entities)
         integer, intent(in) :: mesh
         integer, intent(out) :: entities(:)
       end function
       integer function FEM_Mesh_get_attributes(mesh,entity,attrs)
         integer, intent(in) :: mesh, entity
         integer, intent(out) :: attrs(:)
       end function
       
       
       function FEM_Create_simple_field(base_type, vec_len)
          integer, intent(in)  :: base_type, vec_len
          integer  :: FEM_Create_Simple_Field
       end function 
       
       function FEM_Create_field(base_type, vec_len, init_offset, distance)
          integer, intent(in)  :: base_type, vec_len, init_offset, distance
          integer  :: FEM_Create_Field
       end function 
       
       function FEM_Comm_shared(mesh,ent)
          integer, intent(in)  :: mesh,ent
          integer  :: FEM_Comm_shared
       end function 
       function FEM_Comm_ghost(mesh,ent)
          integer, intent(in)  :: mesh,ent
          integer  :: FEM_Comm_ghost
       end function 
       
       function FEM_Get_node_ghost()
         integer :: FEM_Get_node_ghost
       end function
       function FEM_Get_elem_ghost(elemType)
         integer, intent(in) :: elemType
         integer :: FEM_Get_elem_ghost
       end function    

       subroutine FEM_Exchange_ghost_lists(elemType,nIdx,localIdx)
         integer,intent (in) :: elemType,nIdx
         integer, intent(in) :: localIdx(nIdx)
       end subroutine

       function FEM_Get_ghost_list_length()
         integer :: FEM_Get_ghost_list_length
       end function

       function FEM_Get_sparse_length(sID)
         integer :: FEM_Get_sparse_length
         integer, intent(in) ::sID
       end function

       subroutine FEM_Serial_split(nChunk)
         integer,intent (in) :: nChunk
       end subroutine
       subroutine FEM_Serial_begin(chunk)
         integer,intent (in) :: chunk
       end subroutine
       subroutine FEM_Serial_read(chunk,nChunks)
         integer,intent (in) :: chunk,nChunks
       end subroutine
       subroutine FEM_Serial_assemble()
       end subroutine
       
       function FEM_Get_comm_partners()
         integer :: FEM_Get_comm_partners
       end function
       function FEM_Get_comm_partner(partnerNo)
         integer, intent(in) :: partnerNo
         integer :: FEM_Get_comm_partner
       end function
       function FEM_Get_comm_count(partnerNo)
         integer, intent(in) :: partnerNo
         integer :: FEM_Get_comm_count
       end function
       
       end interface

