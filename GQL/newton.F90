!*************************************************************************
#include "arnoldi.f"
#include "GMRESm.F90"
#include "NewtonHook.F90"
!*************************************************************************
!  INPUTS:
!  fld.start:
!     initial guess state.
!
!  shifts.in:
!     Ndt	  	number of timesteps taken in one period
!     sx sz T		initial guesses for shift and period
!  
!  newton_params.in:   
!     T_fixed		Set 1 for TW, 0 for PO.
!     m  nits  ncdg	GMRES dimension, max newton its, num eigvecs
!     rel_err		newton relative error |X(T)-X(0)| / |X(0)|
!     del  mndl  mxdl   trust-region parameters
!     gtol  epsJ	GMRES error for step, eps in Jacobian approx
!  recommended newton_params.in:
!     1
!     150  100  10
!     1d-8
!     -1d0 1d-7 1d+7
!     1d-4 1d-6
!
!-------------------------------------------------------------------------
!  Newton vector:
!    current X stored in new_x
!    current error F(X) stored in new_fx
!    X(1) = sx
!    X(2) = sz
!    X(3) = T
!    X(9:)= u, velocity vector
!  Constraints:
!    dX . dX/dx = 0,  dX . dX/dz = 0
!       no update parallel to theta or z shift.
!    (F(X)-X). dX/dt = 0 .
!       no update along direction of trajectory.
!  Jacobian approximation:
!    dF(X)/dX . dX = (F(X+eps.dX)-F(X))/eps
!
!*************************************************************************
 module orbit
   use shared_module, only : NX,NY,NZ,DELTA_T,H_BAR,DY,CU1,CU2,CU3,CP, &
                             CIKX,CIKZ,NXP,NPROCY,NPROCZ,NPROCS,RANK,RANKY, &
                             RANKZ,IERROR,MPI_COMM_Y,MPI_COMM_Z,status
   implicit none
   save   
   integer, parameter     :: TNKZ=2*(NZ/3), NKX=NX/3

   logical          :: T_fixed
   integer          :: m, nits, ncgd
   double precision :: rel_err
   double precision :: del, mndl, mxdl
   double precision :: gtol, epsJ

   integer, parameter :: ms = 1
!   integer, parameter :: n = 8 + 8*(JEND-JSTART+1)*(TNKZ+1)*(NKX+1)
   integer :: JS_, JE_, n, NY_TOT
   double precision   :: tol, scaleU(3), scaleT
   integer :: info, ndts(0:9), io_save1
   double precision, allocatable :: w(:),DYY(:)
   complex*16, allocatable, dimension(:,:,:) :: GCU1,GCU2,GCU3,GCP

 end module orbit


!*************************************************************************
 subroutine newton_channel()
!*************************************************************************
   use orbit
   use shared_module, only : NX,NY,NZ,DELTA_T,H_BAR,DY,CU1,CU2,CU3,CP, &
                             CIKX,CIKZ,NPROCY,RANK,RANKY,RANKZ,NXP,IERROR
   use newton
!   use shared_module, only : JS_=>JSTART, JE_=>JEND
!   use shared_module, only : CP  !Yongyun added
   use shared_module, only : TN_=>TNKZ, NK_=>NKX
   USE diabloio_module, only : SAVE_FLOW
   implicit none
   double precision :: d
   include "mpif.h"
   external :: getrhs, multJ, multJp, saveorbit 
   double precision, external :: dotprod, dotprod_ms
   integer :: nms, i1, i2, i3, p, j
   INTEGER :: I,NR,YS,YE
   character(4) :: cnumi
   double precision :: DYTEMP(0:NY+1)

! Initialize variables
   IF(RANK.EQ.0) THEN
     NY_TOT=(NY-1)*NPROCY+1
   ELSE
     NY_TOT=2
   END IF
   JS_=2
   JE_=NY_TOT
   n=8 + 8*(JE_-JS_+1)*(TNKZ+1)*(NKX+1)
   
   Allocate(GCU1(0:NX/2,0:NZ+1,0:NY_TOT+1),GCU2(0:NX/2,0:NZ+1,0:NY_TOT+1), &
            GCU3(0:NX/2,0:NZ+1,0:NY_TOT+1),GCP(0:NX/2,0:NZ+1,0:NY_TOT+1))
   Allocate(w(JS_:JE_),DYY(0:NY_TOT+1))

   DYY(:)=0.0D0
   IF(RANK.EQ.0) THEN
     DYY(0:NY+1)=DY(0:NY+1)
     DO NR=1,NPROCY-1
       CALL MPI_RECV(DYTEMP,NY+2, &
                    MPI_DOUBLE_PRECISION, &
                    NR,999,MPI_COMM_WORLD,STATUS,IERROR)
       YS=1; YE=NY-1
       IF(NR.EQ.NPROCY-1) YE=NY+1
       DO J=YS,YE
         DYY(J+NR*(NY-1))=DYTEMP(J)
       END DO
     END DO
   ELSE IF(RANKZ.EQ.0) THEN
     CALL MPI_SEND(DY,NY+2, &
                  MPI_DOUBLE_PRECISION, &
                  0,999,MPI_COMM_WORLD,STATUS,IERROR)
   END IF

!   IF(RANK.EQ.0) WRITE(*,*) 'DYY(0:NY_TOT+1)=',DYY(0:NY_TOT+1)

   DO j = JS_, JE_
     w(j) = dsqrt(dble(DYY(j)))
   END DO
   
!   if(JS_/=JSTART) stop 'JSTART err'
!   if(JE_/=JEND  ) stop 'JEND   err'
   if(TN_/=TNKZ  ) stop 'TNKX   err'
   if(NK_/=NKX   ) stop 'NKX    err'

   open(99,status='old',file='newton_params.in')
   read(99,*) i1
   read(99,*) m, nits, ncgd
   read(99,*) rel_err
   read(99,*) del, mndl, mxdl
   read(99,*) gtol, epsJ
   close(99)
   T_fixed = (i1==1)
   nms = n * ms
   allocate(new_x(nms))
   allocate(new_fx(nms))
  
   open(99,status='old',file='shifts.in')
   do p = 0, ms-1
      read(99,*) ndts(p)
   end do
   do p = 0, ms-1
      new_x(p*n+1:p*n+8) = 0d0
      read(99,*) new_x(p*n+1), new_x(p*n+2), new_x(p*n+3)
   end do
   close(99)

!IF(RANK.EQ.0) WRITE(*,*) 'w(0:NY_TOT+1)=',w(JS_:JE_)
   call get_scales()
! Broadcast scaleT
   CALL MPI_BCAST(scaleT,1,MPI_DOUBLE_PRECISION,0, &
         MPI_COMM_WORLD,ierror)

   CALL LOCAL2GLOBAL(CU1,GCU1)
   CALL LOCAL2GLOBAL(CU2,GCU2)
   CALL LOCAL2GLOBAL(CU3,GCU3)
   CALL LOCAL2GLOBAL(CP,GCP)

   call cucp2vec(GCU1,GCU2,GCU3,GCP, new_x)
   call load_ms_states(0)
   d = dotprod_ms(-1,new_x,new_x)

   CALL MPI_BCAST(d,1,MPI_DOUBLE_PRECISION,0, &
          MPI_COMM_WORLD,ierror)   
   tol  = rel_err * dsqrt(d)
   del  = del  * dsqrt(d)
   mndl = mndl * dsqrt(d)
   mxdl = mxdl * dsqrt(d)
   
   info = 1
   call newtonhook(getrhs, multJ, multJp, saveorbit, dotprod_ms, &
                   m, nms, gtol, tol, del, mndl, mxdl, nits, info)

   CALL MPI_BCAST(ncgd,1,MPI_INTEGER,0, &
         MPI_COMM_WORLD,ierror)
   CALL MPI_BCAST(info,1,MPI_INTEGER,0, &
         MPI_COMM_WORLD,ierror)   
   if(ncgd>0 .and. info==0)  call getEigen() 

   Deallocate(GCU1,GCU2,GCU3,GCP,w)
!*************************************************************************
 end subroutine newton_channel   
!*************************************************************************
!-------------------------------------------------------------------------
! get shooting points
!-------------------------------------------------------------------------
 subroutine load_ms_states(i)
   use orbit
   implicit none
   integer, intent(in) :: i
   if(ms/=1) stop 'NOT IMPLEMENTED'
 end subroutine load_ms_states


!-------------------------------------------------------------------------
!  save state
!-------------------------------------------------------------------------
 subroutine io_save_state
 USE orbit
 USE hdf5_module
 CHARACTER*55 FNAME
 CHARACTER*4 FNUM
 WRITE(FNUM,'(I4.4)') io_save1
 FNAME='out'//FNUM//'.h5'
 call WriteHDF5(FNAME,.TRUE.)
!   call SAVE_FLOW(.false.) 
 end subroutine io_save_state


!-------------------------------------------------------------------------
!  timestep, shift
!-------------------------------------------------------------------------
 subroutine steporbit(ndts_,x, y)
!   use orbit, only : n, scaleT
   Use orbit
   use shared_module
   use channel_module
   use diabloio_module, only : SAVE_FLOW
   implicit none
   integer :: N_
   integer,          intent(in)  :: ndts_
   double precision, intent(in)  :: x(n)
   double precision, intent(out) :: y(n)
   double precision  :: par(3)
   integer :: ndt, ia(8)
   logical :: fexist

   par = x(:3) * scaleT

   CALL MPI_BCAST(par,3,MPI_DOUBLE_PRECISION,0, &
         MPI_COMM_WORLD,ierror)

   call vec2cucp(x, GCU1,GCU2,GCU3,GCP)

   CALL GLOBAL2LOCAL(GCU1,CU1)
   CALL GLOBAL2LOCAL(GCU2,CU2)
   CALL GLOBAL2LOCAL(GCU3,CU3)
   CALL GLOBAL2LOCAL(GCP,CP)

   if(ndts_/=1) then
      DELTA_T = par(3) / dble(ndts_)
      H_BAR(1)=DELTA_T*(8.0/15.0)
      H_BAR(2)=DELTA_T*(2.0/15.0)
      H_BAR(3)=DELTA_T*(5.0/15.0)
   end if

   do ndt = 1, ndts_
!- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!						EXECUTE TIMESTEP
!- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!WRITE(*,*) 'RANK=',RANK,DELTA_T,PAR
      TIME_STEP = TIME_STEP + 1
!      IF (RANK.EQ.0.AND.MOD(TIME_STEP,SAVE_FLOW_INT).EQ.0) THEN
!         WRITE(6,*) 'Now beginning TIME_STEP = ',TIME_STEP
!      END IF
      DO RK_STEP=1,3
         IF (TIME_AD_METH.EQ.1) CALL RK_CHAN_1
         IF (TIME_AD_METH.EQ.2) CALL RK_CHAN_2
      END DO
          
      if(ndts_/=1) then
         TIME=TIME+DELTA_T
         FIRST_TIME=.FALSE.
!         IF (MOD(TIME_STEP,SAVE_STATS_INT).EQ.0) CALL SAVE_STATS_CHAN(.FALSE.)
!        IF (MOD(TIME_STEP,SAVE_FLOW_INT).EQ.0)  CALL SAVE_FLOW(.FALSE.)    
         IF (MOD(TIME_STEP,5).EQ.0)  CALL TRACE
!         CALL SPEED_CALC   
      end if
!      
      if(modulo(TIME_STEP,10)==0) then
         inquire(file='RUNNING', exist=fexist)
         if(fexist) cycle
!         call date_and_time(values=ia)
!         end_time = ia(3)*86400+ia(5)*3600+ia(6)*60+ia(7)+ia(8)*0.001
!         print*,'Elapsed Time (sec): ',end_time-start_time
!         print*,'Secs per Iteration: ',(end_time-start_time)/TIME_STEP
         stop 'RUNNING deleted!'
      end if
!- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   end do

   CALL LOCAL2GLOBAL(CU1,GCU1)
   CALL LOCAL2GLOBAL(CU2,GCU2)
   CALL LOCAL2GLOBAL(CU3,GCU3)
   CALL LOCAL2GLOBAL(CP,GCP)

   call cucp2vec(GCU1,GCU2,GCU3,GCP, y)
   if(ndts_/=1)  call vec_shift(-par(1),-par(2),y, y)

 end subroutine steporbit
 

!-------------------------------------------------------------------------
! get scale factors for U and T
!-------------------------------------------------------------------------
 subroutine get_scales()
   use newton
   use orbit
   use shared_module, only : NX,NY,NZ,DELTA_T,H_BAR,DY,CU1,CU2,CU3,CP, &
                             CIKX,CIKZ
   USE diabloio_module, only : SAVE_FLOW
   implicit none
   double precision :: d
   double precision, external :: dotprod
   integer :: p


INTEGER :: I,J,K
   
   scaleU(1) = 1d0
   scaleU(2) = 3d0
   scaleU(3) = 3d0

! MPI local to global
   CALL LOCAL2GLOBAL(CU1,GCU1)
   CALL LOCAL2GLOBAL(CU2,GCU2)
   CALL LOCAL2GLOBAL(CU3,GCU3)
   CALL LOCAL2GLOBAL(CP,GCP)

   call cucp2vec(GCU1,GCU2,GCU3,GCP, new_x)
   d = dotprod(-1,new_x,new_x)
   scaleT   = new_x(3) / dsqrt(d)
   do p = 0, ms-1
      new_x(p*n+1:p*n+3) = new_x(p*n+1:p*n+3) / scaleT
   end do

   IF(RANK.EQ.0) print*, 'Scales:', real(scaleU), real(scaleT)
   
 end subroutine get_scales
 

!-------------------------------------------------------------------------
!  save state as a single vector
!-------------------------------------------------------------------------
 subroutine cucp2vec(c1,c2,c3,pp, v)
   use orbit
   implicit none
   complex*16, intent(in) :: c1(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16, intent(in) :: c2(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16, intent(in) :: c3(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16, intent(in) :: pp(0:NX/2,0:NZ+1,0:NY_TOT+1)
   double precision, intent(out) :: v(n)
   double precision, save :: s(3)
   integer :: i_, i,j,k

   i_ = 9
   do j = JS_, JE_
      s = scaleU * w(j)
      do k = 0, TNKZ
         do i = 0, NKX
            v(i_  ) =  dble(c1(i,k,j)) * s(1)
            v(i_+1) = dimag(c1(i,k,j)) * s(1)  
            v(i_+2) =  dble(c2(i,k,j)) * s(2)
            v(i_+3) = dimag(c2(i,k,j)) * s(2)
            v(i_+4) =  dble(c3(i,k,j)) * s(3)
            v(i_+5) = dimag(c3(i,k,j)) * s(3)
            v(i_+6) =  dble(pp(i,k,j)) * s(1)
            v(i_+7) = dimag(pp(i,k,j)) * s(1)
            i_ = i_ + 8
         end do
      end do
   end do

 end subroutine cucp2vec


!-------------------------------------------------------------------------
!  get state from vector
!-------------------------------------------------------------------------
 subroutine vec2cucp(v, c1,c2,c3,pp)
   use orbit
   implicit none
   double precision, intent(in) :: v(n)
   complex*16, intent(out) :: c1(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16, intent(out) :: c2(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16, intent(out) :: c3(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16, intent(out) :: pp(0:NX/2,0:NZ+1,0:NY_TOT+1)
   double precision, save :: s(3)
   integer :: i_, i,j,k

   c1=CMPLX(0.0D0,0.0D0)
   c2=CMPLX(0.0D0,0.0D0)
   c3=CMPLX(0.0D0,0.0D0)
   pp=CMPLX(0.0D0,0.0D0)

   i_ = 9
   do j = JS_, JE_
      s = scaleU * w(j)
      do k = 0, TNKZ
         do i = 0, NKX
            c1(i,k,j) = dcmplx(v(i_  ),v(i_+1)) / s(1)
            c2(i,k,j) = dcmplx(v(i_+2),v(i_+3)) / s(2)
            c3(i,k,j) = dcmplx(v(i_+4),v(i_+5)) / s(3)
            pp(i,k,j) = dcmplx(v(i_+6),v(i_+7)) / s(1)
            i_ = i_ + 8
         end do
      end do
   end do
   
 end subroutine vec2cucp


!-------------------------------------------------------------------------
!  shift a vector,  s(x+sx,z+sz)=v(x,z) => s(x,z)=v(x-sx,z-sz)
!-------------------------------------------------------------------------
 subroutine vec_shift(sx,sz,v, s)
   use orbit
   implicit none
   double precision, intent(in)  :: sx, sz
   double precision, intent(in)  :: v(n)
   double precision, intent(out) :: s(n)
   complex*16 :: c1(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16 :: c2(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16 :: c3(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16 :: pp(0:NX/2,0:NZ+1,0:NY_TOT+1)
   complex*16 :: cc, eikx(0:NKX), eikz(0:TNKZ)
   integer :: i, j, k
   do i = 0, NKX		! KX(I)=2.pi.i.I/LX
      eikx(i) = exp(-CIKX(i)*sx)
   end do
   do k = 0, TNKZ		! KZ(K)=2.pi.i.K/LZ
      eikz(k) = exp(-CIKZ(k)*sz)
   end do
   call vec2cucp(v, c1,c2,c3,pp)
   do j = JS_, JE_
      do k = 0, TNKZ
         do i = 0, NKX
            cc = eikx(i)*eikz(k)
            c1(i,k,j) = cc * c1(i,k,j)
            c2(i,k,j) = cc * c2(i,k,j)
            c3(i,k,j) = cc * c3(i,k,j)
            pp(i,k,j) = cc * pp(i,k,j)
         end do
      end do
   end do
   call cucp2vec(c1,c2,c3,pp, s)
 end subroutine vec_shift


!-------------------------------------------------------------------------
!  tangent to shift of v in x (i==1) or z (i==2) 
!-------------------------------------------------------------------------
 subroutine getshiftdir(i,v, s)
   use orbit
   implicit none
   integer,          intent(in)  :: i
   double precision, intent(in)  :: v(n)
   double precision, intent(out) :: s(n)
   double precision :: v_(n)
   
   if(i==1) call vec_shift(epsJ,0d0,v, v_)
   if(i==2) call vec_shift(0d0,epsJ,v, v_)
   s = (v_-v) / epsJ
 
 end subroutine getshiftdir


!-------------------------------------------------------------------------
! dot prod of line vector
!-------------------------------------------------------------------------
 double precision function dotprod(n_,a,b)
   use orbit
   implicit none
   integer,          intent(in) :: n_
   double precision, intent(in) :: a(n), b(n)
   double precision :: d,d_
   integer :: n1,nn

   nn = n
   n1 = 1
   if(n_==-1) n1 = 9
   d = dot_product(a(n1:nn),b(n1:nn))
   dotprod = d
 end function dotprod


!-------------------------------------------------------------------------
! dot prod for multiple shooting
!-------------------------------------------------------------------------
 double precision function dotprod_ms(n_,a,b)
   use orbit
   implicit none
   integer,          intent(in) :: n_
   double precision, intent(in) :: a(n*ms), b(n*ms)
   double precision, external :: dotprod
   double precision :: d 
   integer :: p
   
   d = 0d0
   do p = 0, ms-1
      d = d + dotprod(n_,a(p*n+1),b(p*n+1))
   end do
   dotprod_ms = d

 end function dotprod_ms


!-------------------------------------------------------------------------
!  function to be minimised   
!-------------------------------------------------------------------------
 subroutine getrhs(nms,x, y)
   use orbit
   implicit none
   integer,          intent(in)  :: nms
   double precision, intent(in)  :: x(nms)
   double precision, intent(out) :: y(nms)
   double precision :: x_(n), y_(n*ms)
   integer :: p, p1
   
   do p = 0, ms-1
      p1 = modulo(p+1,ms)
      x_ = x(p*n+1:p*n+n)
      call steporbit(ndts(p),x_, y_(p1*n+1))
   end do
   y = y_ - x					! diff
   do p = 0, ms-1
      y(p*n+1:p*n+8) = 0d0			! constraints, rhs=0
   end do

 end subroutine getrhs


!-------------------------------------------------------------------------
!  Jacobian of function + lhs of constraints on update
!-------------------------------------------------------------------------
 subroutine multJ(nms,x, y)
   use newton
   use orbit,    only : n, ms, T_fixed, epsJ, DELTA_T
   USE shared_module, only : RANK,IERROR
   implicit none
   include "mpif.h"
   integer,          intent(in)  :: nms
   double precision, intent(in)  :: x(nms)
   double precision, intent(out) :: y(nms)   
   double precision, external :: dotprod, dotprod_ms
   double precision :: eps, s(n*ms)
   integer :: p
    				! (F(x0+eps.x)-F(x0))/eps
   eps = dsqrt(dotprod_ms(1,x,x))
   CALL MPI_BCAST(eps,1,MPI_DOUBLE_PRECISION,0, &
          MPI_COMM_WORLD,ierror)
   if(eps==0d0)  stop 'multJ: eps=0 (1)'
   eps = epsJ * dsqrt(dotprod_ms(1,new_x,new_x)) / eps
   CALL MPI_BCAST(eps,1,MPI_DOUBLE_PRECISION,0, &
          MPI_COMM_WORLD,ierror)
   if(eps==0d0)  stop 'multJ: eps=0 (2)'
   y = new_x + eps*x
   call getrhs(nms,y, s)
   y = (s - new_fx) / eps

   do p = 0, ms-1
       				! no update in shift directions 
      call getshiftdir(1,new_x(p*n+1), s) 
      y(p*n+1) = dotprod(-1,s,x(p*n+1))
      call getshiftdir(2,new_x(p*n+1), s) 
      y(p*n+2) = dotprod(-1,s,x(p*n+1))
      				! no update in trajectory direction
      call steporbit(1,new_x(p*n+1), s)
      s(:n) = (s(:n) - new_x(p*n+1:p*n+n)) / dble(DELTA_T)
      y(p*n+3) = dotprod(-1,s,x(p*n+1))
      				! special cases
      if(T_fixed)  y(p*n+3) = x(p*n+3)
   end do
   
 end subroutine multJ
 

!-------------------------------------------------------------------------
!  preconditioner for multJ   
!-------------------------------------------------------------------------
 subroutine multJp(n, x)
   implicit none
   integer,          intent(in)    :: n
   double precision, intent(inout) :: x(n)
 end subroutine multJp


!-------------------------------------------------------------------------
!  called at each newton iteration   
!-------------------------------------------------------------------------
 subroutine saveorbit()
   use newton
   use orbit
   use shared_module, only : NX,NY,NZ,DELTA_T,H_BAR,DY,CU1,CU2,CU3,CP, &
                             CIKX,CIKZ
  
   implicit none
   include "mpif.h"
   double precision, external :: dotprod_ms
   double precision :: norm_x, p(3)
   integer :: p_, ndt
     
   ndt = sum(ndts(0:ms-1))
   norm_x = dsqrt(dotprod_ms(-1,new_x,new_x))

   IF(RANK.EQ.0) THEN   
   open(99,status='unknown',access='append',file='newton.dat')
   if(new_nits==0)  write(99,*) ndt, m, n
   write(99,'(2I6,4e13.5)')  &
      new_nits, new_gits, new_tol, new_del, new_tol/norm_x, norm_x
   close(99)

   open(99,status='unknown',access='append',file='shifts.dat')
   do p_ = 0, ms-1
      if(new_nits==0)  write(99,*) ndts(p_)
   end do
   do p_ = 0, ms-1
      p = new_x(p_*n+1:p_*n+3)*scaleT
      write(99,'(1I6,3e24.16)')  new_nits, p(1), p(2), p(3)
   end do
   close(99)
   END IF

   do p_ = 0, ms-1
      call vec2cucp(new_x(p_*n+1), GCU1,GCU2,GCU3,GCP)
      CALL GLOBAL2LOCAL(GCU1,CU1)
      CALL GLOBAL2LOCAL(GCU2,CU2)
      CALL GLOBAL2LOCAL(GCU3,CU3)
      CALL GLOBAL2LOCAL(GCP,CP)

      if(ms==1) io_save1 = new_nits
      if(ms/=1) io_save1 = new_nits*10 + p_
      call io_save_state()
   end do  

 end subroutine saveorbit
 
 
!-------------------------------------------------------------------------
!  linearised time step for arnoldi
!-------------------------------------------------------------------------
 subroutine multA(x,y0,epsA, y)
   use newton, only : new_x
   use orbit,  only : ndts, n, ms
   USE shared_module, only : IERROR
   implicit none
   include "mpif.h"
   double precision, intent(in)  :: x(n), y0(n*ms), epsA
   double precision, intent(out) :: y(n)
   double precision, external :: dotprod
   double precision :: eps
   integer :: p
   
   y = x   
   do p = 0, ms-1
      eps = dsqrt(dotprod(-1,y,y))
      CALL MPI_BCAST(eps,1,MPI_DOUBLE_PRECISION,0, &
             MPI_COMM_WORLD,ierror)
      if(eps==0d0)  stop 'multA: eps=0 (1)'
      eps = epsA * dsqrt(dotprod(-1,new_x(p*n+1),new_x(p*n+1))) / eps
      CALL MPI_BCAST(eps,1,MPI_DOUBLE_PRECISION,0, &
             MPI_COMM_WORLD,ierror)
      if(eps==0d0)  stop 'multA: eps=0 (2)'
      y = new_x(p*n+1:p*n+n) + eps*y
      y(:8) = new_x(p*n+1:p*n+8)
      call steporbit(ndts(p),y, y)
      y = (y - y0(p*n+1:p*n+n)) / eps
   end do

 end subroutine multA


!-------------------------------------------------------------------------
! get eigenvalues and eigenfunctions about state
!-------------------------------------------------------------------------
 subroutine getEigen()
   use orbit 
   use newton
   use shared_module, only : NX,NY,NZ,DELTA_T,H_BAR,DY,CU1,CU2,CU3,CP, &
                             CIKX,CIKZ,IERROR,RANK
   implicit none
   include "mpif.h"
   double precision, external :: dotprod
   double precision, allocatable :: q(:,:)
   double precision :: sv(n), b(n,ncgd*2), wr(ncgd*3), wi(ncgd*3), h(m,m)
   double precision :: y0(n*ms), eps,epsA, st,sz,t, rerr
   integer :: i,j,k, p, ifail

   allocate(q(n,m))
   epsA = epsJ
   rerr = min(rel_err,gtol)
   t    = sum(new_x(3::n))*scaleT
   do p = 0, ms-1
      call steporbit(ndts(p),new_x(p*n+1), y0(p*n+1))
   end do
   						! random initial input 
   call vec_shift(1d-1,1d-1,new_x, sv)
      						! main arnoldi loop
   k = 0
   do while(.true.)
      sv(:8) = 0d0
      call arnold(n,k,m,ncgd,dotprod,rerr,sv,h,q,b,wr,wi,ifail)
      CALL MPI_BCAST(ifail,1,MPI_INTEGER,0, &
             MPI_COMM_WORLD,ierror)
      if(k>ncgd+2) then
         IF(RANK.EQ.0) print*, 'getEigen: k =', k
         do i = 1, ncgd
            IF(RANK.EQ.0) print*, 'getEigen: ', real(dlog(dsqrt(wr(i)**2+wi(i)**2))/t)
         end do 
      end if
      if(ifail==-1) then
         IF(RANK.EQ.0) print*, ' arnoldi converged!'
         exit
      else if(ifail==0) then
         call multA(sv,y0,epsA, sv)
      else if(ifail==1) then
         IF(RANK.EQ.0) print*, 'getEigen: WARNING: arnoldi reached max its'
         exit
      else if(ifail>=2) then
         IF(RANK.EQ.0) print*, 'getEigen: arnoldi error'
         deallocate(q)
         return
      end if   
   end do
                                ! magnitude and angle of Floq mult
   wr(ncgd*2+1:) = sqrt(wr(:ncgd)**2+wi(:ncgd)**2)
   wi(ncgd*2+1:) = atan2(wi(:ncgd),wr(:ncgd)) !in (-pi,pi]
                                ! convert Floquet to growth rates
   wr(ncgd+1:ncgd*2) = wr(:ncgd)
   wi(ncgd+1:ncgd*2) = wi(:ncgd)
   call logeig(n,ncgd,b,wr,wi,t,ifail)
   deallocate(q)
   IF(RANK.EQ.0) THEN
      				! save eigvals and Floq multipliers
   open(99,status='unknown',file='arnoldi.dat')
   write(99,*) ' its  = ', k
   write(99,*) ' epsA = ', real(epsA)
   write(99,*) ' T    = ', t
   do i = 1, ncgd
      				! Floq exp, Floq mult, magn & angle
      write(99,'(1I4,6e16.8)') i, wr(i), wi(i),  &
         wr(ncgd+i), wi(ncgd+i), wr(ncgd*2+i), wi(ncgd*2+i)
   end do
   close(99) 
   END IF
      				! make norm of eigvecs same as that
                                ! of converged state and save
   do i = 0, ncgd*2
      if(i==0) call vec2cucp(new_x,  GCU1,GCU2,GCU3,GCP)
      if(i/=0) call vec2cucp(b(1,i), GCU1,GCU2,GCU3,GCP)
      CALL GLOBAL2LOCAL(GCU1,CU1)
      CALL GLOBAL2LOCAL(GCU2,CU2)
      CALL GLOBAL2LOCAL(GCU3,CU3)
      CALL GLOBAL2LOCAL(GCP,CP)
      io_save1 = 1000 + i 
      call io_save_state()
   end do

 end subroutine getEigen 

 SUBROUTINE LOCAL2GLOBAL(CU,GCU)
   USE ORBIT
   IMPLICIT NONE
   include "mpif.h"
   COMPLEX*16, INTENT(IN) :: CU(0:NXP,0:NZ+1,0:NY+1)
   COMPLEX*16, INTENT(OUT) :: GCU(0:NX/2,0:NZ+1,0:NY_TOT+1)
   COMPLEX*16, ALLOCATABLE, DIMENSION(:,:,:) :: TEMP
   INTEGER :: I,J,K,NR,RKY,RKZ,YS,YE,XE

   ALLOCATE(TEMP(0:NXP,0:NZ+1,0:NY+1))

   IF(RANK.EQ.0) THEN
     DO J=0,NY+1; DO I=0,NXP
       GCU(I,:,J)=CU(I,:,J)
     END DO; END DO
! Receive data for root
     DO NR=1,NPROCS-1
       CALL MPI_RECV(TEMP,(NXP+1)*(NZ+2)*(NY+2), &
                    MPI_DOUBLE_COMPLEX, &
                    NR,999,MPI_COMM_WORLD,STATUS,IERROR)       
       RKY=MOD(NR,NPROCY)
       RKZ=NR/NPROCY
       YS=1; YE=NY-1; XE=NXP-1
       IF(RKY.EQ.NPROCY-1) YE=NY+1
       IF(RKY.EQ.0) YS=0
       IF(RKZ.EQ.NPROCZ-1) XE=NXP
!       WRITE(*,*) 'NR=',NR,'RKY=',RKY,'RKZ=',RKZ
       DO J=YS,YE; DO I=0,XE
         GCU(I+RKZ*NXP,:,J+RKY*(NY-1))=TEMP(I,:,J)
       END DO; END DO
     END DO
   ELSE
! Send data out for other processors
     TEMP=CU
     CALL MPI_SEND(TEMP,(NXP+1)*(NZ+2)*(NY+2), &
                  MPI_DOUBLE_COMPLEX, &
                  0,999,MPI_COMM_WORLD,STATUS,IERROR)
   END IF

   DEALLOCATE(TEMP)
 
 END SUBROUTINE LOCAL2GLOBAL

 SUBROUTINE GLOBAL2LOCAL(GCU,CU)
   USE ORBIT
   IMPLICIT NONE
   include "mpif.h"
   COMPLEX*16, INTENT(IN) :: GCU(0:NX/2,0:NZ+1,0:NY_TOT+1)
   COMPLEX*16, INTENT(OUT) :: CU(0:NXP,0:NZ+1,0:NY+1)
   COMPLEX*16, ALLOCATABLE, DIMENSION(:,:,:) :: TEMP
   INTEGER :: I,J,NR,RKY,RKZ

   ALLOCATE(TEMP(0:NXP,0:NZ+1,0:NY+1))

   IF(RANK.EQ.0) THEN
     DO J=0,NY+1; DO I=0,NXP
       CU(I,:,J)=GCU(I,:,J)
     END DO; END DO
! Send data from root
     DO NR=1,NPROCS-1
       RKY=MOD(NR,NPROCY)
       RKZ=NR/NPROCY
!       WRITE(*,*) 'NR=',NR,'RKY=',RKY,'RKZ=',RKZ
        DO J=0,NY+1; DO I=0,NXP
          TEMP(I,:,J)=GCU(I+RKZ*NXP,:,J+RKY*(NY-1))
        END DO; END DO
        CALL MPI_SEND(TEMP,(NXP+1)*(NZ+2)*(NY+2), &
                    MPI_DOUBLE_COMPLEX, &
                    NR,999,MPI_COMM_WORLD,STATUS,IERROR)
     END DO
   ELSE
! Receive data out for other processors
     CALL MPI_RECV(TEMP,(NXP+1)*(NZ+2)*(NY+2), &
                  MPI_DOUBLE_COMPLEX, &
                  0,999,MPI_COMM_WORLD,STATUS,IERROR)
     CU=TEMP
   END IF

   DEALLOCATE(TEMP)

 END SUBROUTINE GLOBAL2LOCAL
