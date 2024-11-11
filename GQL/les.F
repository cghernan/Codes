      MODULE les_module
      USE shared_module
      USE mpi_module
      USE hdf5_module

      REAL*8 TEMP(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  Sij(0:NX+1,0:NZP+1,0:NY+1,1:6)
      COMPLEX*16 CSij(0:NXP,0:NZ+1,0:NY+1,1:6)
      REAL*8  Aij(0:NX+1,0:NZP+1,0:NY+1,1:9)
      COMPLEX*16 CAij(0:NXP,0:NZ+1,0:NY+1,1:9)
      REAL*8 cross
      REAL*8 RHS(0:NY+1),RHS2(0:NY+1),D2GYF(0:NY+1)
      INTEGER J1,J2
      INTEGER U_BC_LOWER,W_BC_LOWER,U_BC_UPPER,W_BC_UPPER

       REAL*8  betaV(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  beta11(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  beta22(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  beta33(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  beta12(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  beta13(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  beta23(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  TEMPY(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  fac(0:NX+1,0:NZP+1,0:NY+1)
       REAL*8  NU_T2(0:NX+1,0:NZP+1,0:NY+1)

!      EQUIVALENCE (Sij,CSij)

      CONTAINS
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      subroutine les_chan
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C This subroutine models the terms owing to the subgrid scale stress
C if the computation is to be treated as an LES not a DNS
C This subroutine should be called when the velocity is in fourier space 
C in the periodic directions, on output, the velocity will be 
C in physical space.
C It is assumed that the test filter and the LES filter are performed
C by the same operation
C On output S1 should contain |S| which may be used again in les_chan_th
C if for the subgrid scalar dissipation

      integer i,j,k,l,m,ij

      real*8 S1_mean(1:NY)
      real*8 NU_T_mean(1:NY)
 

! Variables for Dynamic Smagorinsky model:
      real*8 C_SMAG,C_VREMAN
      parameter (C_SMAG=0.03d0)
      parameter (C_VREMAN=0.03d0)
      real*8 DELTA_Y(0:NY+1),DELTA_YF(0:NY+1) 
      real*8 alpha,beta,gamma
      real*8 denominator_sum
      real*8 l_plus,y_plus,A_plus
      real*8 damps,u_tau,TAUW1,TAUW2
      real*8 del_x,del_z
      real*8 dissip,ein,dissip2,dissipx
      REAL*8 STAT(1:NY,1:10)
      CHARACTER*35 FNAME
      CHARACTER*10 GNAME

! Array to store the velocity index for each component of the strain rate tensor
      integer U_index1(6)
      integer U_index2(6)

! Here, alpha is the test/LES filter width ratio
      parameter (alpha=2.d0)
! beta is the LES/grid filter width ratio in the streamwise and spanwise directions
! I'm using grid filter: Yongyun.
      parameter (beta=1.d0)
! gamma is the LES/grid filter width ratio in the wall-normal direction  
      parameter (gamma=1.d0)

! First, for all models, apply boundary conditions to the velocity field
! (fill ghost cells) to ensure accurate calculation of gradients
C Apply Boundary conditions to velocity field
      IF (USE_MPI) THEN
        CALL APPLY_BC_VEL_MPI
      ELSE
        CALL APPLY_BC_VEL_LOWER
        CALL APPLY_BC_VEL_UPPER
      END IF
      
      if (LES_MODEL_TYPE.EQ.1) then
!     Constant Smagorinsky model

! First, compute the rate of strain tensor S_ij

      call compute_strain_chan

      IF (RANKZ.EQ.0) THEN
        UME=dble(CU1(0,0,:))
      ELSE
        UME=0.0D0
      END IF
      TAUW1=0.0D0; TAUW2=0.0D0
      IF(RANKZ.EQ.0) THEN
        TAUW1=(UME(2)-UME(1))/(GYF(2)-GYF(1))
        TAUW2=(UME(NY)-UME(NY-1))/(GYF(NY)-GYF(NY-1))
      END IF
      CALL MPI_BCAST(TAUW1,1,MPI_DOUBLE_PRECISION,0,
     &   MPI_COMM_WORLD,ierror)
      CALL MPI_BCAST(TAUW2,1,MPI_DOUBLE_PRECISION,NPROCY-1,
     &   MPI_COMM_WORLD,ierror)
      IF(U_BC_YMAX.EQ.1) TAUW2=-TAUW1
      u_tau=SQRT(NU*(TAUW1-TAUW2)*0.5D0)

! Convert the velocity to physical space
      call FFT_XZ_TO_PHYSICAL(CU1,U1,0,NY+1)
      call FFT_XZ_TO_PHYSICAL(CU2,U2,0,NY+1)
      call FFT_XZ_TO_PHYSICAL(CU3,U3,0,NY+1)
           
! Compute |S| at GYF points, store in S1
! Interpolation to GYF points is easy since by definition
! GYF points are exactly midway between neighboring GY points
      DO J=1,NY
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=SQRT(
     &                2.d0*Sij(I,K,J,1)**2
     &               +4.d0*(0.5d0*(Sij(I,K,J+1,4)+Sij(I,K,J,4)))**2
     &               +4.d0*Sij(I,K,J,5)**2
     &               +2.d0*Sij(I,K,J,2)**2
     &               +4.d0*(0.5d0*(Sij(I,K,J+1,6)+Sij(I,K,J,6)))**2 
     &               +2.d0*Sij(I,K,J,3)**2 )
          END DO
        END DO
      END DO
! Extend |S| to ghost cells
      CALL HALO_ARRAY_REAL(S1)
      IF(RANKY.EQ.0) THEN
      DO K=0,NZP-1
        DO I=0,NXM
          S1(I,K,0)=S1(I,K,1)
        END DO
      END DO
      END IF
!
      IF(RANKY.EQ.NPROCY-1) THEN
      DO K=0,NZP-1
        DO I=0,NXM
          S1(I,K,NY+1)=S1(I,K,NY)
        END DO
      END DO
      END IF

! Now, compute |S|*S_ij, storing in Sij
! First compute at GYF points 
      DO J=1,NY
        DO K=0,NZP-1
          DO I=0,NXM
            Sij(I,K,J,1)=S1(I,K,J)*Sij(I,K,J,1)
            Sij(I,K,J,5)=S1(I,K,J)*Sij(I,K,J,5)
! CSij(:,:,:,2) is added through an implicit eddy viscosity
            Sij(I,K,J,2)=0.d0
            Sij(I,K,J,3)=S1(I,K,J)*Sij(I,K,J,3)
          END DO
        END DO
      END DO
! Now, compute at GY points, interpolating |S|
      DO J=1,NY+1
        DO K=0,NZP-1
          DO I=0,NXM
! |S| interpolated to GY point 
            TEMP(I,K,J)= (S1(I,K,J)*DYF(j-1)+S1(I,K,J-1)*DYF(j))
     &                  /(2.d0*DY(j))
! The terms dU1/dy and dU3/dy in CSij(:,:,:,4) and CSij(:,:,:,6) respectively
! are subtracted out from Sij here since they are treated implicitly
! in eddy viscosity terms
            Sij(I,K,J,4)=TEMP(I,K,J)
     &        *(Sij(I,K,J,4)-0.5*(U1(I,K,J)-U1(I,K,J-1))/DY(j))
            Sij(I,K,J,6)=TEMP(I,K,J)
     &        *(Sij(I,K,J,6)-0.5*(U3(I,K,J)-U3(I,K,J-1))/DY(j))

           
          END DO
        END DO
      END DO

! Convert |S|*S_ij to Fourier space
      do ij=1,6   
      CALL FFT_XZ_TO_FOURIER(Sij(0,0,0,ij),CSij(0,0,0,ij),0,NY+1)      
      end do
      
! We now have |S|*S_ij stored in Sij in Physical space
! Compute u_tau to get l_plus    
!      u_tau=abs(dble(CU1(0,0,2)-CU1(0,0,1))/(GYF(2)-GYF(1)))+
!     &  abs(dble(CU1(0,0,NY)-CU1(0,0,NY-1))/(GYF(NY)-GYF(NY-1)))

! Compute the filter lengthscale
! Absorb -2.d0*C_SMAG**2 here for effienciency
      l_plus=u_tau/NU
      A_plus=25.

! Define filter width considering dealiasing.
      del_x=DX(1)*3./2. 
      del_z=DZ(1)*3./2.
      
      DO J=1,NY
! At GYF points:
! Constant Smagorinsky with wall-damping function
        y_plus=(1.-ABS(GYF(J)))*l_plus    
        DELTA_YF(J)=-2.d0*C_SMAG**2
     &     *(del_x*beta*DYF(J)*gamma*del_z*beta)**(2.d0/3.d0)
     &     *(1.-exp(-1.*(abs(y_plus/A_plus)**3.d0)))
    
!         damps=(1-exp(-1.*((y_plus/A_plus)**2)))**2
! Wall Damping
!        DELTA_YF(J)=
!     &    -2.d0*(0.1d0*(1.d0-exp((-GYF(J)/(NU*25.d0))**3.d0)))**2
!     &            *(DX(1)*beta*DYF(J)*gamma*DZ(1)*beta)**(2.d0/3.d0)
!        write(*,*) damps,y_plus
      END DO
! Extend to ghost cells 
      IF(RANKY.EQ.0)
     &  DELTA_YF(0)=DELTA_YF(1)
      IF(RANKY.EQ.NPROCY-1)
     &  DELTA_YF(NY+1)=DELTA_YF(NY)      

      DO J=1,NY+1
! At GY points:
! Constant Smagorinsky with wall-damping function
        y_plus=(1.-ABS(GY(J)))*l_plus
        DELTA_Y(J)=-2.d0*C_SMAG**2 
     &        *(del_x*beta*DY(J)*gamma*del_z*beta)**(2.d0/3.d0)
     &     *(1.-exp(-1.*(abs(y_plus/A_plus)**3.d0)))
! Wall Damping
!        DELTA_Y(J)=
!     &    -2.d0*(0.1d0*(1.d0-exp((-GY(J)/(NU*25.d0))**3.d0)))**2
!     &            *(DX(1)*beta*DY(J)*gamma*DZ(1)*beta)**(2.d0/3.d0)
      END DO

! Get the eddy viscosity at GY points
! NU_T = (C_S^2 * DELTA^2)*|S|

      DO J=1,NY+1
        DO K=0,NZP-1
          DO I=0,NXM
            NU_T(I,K,J)=-0.5d0*DELTA_Y(J)*TEMP(I,K,J)
          END DO
        END DO
      END DO
      
! Now, compute TAU, store in the corresponging Sij
      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CSij(I,K,J,1)=DELTA_YF(J)*CSij(I,K,J,1)
            CSij(I,K,J,5)=DELTA_YF(J)*CSij(I,K,J,5)
! CSij(:,:,:,2) is added through an implicit eddy viscosity
!            CSij(I,K,J,2)=DELTA_YF(J)*CSij(I,K,J,2)
            CSij(I,K,J,3)=DELTA_YF(J)*CSij(I,K,J,3)
          END DO
        END DO
      END DO
      DO J=1,NY+1 
        DO K=0,TNKZ
          DO I=0,NXP-1
            CSij(I,K,J,4)=DELTA_Y(J)*CSij(I,K,J,4)
            CSij(I,K,J,6)=DELTA_Y(J)*CSij(I,K,J,6)
          END DO
        END DO
      END DO
      

! tau_ij is now contained in CSij in Fourier space


! Skip the process for dynamic models
      goto 9999     

      ELSE IF (LES_MODEL_TYPE.EQ.2) then
!     Vreman model

! First, compute the rate of strain tensor S_ij

      call compute_strain_chan
      call compute_anis_chan

      IF (RANKZ.EQ.0) THEN
        UME=dble(CU1(0,0,:))
      ELSE
        UME=0.0D0
      END IF
      TAUW1=0.0D0; TAUW2=0.0D0
      IF(RANKZ.EQ.0) THEN
        TAUW1=(UME(2)-UME(1))/(GYF(2)-GYF(1))
        TAUW2=(UME(NY)-UME(NY-1))/(GYF(NY)-GYF(NY-1))
      END IF
      CALL MPI_BCAST(TAUW1,1,MPI_DOUBLE_PRECISION,0,
     &   MPI_COMM_WORLD,ierror)
      CALL MPI_BCAST(TAUW2,1,MPI_DOUBLE_PRECISION,NPROCY-1,
     &   MPI_COMM_WORLD,ierror)
      IF(U_BC_YMAX.EQ.1) TAUW2=-TAUW1
      u_tau=SQRT(NU*(TAUW1-TAUW2)*0.5D0)
! Convert the velocity to physical space
      call FFT_XZ_TO_PHYSICAL(CU1,U1,0,NY+1)
      call FFT_XZ_TO_PHYSICAL(CU2,U2,0,NY+1)
      call FFT_XZ_TO_PHYSICAL(CU3,U3,0,NY+1)
           

! Now, compute at GY points, interpolating |S|
      DO J=1,NY+1
        DO K=0,NZP-1
          DO I=0,NXM
! |S| interpolated to GY point 
! The terms dU1/dy and dU3/dy in CSij(:,:,:,4) and CSij(:,:,:,6) respectively
! are subtracted out from Sij here since they are treated implicitly
! in eddy viscosity terms
            Sij(I,K,J,4)=
     & (Sij(I,K,J,4)-0.5*(U1(I,K,J)-U1(I,K,J-1))/DY(j))
            Sij(I,K,J,6)=
     & (Sij(I,K,J,6)-0.5*(U3(I,K,J)-U3(I,K,J-1))/DY(j))
          
          END DO
        END DO
      END DO

! Define filter width considering dealiasing.
      del_x=(DX(1)*3./2.)**2 
      del_z=(DZ(1)*3./2.)**2
      
     
! Extend to ghost cells 

!!!!!!!!!!!!!!!!!!!!!!!!!!VREMAN PART

       DO J=1,NY
       DO K=0,NZP-1
       DO I=0,NXM          
         beta11(I,K,J)=
     &del_x*(aij(i,k,j,1)**2)+
     &(DYF(j)**2)*(0.5d0*(aij(i,k,j,9)+aij(i,k,j+1,9)))
     &**2+
     &del_z*(aij(i,k,j,7)**2)

            beta12(I,K,J)=
     &del_x*aij(i,k,j,1)*(0.5d0*(aij(i,k,j,4)+aij(i,k,j+1,4)))+
     &(dYF(j)**2)*0.5d0*(aij(i,k,j,9)+aij(i,k,j+1,9))
     &*aij(i,k,j,2)+
     &del_z*aij(i,k,j,7)*(0.5d0*(aij(i,k,j,6)+aij(i,k,j+1,6)))

            beta13(I,K,J)=
     &del_x*aij(i,k,j,1)*aij(i,k,j,5)+
     &(dYF(j)**2)*(0.5d0*(aij(i,k,j,2)+aij(i,k,j+1,2)))*
     &         (0.5d0*(aij(i,k,j,8)+aij(i,k,j+1,8)))+
     &del_z*aij(i,k,j,7)*aij(i,k,j,3)

        beta23(I,K,J)=
     &del_x*(0.5d0*(aij(i,k,j,4)+aij(i,k,j+1,4)))*aij(i,k,j,5)+
     &(dYF(j)**2)*aij(i,k,j,2)*0.5*(aij(i,k,j,8)
     &+aij(i,k,j+1,8))+
     &+del_z*0.5d0*(aij(i,k,j,6)+aij(i,k,j+1,6))*aij(i,k,j,3)

        beta22(I,K,J)=
     &del_x*(0.5d0*(aij(i,k,j,4)+aij(i,k,j+1,4)))**2+
     &(dYF(j)**2)*(aij(i,k,j,2)**2)+
     &del_z*(0.5d0*(aij(i,k,j,6)+aij(i,k,j+1,6))**2)

        beta33(I,K,J)=
     &del_x*aij(i,k,j,5)**2+
     &(dYf(j)**2)*(0.5d0*(aij(i,k,j,8)+aij(i,k,j+1,8))**2)+
     &del_z*aij(i,k,j,3)**2

       END DO
       END DO
       END DO

       
       DO K=0,NZP-1
       DO I=0,NXM 
       DO J=1,NY
         betaV(I,K,J)=
     &   dmax1(0.d0,
     &   beta11(i,k,j)*beta22(i,k,j)
     & -(beta12(i,k,j)**2)
     & +beta11(i,k,j)*beta33(i,k,j)
     & -(beta13(i,k,j)**2)
     & +beta22(i,k,j)*beta33(i,k,j)
     & -(beta23(i,k,j)**2)             
     & )
     
     
        fac(i,k,j)=max(0.d0,sqrt(
     &  betaV(i,k,j)/(
     &   aij(i,k,j,1)*aij(i,k,j,1)+
     &  aij(i,k,j,2)*aij(i,k,j,2)+    
     &  aij(i,k,j,3)*aij(i,k,j,3)+ 
     &  aij(i,k,j,5)*aij(i,k,j,5)+
     &  aij(i,k,j,7)*aij(i,k,j,7)+
     &  (0.5d0*(aij(i,k,j,4)+aij(i,k,j+1,4))**2)+
     &  (0.5d0*(aij(i,k,j,6)+aij(i,k,j+1,6))**2)+
     &  (0.5d0*(aij(i,k,j,8)+aij(i,k,j+1,8))**2)+
     &  (0.5d0*(aij(i,k,j,9)+aij(i,k,j+1,9))**2)
     &  )))


       END DO
       END DO
       END DO   
       
       DO K=0,NZP-1
         DO I=0,NXM
          DO J=1,NY+1
             TempY(i,k,j)=
     &       (fac(I,K,J)*DYF(j-1)+fac(I,K,J-1)*DYF(j))
     &        /(2.d0*DY(j))

             NU_T(i,k,j)=C_Vreman*
     &       TempY(i,k,j)

         END DO
!        NU_T(i,k,1)=0.d0
!        NU_T(i,k,NY)=0
        END DO
       END DO

       CALL HALO_ARRAY_REAL(NU_T)

       DO K=0,NZP-1
         DO I=0,NXM
          DO J=1,NY
             NU_T2(i,k,j)=-2.d0*
     &       (NU_T(I,K,J)*DY(j+1)+NU_T(I,K,J+1)*DY(j))
     &        /(2.d0*DYF(j))
         END DO
!        NU_T2(i,k,1)=0.d0
        IF(RANKY.EQ.NPROCY-1) then
         NU_T2(i,k,NY+1)=0
        ELSEIF (RANKY.EQ.0) then
         NU_T2(i,k,0)=0.d0
        END IF
!       NU_T2(i,k,NY)=0
       END DO
      END DO

      CALL HALO_ARRAY_REAL(NU_T2)

!      write(*,*) 'CSij(10,10,30,1)les ',CSij(10,10,30,1)

! Now, compute TAU, store in the corresponging Sij
       DO J=1,NY
         DO K=0,NZP-1
           DO I=0,NXM
            Sij(I,K,J,1)=NU_T2(i,k,J)*Sij(I,K,J,1)
            Sij(I,K,J,5)=NU_T2(i,k,J)*Sij(I,K,J,5)
            Sij(I,K,J,3)=NU_T2(i,k,J)*Sij(I,K,J,3)
           END DO
         END DO
       END DO
       DO J=1,NY+1
         DO K=0,NZP-1
           DO I=0,NXM
            Sij(I,K,J,4)=NU_T2(i,k,J)*Sij(I,K,J,4)
            Sij(I,K,J,6)=NU_T2(i,k,J)*Sij(I,K,J,6)
           END DO
         END DO
       END DO

!      write(*,*) 'CSij(10,10,30,1)les2 ',CSij(10,10,30,1)

      DO ij=1,6
            CALL FFT_XZ_TO_FOURIER(Sij(0,0,0,ij),CSij(0,0,0,ij),0,NY+1)
      END DO

!      ELSE IF (LES_MODEL_TYPE.eq.3) then
      ELSE

        pause 'Error, unsupported LES_MODEL_TYPE chosen'
      END IF

! For the Constant Smargorinsky model: added by Yongyun.
9999  continue

! When using a Near-wall model, don't use LES at the wall
      IF ((U_BC_LOWER.EQ.3).or.(W_BC_LOWER.EQ.3)) then
         J1=2
      ELSE 
         J1=1
      END IF

      IF ((U_BC_UPPER.EQ.3).or.(W_BC_UPPER.EQ.3)) then
         J2=NY-1
      ELSE
         J2=NY
      END IF

! Zero Tau above J2+1 where it is not used
      DO ij=1,6
        DO J=J2+2,NY+1
          DO K=0,TNKZ
            DO I=0,NXP-1
              CSij(I,K,J,ij)=0.d0
            END DO
          END DO
        END DO
      END DO

! Now, add the subgrid scale forcing to CFi
! (This includes the subgrid scale stress as an explicit R-K term

      DO J=J1,J2
        DO K=0,TNKZ
          DO I=0,NXP-1
            CF1(I,K,J)=CF1(I,K,J)
     &                -CIKX(I)*CSij(I,K,J,1)
     &                -(CSij(I,K,J+1,4)-CSij(I,K,J,4))/DYF(j)
     &                -CIKZ(K)*CSij(I,K,J,5)
            CF3(I,K,J)=CF3(I,K,J)
     &                -CIKX(I)*CSij(I,K,J,5)
     &                -(CSij(I,K,J+1,6)-CSij(I,K,J,6))/DYF(J)
     &                -CIKZ(K)*CSij(I,K,J,3)   
          END DO
        END DO
      END DO
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
           CF2(I,K,J)=CF2(I,K,J)
     &                -CIKX(I)*CSij(I,K,J,4)
     &                -(CSij(I,K,J,2)-CSij(I,K,J-1,2))/DY(j)*0.0
     &                -CIKZ(K)*CSij(I,K,J,6)
         END DO
        END DO
      END DO
             	   

        
      IF ((MOD(TIME_STEP,SAVE_STATS_INT).EQ.0).AND.(RK_STEP.EQ.1)) THEN
! Get plane averages
        do j=1,NY
          S1_mean(j)=0.d0
          NU_T_mean(j)=0.d0
          do i=0,NXM
          do k=0,NZP-1
            S1_mean(j)=S1_mean(j)+S1(I,K,J)
            NU_T_mean(j)=NU_T_mean(j)+NU_T(I,K,J)
          end do
          end do
          S1_mean(j)=S1_mean(j)/dble(NX*NZ)
          NU_T_mean(j)=NU_T_mean(j)/dble(NX*NZ)
        end do
        CALL MPI_ALLREDUCE(MPI_IN_PLACE,S1_mean,NY,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_Z,ierror)

#ifdef HDF5
        FNAME='mean_les.h5'
        GNAME='statistics_les'
        IF(RANKZ.EQ.0) THEN
          DO J=1,NY
            STAT(J,1)=REAL(J)
          END DO
          STAT(1:NY,2)=GYF(1:NY)
          DO ij=1,6
            STAT(1:NY,2+ij)=DBLE(CSij(0,0,1:NY,ij))
          END DO
          STAT(1:NY,9)=NU_T_MEAN(1:NY)
          STAT(1:NY,10)=S1_MEAN(1:NY)
!
          CALL WriteHDF5_STATISTICS(FNAME,GNAME,STAT,10)
      END IF
#else
        IF(USE_MPI) THEN
          open(42,file='mean_les'//trim(MPI_IO_NUM)//'.txt',
     &         form='formatted',status='unknown')
        ELSE
          open(42,file='mean_les.txt',form='formatted',status='unknown')
        END IF
        write(42,*) TIME_STEP,TIME,DELTA_T
!        write(42,*) 'Smargorinsky constant=',C_SMAG
        do j=1,NY
          write(42,420) j,GYF(J)
     &    ,(dble(CSij(0,0,J,ij)),ij=1,6),
     &     NU_T_MEAN(j),
!     &     -1.*rhs(j),
     &     S1_mean(j)
!          WRITE(*,'(2E20.10)') GYF(J),NU_T_mean(j)
        end do
#endif
      END IF
  420     format(I3,' ',9(F20.9,' '))
  450     format(5E15.6)    

      RETURN
      END subroutine les_chan


!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      subroutine compute_strain_chan
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C This subroutine computes S_ij for the filtered velocity field
C The input velocity field should be in fourier space in the periodic
C directions.
C For use in the LES model in channel flow (2 periodic directions)

      integer I,J,K,ij

      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CSij(I,K,J,1)=CIKX(I)*CU1(I,K,J)
            CSij(I,K,J,2)=(CU2(I,K,J+1)-CU2(I,K,J))/DYF(j) 
            CSij(I,K,J,3)=CIKZ(K)*CU3(I,K,J)  
            CSij(I,K,J,5)=0.5d0*(CIKZ(K)*CU1(I,K,J)+CIKX(I)*CU3(I,K,J))
          END DO
        END DO
      END DO
      DO J=1,NY+1
        DO K=0,TNKZ
          DO I=0,NXP-1
            CSij(I,K,J,4)=0.5d0*( (CU1(I,K,J)-CU1(I,K,J-1))/DY(j)
     &                          +CIKX(I)*CU2(I,K,J) ) 
            CSij(I,K,J,6)=0.5d0*( CIKZ(K)*CU2(I,K,J)
     &                         +(CU3(I,K,J)-CU3(I,K,J-1))/DY(j) )
          END DO
        END DO
      END DO  

      DO ij=1,6
        CALL HALO_ARRAY_COMPLEX(CSij(:,:,:,ij))
      END DO

! Convert rate of strain tensor to physical space
      do ij=1,6
        call FFT_XZ_TO_PHYSICAL(CSij(0,0,0,ij),Sij(0,0,0,ij),0,NY+1)
      end do 
! We now have S_ij in Physical space
      RETURN
      END subroutine compute_strain_chan
!
      subroutine compute_anis_chan
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C This subroutine computes S_ij for the filtered velocity field
C The input velocity field should be in fourier space in the periodic
C directions.
C For use in the LES model in channel flow (2 periodic directions)

      integer I,J,K,ij

      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CAij(I,K,J,1)=CIKX(I)*CU1(I,K,J)
            CAij(I,K,J,2)=(CU2(I,K,J+1)-CU2(I,K,J))/DYF(j) 
            CAij(I,K,J,3)=CIKZ(K)*CU3(I,K,J)  
            CAij(I,K,J,5)=CIKX(I)*CU3(I,K,J)
            CAij(I,K,J,7)=CIKZ(K)*CU1(I,K,J)
          END DO
        END DO
      END DO
      DO J=1,NY+1
        DO K=0,TNKZ
          DO I=0,NXP-1
            CAij(I,K,J,4)=CIKX(I)*CU2(I,K,J) 
            CAij(I,K,J,6)=CIKZ(K)*CU2(I,K,J)
            CAij(I,K,J,8)=(CU3(I,K,J)-CU3(I,K,J-1))/DY(j)  
            CAij(I,K,J,9)=(CU1(I,K,J)-CU1(I,K,J-1))/DY(j)                
          END DO
        END DO
      END DO  

      DO ij=1,9
        CALL HALO_ARRAY_COMPLEX(CAij(:,:,:,ij))
      END DO

! Convert rate of strain tensor to physical space
      do ij=1,9
        call FFT_XZ_TO_PHYSICAL(CAij(0,0,0,ij),Aij(0,0,0,ij),0,NY+1)
      end do 
! We now have S_ij in Physical space
      RETURN
      END subroutine compute_anis_chan
      END MODULE les_module

