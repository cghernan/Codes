      MODULE channel_module
      USE shared_module
      USE solvers_module
      USE les_module
      USE fft_module
      CONTAINS
C******************************************************************************|
C channel.f, the channel-flow solvers for diablo.                  VERSION 0.9
C These solvers were written primarily by John R. Taylor (spring 2005).
C Modified by Carlos Gonzalez Hernandez (2024).
C******************************************************************************|
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE INIT_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C Initialize any constants here
      INTEGER J, N

! Open a NetCDF file for writing standardized output
!      call NETCDF_OPEN_STATS_CHAN

      PI=4.D0*ATAN(1.D0)

! Defined starting and ending indeces in the wall-bounded direction
       IF (RANK.EQ.0) 
     &     write(*,*) 'U_BC_YMIN: ',U_BC_YMIN
        IF (U_BC_YMIN.EQ.0) THEN
          JSTART=2
        ELSE IF (U_BC_YMIN.EQ.1) THEN
          JSTART=1
        ELSE
          JSTART=2
        END IF
! Now, set the indexing for the scalar equations
        DO N=1,N_TH
          IF (TH_BC_YMIN(N).EQ.0) THEN
            JSTART_TH(N)=2
          ELSE IF (TH_BC_YMIN(N).EQ.1) THEN
            JSTART_TH(N)=1
          ELSE
            JSTART_TH(N)=2
          END IF
        END DO
        IF (RANK.EQ.0) 
     &       write(*,*) 'U_BC_YMAX: ',U_BC_YMAX
        IF (U_BC_YMAX.EQ.0) THEN
          JEND=NY-1
        ELSE IF (U_BC_YMAX.EQ.1) THEN
          JEND=NY
        ELSE
          JEND=NY-1
        END IF

! Set the upper and lower limits of timestepping of the scalar equations
        DO N=1,N_TH
        IF (TH_BC_YMAX(N).EQ.0) THEN
          JEND_TH(N)=NY-1
        ELSE IF (TH_BC_YMAX(N).EQ.1) THEN
          JEND_TH(N)=NY
        ELSE
          JEND_TH(N)=NY-1
        END IF
        END DO

      RETURN
      END SUBROUTINE INIT_CHAN

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE RK_CHAN_1
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C Main time-stepping algorithm for the channel-flow case.
C This algorithm uses Crank-Nicolson for all terms involving vertical
C derivatives (viscous and nonlinear) and 3rd order Runge-Kutta for the
C rest of the terms
C INPUTS  (in Fourier space):  CUi, CP, and (if k>1) CFi at (k-1)  (for i=1,2,3)
C OUTPUTS (in Fourier space):  CUi, CP, and (if k<3) CFi at (k)
C Each RK step, there are 14 FFT calls. 11 storage variables are used.     
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|

      INTEGER I,J,K,N,ISTART,NN,MM,II    
      REAL*8 TEMP1, TEMP2, TEMP3, TEMP4, TEMP5, UBULK
      REAL*8 K_P, TAUW1, TAUW2

! For Symmetry reduction
      INTEGER IMOD,KMOD
      COMPLEX*16 CU1S_TEMP(0:NXP,0:NZ+1,0:NY+1),
     &       CU2S_TEMP(0:NXP,0:NZ+1,0:NY+1),
     &       CU3S_TEMP(0:NXP,0:NZ+1,0:NY+1),
     &       CPS_TEMP(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 KAPPAX,LSX
      CHARACTER*55 FNAME

C Communicate the information between ghost cells
!      IF (USE_MPI) THEN
!        CALL GHOST_CHAN_MPI
!      END IF

C Define the constants that are used in the time-stepping
C For reference, see Numerical Renaissance
      TEMP1=NU * H_BAR(RK_STEP) / 2.0
      TEMP2=H_BAR(RK_STEP) / 2.0
      TEMP3=ZETA_BAR(RK_STEP) * H_BAR(RK_STEP)
      TEMP4=H_BAR(RK_STEP)
      TEMP5=BETA_BAR(RK_STEP) * H_BAR(RK_STEP)

C First, we will compute the explicit RHS terms and store in Ri
C Note, Momentum equation and hence the RHS is evaluated at the
C corresponding velocity points.

#ifdef NEWTON
!===   Shift and reflect symmetry in Physical space =================
! Note that spanwise and streamwise grind points should be even 
!
      CU1S_TEMP=CU1
      CU2S_TEMP=CU2
      CU3S_TEMP=CU3
      CPS_TEMP=CP

      LSX=LX/2.0D0
      DO J=0,NY+1
        Do K=0,TNKZ
          KMOD=MOD(TNKZ+1-K,TNKZ+1)
          DO I=0,NXP-1
            CU1(I,K,J)=(CU1S_TEMP(I,KMOD,J)*EXP(-CIKX(I)*LSX)+
     &                  CU1(I,K,J))*0.5D0
            CU2(I,K,J)=(CU2S_TEMP(I,KMOD,J)*EXP(-CIKX(I)*LSX)+
     &                  CU2(I,K,J))*0.5D0
            CU3(I,K,J)=(-CU3S_TEMP(I,KMOD,J)*EXP(-CIKX(I)*LSX)+
     &                  CU3(I,K,J))*0.5D0
            CP(I,K,J)=(CPS_TEMP(I,KMOD,J)*EXP(-CIKX(I)*LSX)+
     &                  CP(I,K,J))*0.5D0
          END DO
        END DO
      END DO

!      FNAME='test.h5'
!      call WriteHDF5(FNAME,.TRUE.)
!      stop

!===   Shift and reflect symmetry in Physical space =================
#endif

C Store the old velocity in the RHS vector
      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
            CR1(I,K,J)=CU1(I,K,J)
            CR3(I,K,J)=CU3(I,K,J)
          END DO
        END DO
      END DO
      DO J=2,NY 
        DO K=0,TNKZ
          DO I=0,NXP-1
            CR2(I,K,J)=CU2(I,K,J)
          END DO
        END DO
      END DO

C Add the R-K term from the rk-1 step 
      IF (RK_STEP .GT. 1) THEN
        DO J=JSTART,JEND
          DO K=0,TNKZ
            DO I=0,NXP-1
              CR1(I,K,J)=CR1(I,K,J)+TEMP3*CF1(I,K,J)
              CR3(I,K,J)=CR3(I,K,J)+TEMP3*CF3(I,K,J)
            END DO
          END DO
        END DO
        DO J=2,NY
          DO K=0,TNKZ
            DO I=0,NXP-1
              CR2(I,K,J)=CR2(I,K,J)+TEMP3*CF2(I,K,J)
            END DO
          END DO
        END DO
      END IF

          
C Take the y-derivative of the pressure at GY points in Fourier space
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CS1(I,K,J)=(CP(I,K,J) - CP(I,K,J-1)) / DY(J)
          END DO
        END DO
      END DO

C Add the pressure gradient to the RHS as explicit Euler
      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
            CR1(I,K,J)=CR1(I,K,J)-TEMP4*(CIKX(I)*CP(I,K,J))
            CR3(I,K,J)=CR3(I,K,J)-TEMP4*(CIKZ(K)*CP(I,K,J))
          END DO
        END DO
      END DO
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CR2(I,K,J)=CR2(I,K,J)-TEMP4*CS1(I,K,J)
          END DO
        END DO
      END DO


C Here, add the constant, forcing pressure gradient
C There are several ways of doing this
      IF (F_TYPE.EQ.1) THEN 
C Add forcing for a constant pressure gradient
        DO J=JSTART,JEND
          IF (RANKZ.eq.0) CR1(0,0,J)=CR1(0,0,J)-TEMP4*PX0
        END DO
      ELSE IF (F_TYPE.EQ.0) THEN  
C Add the mean pressure gradient to keep Ubulk constant
C This section needs to be parallelized 
        IF (RANKZ.EQ.0) THEN
          UME=dble(CU1(0,0,:))
        ELSE
          UME=0.0D0
        END IF
        CALL INTEGRATE_Y_VAR(UME,UBULK,MPI_COMM_WORLD)
        TAUW1=0.0D0; TAUW2=0.0D0
        IF(RANKZ.EQ.0) THEN
          TAUW1=(UME(2)-UME(1))/(GYF(2)-GYF(1))
          TAUW2=(UME(NY)-UME(NY-1))/(GYF(NY)-GYF(NY-1))
        END IF
        CALL MPI_BCAST(TAUW1,1,MPI_DOUBLE_PRECISION,0,
     &     MPI_COMM_WORLD,ierror)
        CALL MPI_BCAST(TAUW2,1,MPI_DOUBLE_PRECISION,NPROCY-1,
     &     MPI_COMM_WORLD,ierror)
!        WRITE(*,*) 'UBULK=',UBULK,RANKY,RANKZ,TAUW1,TAUW2
        PX0=-1.*NU*(TAUW1-TAUW2)/Ly
        K_P=1.5D0
        PX0=PX0+K_P*(UBULK-UBULK0)
        IF(RANKZ.EQ.0) THEN
          DO J=JSTART,JEND
            CR1(0,0,J)=CR1(0,0,J)-TEMP4*PX0
          END DO
        END IF
!
      ELSE IF (F_TYPE.EQ.2) THEN
C If oscillatory pressure gradient
        DO J=JSTART,JEND
           IF (RANKZ.eq.0) CR1(0,0,J)=CR1(0,0,J)-
     &          TEMP4*(PX0+AMP_OMEGA0*cos(OMEGA0*TIME))
        END DO

C End if forcing type
      END IF

C Now compute the term R-K term Ai
C Compile terms of Ai in CFi which will be saved for next time step
C First, store the horizontal viscous terms in CFi
      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
            CF1(I,K,J)=-NU * KX2(I) * CU1(I,K,J) 
     &            - NU * KZ2(K) * CU1(I,K,J)
            CF3(I,K,J)=-NU * KX2(I) * CU3(I,K,J) 
     &            - NU * KZ2(K) * CU3(I,K,J)
          END DO
        END DO
      END DO
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CF2(I,K,J)=-NU * KX2(I) * CU2(I,K,J) 
     &            - NU * KZ2(K) * CU2(I,K,J)
          END DO 
        END DO
      END DO

! Add the terms owing to the system rotation (Coriolis terms) 
! Assume that the flow is on an f-plane
      DO K=0,TNKZ
        DO I=0,NXP-1
          DO J=JSTART,JEND
            CF1(I,K,J)=CF1(I,K,J)+I_RO_TAU*CU3(I,K,J)
            CF3(I,K,J)=CF3(I,K,J)-I_RO_TAU*CU1(I,K,J)
          END DO
        END DO
      END DO

! Do for each scalar
      DO N=1,N_TH
! If a scalar contributes to the denisty, RI_TAU is not equal to zero and
! add the buoyancy term as explicit R-K.  Don't add the 0,0 mode which 
! corresponds to a plane average.  The plane averaged density balances
! the hydrostratic pressure component.
      DO J=2,NY
        DO K=1,TNKZ
          DO I=0,NXP-1
! Use second order interpolation
             CF2(I,K,J)=CF2(I,K,J)+RI_TAU(N)*
     &      (CTH(I,K,J,N)*DYF(J-1)+CTH(I,K,J-1,N)*DYF(J))/(2.d0*DY(J))
          END DO
        END DO
        K=0
        IF (RANKZ.eq.0) THEN
           ISTART=1
        ELSE
           ISTART=0
        END IF
        DO I=ISTART,NXP-1
             CF2(I,K,J)=CF2(I,K,J)+RI_TAU(N)*
     &      (CTH(I,K,J,N)*DYF(J-1)+CTH(I,K,J-1,N)*DYF(J))/(2.d0*DY(J))
        END DO
      END DO

! Now, compute the RHS vector for the scalar equations
! Since TH is defined at horizontal velocity points, the
! scalar update equation will be very similar to the horizontal
! velocity update.

! We will store the RHS scalar terms in CRTH, RTH
! The k-1 term for the R-K stepping is saved in FTH, CFTH

! First, build the RHS vector, use CRTH
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,TNKZ
          DO I=0,NXP-1
          CRTH(I,K,J,N)=CTH(I,K,J,N)
         ENDDO
       END DO
      END DO
! Add term from k-2 step to free up CFTH variable
      IF (RK_STEP .GT. 1) THEN
        DO J=JSTART_TH(N),JEND_TH(N)
          DO K=0,TNKZ
            DO I=0,NXP-1
              CRTH(I,K,J,N)=CRTH(I,K,J,N)+TEMP3*CFTH(I,K,J,N)
            END DO
          END DO
        END DO
       END IF

! Now compute the explicit R-K term Ai
! Compile terms of Ai in CFi which will be saved for next time step
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,TNKZ
          DO I=0,NXP-1
            CFTH(I,K,J,N)=-(NU/PR(N)) * KX2(I) * CTH(I,K,J,N)
     &            - (NU/PR(N)) * KZ2(K) * CTH(I,K,J,N)
          END DO
        END DO
      END DO

C End do number of passive scalars (N_TH)
      END DO

C If we are considering an LES, then add the subgrid scale stress:
C Here, velocity and CFi should be in Fourier space
C The subgrid scale stress is added to CFi:   CFi=CFi - d/dx_i tau_ij

      IF (LES.AND.((.NOT.CREATE_NEW_FLOW).OR.(TIME_STEP.GT.100))) THEN
!         IF (RANK.EQ.0) 
!     &        write(*,*) ' LES not implemented! '
!         stop 
C If we have created new flow with random perturbations, wait for a
C spinup before applying the subgrid model for stability purposes
C In the process, Ui is converted to physical space
          call les_chan
C Add the subgrid scale scalar flux to the scalar equations
          DO N=1,N_TH
!            call les_chan_th(N)
          END DO
      ELSE 
C If the subgrid model has not been called, then it is necessary to 
C convert to physical space.
        CALL FFT_XZ_TO_PHYSICAL(CU1,U1,0,NY+1)
        CALL FFT_XZ_TO_PHYSICAL(CU2,U2,0,NY+1)
        CALL FFT_XZ_TO_PHYSICAL(CU3,U3,0,NY+1)
! Transform THETA to physical space for computation of nonlinear terms
! Here pass the first location in memory of the array for scalar n

        DO N=1,N_TH
          CALL FFT_XZ_TO_PHYSICAL(CTH(0,0,0,N),TH(0,0,0,N),0,NY+1)
        END DO      


      END IF

        CALL FFT_XZ_TO_FOURIER(U1,CU1,0,NY+1)
        CALL FFT_XZ_TO_FOURIER(U2,CU2,0,NY+1)
        CALL FFT_XZ_TO_FOURIER(U3,CU3,0,NY+1)

C GQL in x,z-directions, NN <= NXP-1, MM <= NKZ
C EXAMPLE: QL:  NN=0, MM=NKZ
C EXAMPLE: GQL in x-direction:  NN<=NXP-1, MM=NKZ
C EXAMPLE: GQL in z-direction:  NN=NXP-1, MM<=NKZ
C EXAMPLE: GQL in x,z-directions:  NN<=NXP-1, MM<=NKZ

      NN=NKX
      MM=0

      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
           II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CU1P(I,K,J)=CMPLX(0.D0,0.D0)
            CU2P(I,K,J)=CMPLX(0.D0,0.D0)
            CU3P(I,K,J)=CMPLX(0.D0,0.D0)
            CU1M(I,K,J)=CU1(I,K,J)
            CU2M(I,K,J)=CU2(I,K,J)
            CU3M(I,K,J)=CU3(I,K,J)
           ELSE
            CU1M(I,K,J)=CMPLX(0.D0,0.D0)
            CU2M(I,K,J)=CMPLX(0.D0,0.D0)
            CU3M(I,K,J)=CMPLX(0.D0,0.D0)
            CU1P(I,K,J)=CU1(I,K,J)
            CU2P(I,K,J)=CU2(I,K,J)
            CU3P(I,K,J)=CU3(I,K,J)
           END IF	 
          END DO 
        END DO
      END DO

      CALL HALO_ARRAY_COMPLEX(CU1M)
      CALL HALO_ARRAY_COMPLEX(CU2M)
      CALL HALO_ARRAY_COMPLEX(CU3M)
      CALL HALO_ARRAY_COMPLEX(CU1P)
      CALL HALO_ARRAY_COMPLEX(CU2P)
      CALL HALO_ARRAY_COMPLEX(CU3P)

      CALL FFT_XZ_TO_PHYSICAL(CU1,U1,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CU2,U2,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CU3,U3,0,NY+1)
      
      CALL FFT_XZ_TO_PHYSICAL(CU1M,U1M,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CU2M,U2M,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CU3M,U3M,0,NY+1)

      CALL FFT_XZ_TO_PHYSICAL(CU1P,U1P,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CU2P,U2P,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CU3P,U3P,0,NY+1)

C U1*U3
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=U3(I,K,J)*U1(I,K,J)
            Q1(I,K,J)=U3M(I,K,J)*U1M(I,K,J)
     &      +U3P(I,K,J)*U1P(I,K,J)
            Q2(I,K,J)=S1(I,K,J)-U3M(I,K,J)*U1M(I,K,J)
     &      -U3P(I,K,J)*U1P(I,K,J)
          END DO
        END DO
      END DO

      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF1(I,K,J)=CF1(I,K,J)-CIKZ(K)*CQ1(I,K,J)
            CF3(I,K,J)=CF3(I,K,J)-CIKX(I)*CQ1(I,K,J)
          ELSE
            CF1(I,K,J)=CF1(I,K,J)-CIKZ(K)*CQ2(I,K,J)
            CF3(I,K,J)=CF3(I,K,J)-CIKX(I)*CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO   

C U1*U1
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=U1(I,K,J)*U1(I,K,J)
            Q1(I,K,J)=U1M(I,K,J)*U1M(I,K,J)
     &      +U1P(I,K,J)*U1P(I,K,J)
            Q2(I,K,J)=S1(I,K,J)-U1M(I,K,J)*U1M(I,K,J)
     &      -U1P(I,K,J)*U1P(I,K,J)
          END DO
        END DO
      END DO

      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)
      
      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)
  
      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF1(I,K,J)=CF1(I,K,J)-CIKX(I)*CQ1(I,K,J)
          ELSE
            CF1(I,K,J)=CF1(I,K,J)-CIKX(I)*CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

C U3*U3
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=U3(I,K,J)*U3(I,K,J)
            Q1(I,K,J)=U3M(I,K,J)*U3M(I,K,J)
     &      +U3P(I,K,J)*U3P(I,K,J)
            Q2(I,K,J)=S1(I,K,J)-U3M(I,K,J)*U3M(I,K,J)
     &      -U3P(I,K,J)*U3P(I,K,J)
          END DO
        END DO
      END DO
      
      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF3(I,K,J)=CF3(I,K,J)-CIKZ(K)*CQ1(I,K,J)
          ELSE
            CF3(I,K,J)=CF3(I,K,J)-CIKZ(K)*CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

C U1*U2
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=((DYF(J)*U1(I,K,J)
     &                +DYF(J-1)*U1(I,K,J-1))/(2.*DY(J))) 
     &                *U2(I,K,J)
            Q1(I,K,J)=((DYF(J)*U1M(I,K,J)
     &                +DYF(J-1)*U1M(I,K,J-1))/(2.*DY(J))) 
     &                *U2M(I,K,J)+((DYF(J)*U1P(I,K,J)
     &                +DYF(J-1)*U1P(I,K,J-1))/(2.*DY(J))) 
     &                *U2P(I,K,J)
            Q2(I,K,J)=S1(I,K,J)-((DYF(J)*U1M(I,K,J)
     &                +DYF(J-1)*U1M(I,K,J-1))/(2.*DY(J))) 
     &                *U2M(I,K,J)-((DYF(J)*U1P(I,K,J)
     &                +DYF(J-1)*U1P(I,K,J-1))/(2.*DY(J))) 
     &                *U2P(I,K,J)
          END DO
        END DO
      END DO
      
      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF2(I,K,J)=CF2(I,K,J)-CIKX(I)*CQ1(I,K,J)
          ELSE
            CF2(I,K,J)=CF2(I,K,J)-CIKX(I)*CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

C U3*U2
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=((DYF(J)*U3(I,K,J)
     &                +DYF(J-1)*U3(I,K,J-1))/(2.*DY(J))) 
     &                *U2(I,K,J)
            Q1(I,K,J)=((DYF(J)*U3M(I,K,J)
     &                +DYF(J-1)*U3M(I,K,J-1))/(2.*DY(J))) 
     &                *U2M(I,K,J)+((DYF(J)*U3P(I,K,J)
     &                +DYF(J-1)*U3P(I,K,J-1))/(2.*DY(J))) 
     &                *U2P(I,K,J)
            Q2(I,K,J)=S1(I,K,J)-((DYF(J)*U3M(I,K,J)
     &                +DYF(J-1)*U3M(I,K,J-1))/(2.*DY(J))) 
     &                *U2M(I,K,J)-((DYF(J)*U3P(I,K,J)
     &                +DYF(J-1)*U3P(I,K,J-1))/(2.*DY(J))) 
     &                *U2P(I,K,J)
          END DO
        END DO
      END DO
      
      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=2,NY
        DO K=0,TNKZ   	    
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF2(I,K,J)=CF2(I,K,J)-CIKZ(K)*CQ1(I,K,J)
          ELSE
            CF2(I,K,J)=CF2(I,K,J)-CIKZ(K)*CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

! Add the vertical derivative term explicitly
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=
     &       (U1(I,K,J+1)*U2(I,K,J+1) + U1(I,K,J)*U2(I,K,J+1)
     &      - U1(I,K,J)*U2(I,K,J) - U1(I,K,J-1)*U2(I,K,J))/(2.d0*DYF(J))
            Q1(I,K,J)=(
     &       (U1M(I,K,J+1)*U2M(I,K,J+1) + U1M(I,K,J)*U2M(I,K,J+1)
     &  - U1M(I,K,J)*U2M(I,K,J) - U1M(I,K,J-1)*U2M(I,K,J))/(2.d0*DYF(J))
     &       +(U1P(I,K,J+1)*U2P(I,K,J+1) + U1P(I,K,J)*U2P(I,K,J+1)
     &  - U1P(I,K,J)*U2P(I,K,J) - U1P(I,K,J-1)*U2P(I,K,J))
     &      /(2.d0*DYF(J)))     
            Q2(I,K,J)=S1(I,K,J)-(
     &       (U1M(I,K,J+1)*U2M(I,K,J+1) + U1M(I,K,J)*U2M(I,K,J+1)
     &  - U1M(I,K,J)*U2M(I,K,J) - U1M(I,K,J-1)*U2M(I,K,J))/(2.d0*DYF(J))
     &       +(U1P(I,K,J+1)*U2P(I,K,J+1) + U1P(I,K,J)*U2P(I,K,J+1)
     &  - U1P(I,K,J)*U2P(I,K,J) - U1P(I,K,J-1)*U2P(I,K,J))
     &      /(2.d0*DYF(J)))     
          END DO
        END DO
      END DO

      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF1(I,K,J)=CF1(I,K,J)-CQ1(I,K,J)
          ELSE
            CF1(I,K,J)=CF1(I,K,J)-CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

! Add the vertical derivative term explicitly
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=
     &     (U3(I,K,J+1)*U2(I,K,J+1) + U3(I,K,J)*U2(I,K,J+1)
     &     - U3(I,K,J)*U2(I,K,J) - U3(I,K,J-1)*U2(I,K,J))/(2.d0*DYF(J))
           Q1(I,K,J)=((U3M(I,K,J+1)*U2M(I,K,J+1) 
     &      + U3M(I,K,J)*U2M(I,K,J+1)
     & - U3M(I,K,J)*U2M(I,K,J) - U3M(I,K,J-1)*U2M(I,K,J))/(2.d0*DYF(J)) 
     &     +(U3P(I,K,J+1)*U2P(I,K,J+1) + U3P(I,K,J)*U2P(I,K,J+1)
     &  - U3P(I,K,J)*U2P(I,K,J) - U3P(I,K,J-1)*U2P(I,K,J))
     &      /(2.d0*DYF(J)))  
           Q2(I,K,J)=S1(I,K,J)-((U3M(I,K,J+1)*U2M(I,K,J+1) 
     &      + U3M(I,K,J)*U2M(I,K,J+1)
     & - U3M(I,K,J)*U2M(I,K,J) - U3M(I,K,J-1)*U2M(I,K,J))/(2.d0*DYF(J)) 
     &     +(U3P(I,K,J+1)*U2P(I,K,J+1) + U3P(I,K,J)*U2P(I,K,J+1)
     &  - U3P(I,K,J)*U2P(I,K,J) - U3P(I,K,J-1)*U2P(I,K,J))
     &      /(2.d0*DYF(J)))  
          END DO
        END DO
      END DO

      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF3(I,K,J)=CF3(I,K,J)-CQ1(I,K,J)
          ELSE
            CF3(I,K,J)=CF3(I,K,J)-CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

! Add the vertical derivative term explicitly
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=
     &     (0.25d0*(U2(I,K,J)+U2(I,K,J+1))**2.d0
     &     -0.25d0*(U2(I,K,J)+U2(I,K,J-1))**2.d0)/DY(J)
            Q1(I,K,J)=(
     &     (0.25d0*(U2M(I,K,J)+U2M(I,K,J+1))**2.d0
     &     -0.25d0*(U2M(I,K,J)+U2M(I,K,J-1))**2.d0)/DY(J)
     &    +(0.25d0*(U2P(I,K,J)+U2P(I,K,J+1))**2.d0
     &     -0.25d0*(U2P(I,K,J)+U2P(I,K,J-1))**2.d0)/DY(J))
            Q2(I,K,J)=S1(I,K,J)-(
     &     (0.25d0*(U2M(I,K,J)+U2M(I,K,J+1))**2.d0
     &     -0.25d0*(U2M(I,K,J)+U2M(I,K,J-1))**2.d0)/DY(J)
     &    +(0.25d0*(U2P(I,K,J)+U2P(I,K,J+1))**2.d0
     &     -0.25d0*(U2P(I,K,J)+U2P(I,K,J-1))**2.d0)/DY(J))
          END DO
        END DO
      END DO

      CALL HALO_ARRAY_REAL(S1)
      CALL HALO_ARRAY_REAL(Q1)
      CALL HALO_ARRAY_REAL(Q2)

      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q1,CQ1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(Q2,CQ2,0,NY+1)

      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
          II = NXP*RANKZ+I
           IF(II.LE.NN.AND.((K.LE.MM.OR.K.GE.(TNKZ-MM+1)))) THEN
            CF2(I,K,J)=CF2(I,K,J)-CQ1(I,K,J)
          ELSE
            CF2(I,K,J)=CF2(I,K,J)-CQ2(I,K,J)
          END IF
          END DO
        END DO
      END DO

C -- At this point, we are done computing the nonlinear terms --

C -- At this point, we are done computing the nonlinear terms --

C Finally, Add CFi to CRi
      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
            CR1(I,K,J)=CR1(I,K,J) + TEMP5 * CF1(I,K,J)
            CR3(I,K,J)=CR3(I,K,J) + TEMP5 * CF3(I,K,J)
          END DO
        END DO
      END DO
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CR2(I,K,J)=CR2(I,K,J) + TEMP5 * CF2(I,K,J)
          END DO
        END DO
      END DO

C Convert RHS terms to physical space
      CALL FFT_XZ_TO_PHYSICAL(CR1,R1,0,NY+1)                 
      CALL FFT_XZ_TO_PHYSICAL(CR2,R2,2,NY)                 
      CALL FFT_XZ_TO_PHYSICAL(CR3,R3,0,NY+1)                 

C Compute the vertical viscous term in physical space and add to RHS
C This is the explicit part of the Crank-Nicolson term
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            R1(I,K,J)=R1(I,K,J)+TEMP1*
     &        (  ((U1(I,K,J+1) - U1(I,K,J)) / DY(J+1)  
     &           -(U1(I,K,J)   - U1(I,K,J-1)) / DY(J)) /DYF(J)  )
            R3(I,K,J)=R3(I,K,J)+TEMP1*
     &        (  ((U3(I,K,J+1) - U3(I,K,J)) / DY(J+1) 
     &           -(U3(I,K,J)   - U3(I,K,J-1)) / DY(J)) /DYF(J)  )
          END DO
        END DO
      END DO
      DO J=2,NY 
        DO K=0,NZP-1
          DO I=0,NXM
            R2(I,K,J)=R2(I,K,J)+TEMP1*
     &        (  ((U2(I,K,J+1) - U2(I,K,J))  / DYF(J) 
     &           -(U2(I,K,J)   - U2(I,K,J-1))/ DYF(J-1))/DY(J)  )
          END DO
        END DO
      END DO

C If we are using a subgrid model, add the eddy viscosity term
C This is an added viscosity that will be treated just like the 
C molecular viscosity with Crank-Nicolson for the vertical derivatives
      IF (LES) then
C Note, NU_T is defined at GY points
!      IF (RANK.EQ.0) 
!     &        write(*,*) ' LES not implemented! '
!      stop
      DO J=JSTART,JEND
        DO K=0,NZP-1
          DO I=0,NXM
            R1(I,K,J)=R1(I,K,J)+TEMP2*
     &        (  (NU_T(I,K,J+1) * (U1(I,K,J+1) - U1(I,K,J)) / DY(J+1)  
     &         -  NU_T(I,K,J) * (U1(I,K,J)   - U1(I,K,J-1)) / DY(J))
     &               /DYF(J)  )
            R3(I,K,J)=R3(I,K,J)+TEMP2*
     &        (  (NU_T(I,K,J+1) * (U3(I,K,J+1) - U3(I,K,J)) / DY(J+1) 
     &        - NU_T(I,K,J) * (U3(I,K,J)   - U3(I,K,J-1)) / DY(J)) 
     &              /DYF(J)  )
          END DO
        END DO
      END DO
! Here, interpolate NU_T to GYF points
      DO J=2,NY 
        DO K=0,NZP-1
          DO I=0,NXM
            R2(I,K,J)=R2(I,K,J)+TEMP2*
     &     ((0.5d0*(NU_T(I,K,J)+NU_T(I,K,J+1))*(U2(I,K,J+1)-U2(I,K,J))
     &                                              / DYF(J) 
     &    -0.5d0*(NU_T(I,K,J)+NU_T(I,K,J-1))*(U2(I,K,J)-U2(I,K,J-1))
     &                                          / DYF(J-1))   /DY(J)  )
          END DO
        END DO
      END DO
      END IF

C -- Here, we are done with computation of Velocity RHS, explicit terms --

C -- Now, remove the spanwise uniform motion as in Hwang JFM 2013, only for u and v (by Oliver)
c      CALL FFT_XZ_TO_FOURIER(R1,CR1,0,NY+1)
c      CALL FFT_XZ_TO_FOURIER(R2,CR2,2,NY)
c      DO J=2,NY
c        DO K=0,TNKZ; DO I=0,NXP-1
c          IF(RANKZ.EQ.0.AND.K.EQ.0.AND.I.NE.0) THEN
c            CR1(I,K,J)=0.0D0
c           CR2(I,K,J)=0.0D0
c          END IF
c        END DO; END DO
c      END DO
C Convert RHS terms to physical space
c      CALL FFT_XZ_TO_PHYSICAL(CR1,R1,0,NY+1)
c      CALL FFT_XZ_TO_PHYSICAL(CR2,R2,2,NY)
C -- Here, we are done with removing spanwise uniform motions

C Now, build the explicit RHS terms for the passive scalar(s)

      DO N=1,N_TH
! Do for each scalar:

! Compute the nonlinear terms that are present in the explicit term A
! U1*TH
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=TH(I,K,J,N)*U1(I,K,J)
          END DO
        END DO
      END DO
      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,TNKZ
          DO I=0,NXP-1
            CFTH(I,K,J,N)=CFTH(I,K,J,N) - CIKX(I) * CS1(I,K,J)
          END DO
        END DO
      END DO
! U3*TH 
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=TH(I,K,J,N)*U3(I,K,J)
          END DO
        END DO
      END DO
      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,TNKZ
          DO I=0,NXP-1
            CFTH(I,K,J,N)=CFTH(I,K,J,N) - CIKZ(K) * CS1(I,K,J)
          END DO
        END DO
      END DO

! We are done with the horizontal derivatives of the nonlinear terms
! Add the vertical derivative term explicitly
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,NZP-1
          DO I=0,NXM
            S1(I,K,J)=
     &     (TH(I,K,J+1,N)*U2(I,K,J+1) + TH(I,K,J,N)*U2(I,K,J+1)
     &    -TH(I,K,J,N)*U2(I,K,J)-TH(I,K,J-1,N)*U2(I,K,J))/(2.d0*DYF(J))
          END DO
        END DO
      END DO
      CALL FFT_XZ_TO_FOURIER(S1,CS1,0,NY+1)
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,TNKZ
          DO I=0,NXP-1
            CFTH(I,K,J,N)=CFTH(I,K,J,N) - CS1(I,K,J)
          END DO
        END DO
      END DO


! Add CFTH to the RHS vector CRTH
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,TNKZ
          DO I=0,NXP-1
            CRTH(I,K,J,N)=CRTH(I,K,J,N) + TEMP5 * CFTH(I,K,J,N)
          END DO
        END DO
      END DO
! Done with computation of RHS, explicit terms for the THETA equation
! Transform back to physical space

      CALL FFT_XZ_TO_PHYSICAL(CRTH(0,0,0,N),RTH(0,0,0,N),0,NY+1)    

! Compute the Explicit part of the Crank-Nicolson terms for the TH equation
! First, the vertical derivative viscous term
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,NZP-1
          DO I=0,NXM
            RTH(I,K,J,N)=RTH(I,K,J,N)+(TEMP1/PR(N))*(
     &            ((TH(I,K,J+1,N) - TH(I,K,J,N)) / DY(J+1)
     &            -(TH(I,K,J,N) - TH(I,K,J-1,N)) / DY(J)) / DYF(J) )
          END DO
        END DO
      END DO
! If we are using a subgrid model (LES) then add the eddy diffusivity here
! Note, KAPPA_T is defined at GY points
      IF (LES) THEN
         IF (RANK.EQ.0) 
     &        write(*,*) ' LES not implemented! '
         stop
      DO J=JSTART_TH(N),JEND_TH(N)
        DO K=0,NZP-1
          DO I=0,NXM
            RTH(I,K,J,N)=RTH(I,K,J,N)+TEMP2*(
     &     (KAPPA_T(I,K,J+1,N)*(TH(I,K,J+1,N)-TH(I,K,J,N))/DY(J+1)
     &     -KAPPA_T(I,K,J,N)*(TH(I,K,J,N)-TH(I,K,J-1,N))/DY(J))/DYF(J))
          END DO
        END DO
      END DO  
      END IF

C -- Now, timestep the passive scalar equation --
C      We solve the the passive scalar before the velocity so that
C      it is advected with the velocity from the previous R-K step
C      which we have already made divergence free 
 
! Solve the implicit equation for THETA
! Note that the system size is NY+1, but only 1..NY are used

! Initialize the matrix used to store implicit coefficients
      DO J=0,NY+1
        DO I=0,NXM
          MATL(I,J)=0.
          MATD(I,J)=1.
          MATU(I,J)=0.
          VEC(I,J)=0.
        END DO
      END DO 
    
! Build implicit matrix
! Use quasi-second order interpolation for TH on GY points
      DO K=0,NZP-1
        DO J=JSTART_TH(N),JEND_TH(N)
          DO I=0,NXM
            MATL(I,J) = -(TEMP1/PR(N)) / (DY(J)*DYF(J))
            MATD(I,J) = 1. + (TEMP1/PR(N)) / (DY(J+1)*DYF(J))
     &           +(TEMP1/PR(N)) / (DY(J)*DYF(J))
            MATU(I,J)=-(TEMP1/PR(N)) / (DY(J+1)*DYF(J))
! Define RHS vector
            VEC(I,J)=RTH(I,K,J,N)
          END DO
        END DO
! IF using a subgrid model (LES) then add the eddy diffusivity part implicitly
        IF (LES) THEN
        DO J=JSTART_TH(N),JEND_TH(N)
          DO I=0,NXM   
            MATL(I,J) = MATL(I,J) - TEMP2 * KAPPA_T(I,K,J,N) 
     &                                 / (DY(J)*DYF(J))
            MATD(I,J) = MATD(I,J)+ TEMP2 * KAPPA_T(I,K,J+1,N)
     &                                 / (DY(J+1)*DYF(J))
     &                           + TEMP2 * KAPPA_T(I,K,J,N)
     &                                 / (DY(J)*DYF(J))
            MATU(I,J) = MATU(I,J)- TEMP2 * KAPPA_T(I,K,J+1,N)
     &                                / (DY(J+1)*DYF(J))
          END DO
        END DO
        END IF

! If we are using MPI, then solve the implicit system in separate forward
! and backward sweeps for efficiency
          IF (USE_MPI) THEN

             CALL APPLY_BC_TH_MPI(MATL,MATD,MATU,VEC,N)
! If we are using MPI, split the implicit solve into foward and
! backward sweeps for efficiency
             CALL THOMAS_FORWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
             CALL THOMAS_BACKWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
          ELSE
! Else we are running in serial mode
             CALL APPLY_BC_TH_LOWER(MATL,MATD,MATU,VEC,N)
             CALL APPLY_BC_TH_UPPER(MATL,MATD,MATU,VEC,N)
             CALL THOMAS_REAL(MATL,MATD,MATU,VEC,NY+1,NXM)
! Apply the boundary conditions to our linear system
          END IF

        DO J=JSTART_TH(N),JEND_TH(N)
          DO I=0,NXM
            TH(I,K,J,N)=VEC(I,J)
          END DO
        END DO

! END do k
      END DO 

! End do number of passive scalars
        END DO
        
C Initialize the matrix to zeros to be used for implicit solves
C Note that the system size is NY+1, but only 1..NY are used

! Initialize the matrix used to store implicit coefficients
      DO J=0,NY+1
        DO I=0,NXM
          MATL(I,J)=0.
          MATD(I,J)=1.
          MATU(I,J)=0.
          VEC(I,J)=0.
        END DO
      END DO 

C Build implicit matrix for U2
      DO K=0,NZP-1
        DO J=2,NY
          DO I=0,NXM
            MATL(I,J)= -TEMP1/(DYF(J-1)*DY(J))
            MATD(I,J)=1.+TEMP1/(DYF(J)*DY(J)) + TEMP1/(DYF(J-1)*DY(J)) 
            MATU(I,J)= -TEMP1/(DYF(J)*DY(J))
            VEC(I,J)=R2(I,K,J)
          END DO 
        END DO
        IF (LES) THEN
! IF using a subgrid model (LES) then add the eddy viscosity part implicitly
        DO J=2,NY
          DO I=0,NXM
            MATL(I,J) = MATL(I,J) 
     &      - TEMP2 * 0.5d0*(NU_T(I,K,J)+NU_T(I,K,J-1))/(DYF(J-1)*DY(J))
            MATD(I,J) = MATD(I,J) 
     &      + TEMP2 * 0.5d0*(NU_T(I,K,J)+NU_T(I,K,J+1))/(DYF(J)*DY(J))
     &      + TEMP2 * 0.5d0*(NU_T(I,K,J)+NU_T(I,K,J-1))/(DYF(J-1)*DY(J))
            MATU(I,J) = MATU(I,J) 
     &      - TEMP2 * 0.5d0*(NU_T(I,K,J)+NU_T(I,K,J+1))/(DYF(J)*DY(J))
          END DO
        END DO
        END IF

        IF (USE_MPI) THEN

! First, apply the boundary conditions
          CALL APPLY_BC_U2_MPI(MATL,MATD,MATU,VEC)
! If we are using MPI, split the implicit solve into forward and
! backward sweeps for efficiency
          CALL THOMAS_FORWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
          CALL THOMAS_BACKWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
        ELSE
C Else, we are running in serial mode
C Set the boundary conditions for U2
          CALL APPLY_BC_2_LOWER(MATL,MATD,MATU,VEC)
          CALL APPLY_BC_2_UPPER(MATL,MATD,MATU,VEC)

C Now, solve the tridiagonal system for U2(i,:,k)
          CALL THOMAS_REAL(MATL,MATD,MATU,VEC,NY+1,NXM)
        END IF

        DO J=1,NY+1
          DO I=0,NXM
            U2(I,K,J)=VEC(I,J)
          END DO
        END DO
! End do k
      END DO 

C Solve for U1
C Note, here the matrix will be indexed from 1...NY+1 corresponding to U1(0:NY)

! Initialize the matrix used to store implicit coefficients
      DO J=0,NY+1
        DO I=0,NXM
          MATL(I,J)=0.
          MATD(I,J)=1.
          MATU(I,J)=0.
          VEC(I,J)=0.
        END DO
      END DO 

C Build the implicit system of equations for U1 
      DO K=0,NZP-1
        DO J=JSTART,JEND
          DO I=0,NXM
            MATL(I,J)=-TEMP1/(DY(J)*DYF(J))
            MATD(I,J)=1.-TEMP1*(-1./(DY(J+1)*DYF(J))
     &         -1./(DY(J)*DYF(J))) 
            MATU(I,J)=-TEMP1/(DY(J+1)*DYF(J))
            VEC(I,J)=R1(I,K,J)
          END DO
        END DO
! IF using a subgrid model (LES) then add the eddy viscosity part implicitly
        IF (LES) THEN
        DO J=JSTART,JEND
          DO I=0,NXM
            MATL(I,J) = MATL(I,J) - TEMP2 * NU_T(I,K,J) 
     &                               / (DY(J)*DYF(J))
            MATD(I,J) = MATD(I,J) + TEMP2 * NU_T(I,K,J+1)
     &                              / (DY(J+1)*DYF(J))
     &                            + TEMP2 * NU_T(I,K,J)
     &                              / (DY(J)*DYF(J))
            MATU(I,J) = MATU(I,J) - TEMP2 * NU_T(I,K,J+1)
     &                             / (DY(J+1)*DYF(J))
          END DO
        END DO
        END IF

        IF (USE_MPI) THEN
! First, apply the boundary conditions
          CALL APPLY_BC_U1_MPI(MATL,MATD,MATU,VEC)
! If we are using MPI, split the implicit solve into forward and
! backward sweeps for efficiency
          CALL THOMAS_FORWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
          CALL THOMAS_BACKWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
        ELSE
C Else, we are running in serial mode
C Set the boundary conditions for U1
          CALL APPLY_BC_1_LOWER(MATL,MATD,MATU,VEC)
          CALL APPLY_BC_1_UPPER(MATL,MATD,MATU,VEC)
C Now, solve the tridiagonal system for U1(:,k,:)
          CALL THOMAS_REAL(MATL,MATD,MATU,VEC,NY+1,NXM)
        END IF

        DO J=JSTART-1,JEND+1
          DO I=0,NXM
            U1(I,K,J)=VEC(I,J)
          END DO
        END DO

! End do k
      END DO

! Initialize the matrix used to store implicit coefficients
      DO J=0,NY+1
        DO I=0,NXM
          MATL(I,J)=0.
          MATD(I,J)=1.
          MATU(I,J)=0.
          VEC(I,J)=0.
        END DO
      END DO 

C Solve for U3
C Note, here the matrix will be indexed from 1...NY+1 corresponding to U1(0:NY)
C Build the implicit system of equations for U3
      DO K=0,NZP-1
        DO J=JSTART,JEND
          DO I=0,NXM
            MATL(I,J)=-TEMP1/(DY(J)*DYF(J))
            MATD(I,J)=1.-TEMP1*(-1./(DY(J+1)*DYF(J))
     &         -1./(DY(J)*DYF(J)))
            MATU(I,J)=-TEMP1/(DY(J+1)*DYF(J))
            VEC(I,J)=R3(I,K,J)
          END DO
        END DO
! IF using a subgrid model (LES) then add the eddy viscosity part implicitly
        IF (LES) THEN
        DO J=JSTART,JEND
          DO I=0,NXM
            MATL(I,J) = MATL(I,J) - TEMP2 * NU_T(I,K,J)
     &                               / (DY(J)*DYF(J))
            MATD(I,J) = MATD(I,J) + TEMP2 * NU_T(I,K,J+1)
     &                              / (DY(J+1)*DYF(J))
     &                            + TEMP2 * NU_T(I,K,J)
     &                              / (DY(J)*DYF(J))
            MATU(I,J) = MATU(I,J) - TEMP2 * NU_T(I,K,J+1)
     &                             / (DY(J+1)*DYF(J))
          END DO
        END DO
        END IF

        IF (USE_MPI) THEN
! First, apply the boundary conditions
          CALL APPLY_BC_U3_MPI(MATL,MATD,MATU,VEC)
! If we are using MPI, split the implicit solve into forward and
! backward sweeps for efficiency
          CALL THOMAS_FORWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
          CALL THOMAS_BACKWARD_REAL_MPI(MATL,MATD,MATU,VEC,NY,NX)
        ELSE
C Else, we are running in serial mode
C Set the boundary conditions for U3
          CALL APPLY_BC_3_LOWER(MATL,MATD,MATU,VEC)
          CALL APPLY_BC_3_UPPER(MATL,MATD,MATU,VEC)
C Now, solve the tridiagonal system for U3(i,:,k)
          CALL THOMAS_REAL(MATL,MATD,MATU,VEC,NY+1,NXM)
        END IF

        DO J=JSTART-1,JEND+1
          DO I=0,NXM
            U3(I,K,J)=VEC(I,J)
          END DO
        END DO
! End do k
      END DO

C -- Done getting U1hat, U2hat, U3hat at new RK Step --

C If Variable timestepping and done with one full R-K step, update
C DELTA_T based on the specified CFL number
C This is not parallelized and should be used only in the serial
C version to ensure that each process uses the same timestep
      IF ((VARIABLE_DT).and.(RK_STEP.eq.3)
     &        .and.(MOD(TIME_STEP,UPDATE_DT).EQ.0)) THEN
        CALL COURANT
      END IF

! Transform TH and U to Fourier Space 
      CALL FFT_XZ_TO_FOURIER(U1,CU1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(U2,CU2,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(U3,CU3,0,NY+1)
      DO N=1,N_TH
        CALL FFT_XZ_TO_FOURIER(TH(0,0,0,N),CTH(0,0,0,N),0,NY+1)
      END DO

C Begin second step of the Fractional Step algorithm, making u divergence free
C The following subroutine projects Uhat onto divergence free space

      CALL REM_DIV_CHAN

C Now, phi is stored in CR1, use this to update the pressure field
C Note, here we divide by H_BAR since it was absorbed into PHI in REM_DIV
      DO J=JSTART,JEND
        DO K=0,TNKZ
          DO I=0,NXP-1
            CP(I,K,J)=CP(I,K,J)+CR1(I,K,J)/TEMP4
          END DO
        END DO
      END DO

      ! Fix disparities at the boundary due to the thoms algorithm in parallel
      IF (USE_MPI) THEN
         CALL GHOST_CHAN_MPI
      END IF

      RETURN
      END SUBROUTINE RK_CHAN_1

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE RK_CHAN_2
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C Alternative time-stepping algorithm for the channel-flow case.
C This algorithm uses Crank-Nicolson for all viscous terms and 
C third order Runge-Kutta for all nonlinear terms
C INPUTS  (in Fourier space):  CUi, P, and (if k>1) CFi at (k-1)  (for i=1,2,3)
C OUTPUTS (in Fourier space):  CUi, P, and (if k<3) CFi at (k)
C Each RK step, there are 11 FFT calls. 11 storage variables are used.     
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|

      RETURN
      END SUBROUTINE RK_CHAN_2

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE REM_DIV_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      
C Compute varphi, store in variable CR1.
C Solves for phi in computational space
C H_BAR has been absorbed into PHI, so we are solving for H_BAR*PHI

      INTEGER I,J,K
 
C First, Initialize the matrix components
      DO J=0,NY+1
        DO I=0,NXP-1
          MATL_C(I,J)=0.
          MATD_C(I,J)=1.
          MATU_C(I,J)=0.
          VEC_C(I,J)=(0.,0.)
        END DO
      END DO

C The 2d FFT of Ui should have been taken and stored in CUi
C Solving for phi amounts to solving a tridiagonal system
C First, construct the system to be solved
      DO K=0,TNKZ
        DO J=1,NY
          DO I=0,NXP-1
            MATL_C(I,J)=1./(DY(J)*DYF(J))
            MATD_C(I,J)=-KX2(I)-KZ2(K)
     &         -1./(DY(J+1)*DYF(J))-1./(DY(J)*DYF(J))
            MATU_C(I,J)=1./(DY(J+1)*DYF(J))
          END DO
        END DO

C Now, create the RHS vector
        DO J=1,NY         
          DO I=0,NXP-1
            VEC_C(I,J)=(CIKX(I)*CU1(I,K,J) 
     &            + (CU2(I,K,J+1)-CU2(I,K,J))/DYF(J) 
     &            + CIKZ(K)*CU3(I,K,J))
          END DO
        END DO

        IF (USE_MPI) THEN
C If we are using the MPI package...
          CALL APPLY_BC_REM_DIV_MPI(MATL_C,MATD_C,MATU_C,VEC_C,K)
C First, do all forward sweeps
          CALL THOMAS_FORWARD_COMPLEX_MPI(MATL_C,MATD_C,MATU_C,VEC_C
     &                                 ,NY,NXP)
C Now, do the backward sweeps
          CALL THOMAS_BACKWARD_COMPLEX_MPI(MATL_C,MATD_C,MATU_C,VEC_C
     &                                 ,NY,NXP)
        ELSE
C Else we are running in serial mode
        DO I=0,NKX
          IF ((K.EQ.0).AND.(I.EQ.0)) THEN
C Use homogeneous dirichlet BCS for kx=kz=0 component at bottom wall
C Otherwise the matrix will be singular
            MATL_C(I,1)=0. 
            MATD_C(I,1)=1.
            MATU_C(I,1)=0.
            VEC_C(I,1)=(0.,0.)

            MATL_C(I,NY)=1.
            MATD_C(I,NY)=-1.
            MATU_C(I,NY)=0.
            VEC_C(I,NY)=(0.,0.)
          ELSE
C Use Dirichlet boundary conditions, dp/dz=0 at walls
            MATL_C(I,1)=0.
            MATD_C(I,1)=1.
            MATU_C(I,1)=-1.
            VEC_C(I,1)=(0.,0.)

            MATL_C(I,NY)=1.
            MATD_C(I,NY)=-1.
            MATU_C(I,NY)=0.
            VEC_C(I,NY)=(0.,0.)
          END IF
        END DO
C Now solve the tridiagonal system for phi, store in CR1
        CALL THOMAS_COMPLEX(MATL_C,MATD_C,MATU_C,VEC_C,NY,NKX)
        END IF


        DO J=1,NY
          DO I=0,NXP-1
            CR1(I,K,J)=VEC_C(I,J)
          END DO
        END DO

      END DO

C Now, Solve for CUi, the divergenceless velocity field
      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CU1(I,K,J)=CU1(I,K,J)-CIKX(I)*CR1(I,K,J)
            CU3(I,K,J)=CU3(I,K,J)-CIKZ(K)*CR1(I,K,J)           
          END DO
        END DO
      END DO
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CU2(I,K,J)=CU2(I,K,J)-(CR1(I,K,J)
     &             -CR1(I,K,J-1))/DY(J)
          END DO
        END DO
      END DO

      RETURN
      END SUBROUTINE REM_DIV_CHAN

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE POISSON_P_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C We have CUi, need to compute CP.  Solve tridiagonal system exactly


      INTEGER I,J,K,N
      
	if (flavor.eq.'Basic') then
      IF (RANK.EQ.0) 
     &          WRITE(*,*) 'COMPUTING CP FROM CUI'
	end if


C First, construct the RHS vector, (dui/dxj)(duj/dxi) 
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1 ! NKX
            CF1(I,K,J)=CIKX(I)*CU1(I,K,J)
            CF2(I,K,J)=(CU2(I,K,J+1)-CU2(I,K,J))/DYF(J)
            CF3(I,K,J)=CIKZ(K)*CU3(I,K,J)
          END DO
        END DO
      END DO

      CALL FFT_XZ_TO_PHYSICAL(CF1,F1,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CF2,F2,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CF3,F3,0,NY+1)
      
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            F1(I,K,J)=F1(I,K,J)**2.
            F2(I,K,J)=F2(I,K,J)**2.
            F3(I,K,J)=F3(I,K,J)**2.
          END DO
        END DO
      END DO

      CALL FFT_XZ_TO_FOURIER(F1,CF1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(F2,CF2,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(F3,CF3,0,NY+1)

C Now we have the diagonal terms, add to the rhs term
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1 ! NKX
            CS1(I,K,J)=CF1(I,K,J)+CF2(I,K,J)+CF3(I,K,J)
          END DO
        END DO
      END DO

C Now get the first of the off-diagonal terms
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1 ! NKX
            CF1(I,K,J)=(CU1(I,K,J+1)-CU1(I,K,J-1))/(2.*DYF(J))
            CF2(I,K,J)=CIKX(I)*0.5*(CU2(I,K,J)+CU2(I,K,J+1))
          END DO
        END DO
      END DO

      CALL FFT_XZ_TO_PHYSICAL(CF1,F1,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CF2,F2,0,NY+1)

C Compute product
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            F1(I,K,J)=2.*F1(I,K,J)*F2(I,K,J)
          END DO
        END DO
      END DO
      
      CALL FFT_XZ_TO_FOURIER(F1,CF1,0,NY+1)

C Add to RHS term
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1 ! NKX 
            CS1(I,K,J)=CS1(I,K,J)+CF1(I,K,J)
          END DO
        END DO
      END DO

C Now get the second of the off-diagonal terms
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1 ! NKX
            CF1(I,K,J)=(CU3(I,K,J+1)-CU3(I,K,J-1))/(2.*DYF(J))
            CF2(I,K,J)=CIKZ(K)*0.5*(CU2(I,K,J)+CU2(I,K,J+1))
          END DO
        END DO
      END DO

C Convert to Physical space
      CALL FFT_XZ_TO_PHYSICAL(CF1,F1,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CF2,F2,0,NY+1)

C Compute product
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            F1(I,K,J)=2.*F1(I,K,J)*F2(I,K,J)
          END DO
        END DO
      END DO

      CALL FFT_XZ_TO_FOURIER(F1,CF1,0,NY+1)

C Add to RHS term
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1  ! NKX
            CS1(I,K,J)=CS1(I,K,J)+CF1(I,K,J)
          END DO
        END DO
      END DO

C Now get the third of the off-diagonal terms
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1  ! NKX
            CF1(I,K,J)=CIKZ(K)*CU1(I,K,J)
            CF2(I,K,J)=CIKX(I)*CU3(I,K,J)
          END DO
        END DO
      END DO

      CALL FFT_XZ_TO_PHYSICAL(CF1,F1,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CF2,F2,0,NY+1)
      
C Compute product
      DO J=2,NY
        DO K=0,NZP-1
          DO I=0,NXM
            F1(I,K,J)=2.*F1(I,K,J)*F2(I,K,J)
          END DO
        END DO
      END DO

      CALL FFT_XZ_TO_FOURIER(F1,CF1,0,NY+1)

C Add to RHS term
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1  ! NKX
            CS1(I,K,J)=CS1(I,K,J)+CF1(I,K,J)
          END DO
        END DO
      END DO     
    
C Finally, if the buoyancy force is active, then we need to add
C the contribution of the density to the pressure.  Note that the
C plane averaged density and the corresponding hydrostatic part of the
C pressure have been cancelled, so skip the 0,0 mode
      DO N=1,N_TH
      DO J=2,NY
        DO K=0,TNKZ
          DO I=0,NXP-1  ! NKX  
            IF ((RANKZ.NE.0).OR.(I.NE.0).or.(K.NE.0)) THEN
              CS1(I,K,J)=CS1(I,K,J)+RI_TAU(N)*
     &          (CTH(I,K,J+1,N)-CTH(I,K,J-1,N))/(GYF(J+1)-GYF(J-1))
            END IF
          END DO
        END DO
      END DO
      END DO

C Now, the RHS term should be stored in CS1     

C Construct the tridiagonal system in Fourier space to solve for CP
C First, zero the vectors
      DO J=0,NY+1
        DO I=0,NXP-1
          MATL_C(I,J)=0.d0
          MATD_C(I,J)=1.d0
          MATU_C(I,J)=0.d0
          VEC_C(I,J)=(0.,0.)
        END DO
      END DO

      DO K=0,TNKZ
        DO J=2,NY
          DO I=0,NXP-1
            MATL_C(I,J)=1./(DY(J)*DYF(J))
            MATD_C(I,J)=-KX2(I)-KZ2(K)-1./(DY(J+1)*DYF(J))
     &                    -1./(DY(J)*DYF(J))
            MATU_C(I,J)=1./(DY(J+1)*DYF(J))   
            VEC_C(I,J)=-1.*CS1(I,K,J)
          END DO
        END DO

        IF (USE_MPI) THEN
          CALL APPLY_BC_POISSON_MPI(MATL_C,MATD_C,MATU_C,VEC_C,K)
C First, do the forward sweeps
          CALL THOMAS_FORWARD_COMPLEX_MPI(MATL_C,MATD_C,MATU_C,VEC_C
     &                                 ,NY,NXP)
C Now, do the backwared sweeps to put the solution in VEC_C
          CALL THOMAS_BACKWARD_COMPLEX_MPI(MATL_C,MATD_C,MATU_C,VEC_C
     &                                  ,NY,NXP)
        ELSE
C Else we are running in serial mode
C Apply BCs
        DO I=0,NKX
C Use dirichlet boundary condition at the lower wall to
C prevent the tridiagonal matrix from becomming singular for i,k=0
          IF ((I.EQ.0).AND.(K.EQ.0)) THEN
            MATD_C(I,1)=1.
            MATU_C(I,1)=0.
            VEC_C(I,1)=(0.,0.)
            MATD_C(I,NY)=-1.
            MATL_C(I,NY)=1.
            VEC_C(I,NY)=(0.,0.)
          ELSE
! Here, apply Neumann boundary conditions (dp/dz=0) at the walls
            MATD_C(I,1)=1.
            MATU_C(I,1)=-1.
            VEC_C(I,1)=(0.,0.)
            MATD_C(I,NY)=-1.
            MATL_C(I,NY)=1.
            VEC_C(I,NY)=(0.,0.)
          END IF
        END DO
C Now, solve for CP
        CALL THOMAS_COMPLEX(MATL_C,MATD_C,MATU_C,VEC_C,NY,NKX)
        END IF

        DO J=1,NY
          DO I=0,NXP-1
            CP(I,K,J)=VEC_C(I,J)
          END DO
        END DO
      END DO

      RETURN
      END SUBROUTINE POISSON_P_CHAN

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE CREATE_TH_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C Initialize the scalar fields
C In this subroutine, you should initialize each scalar field for the
C particular problem of interest

      INTEGER I,J,K,N

      DO N=1,N_TH
        IF (CREATE_NEW_TH(N)) THEN

      IF (IC_TYPE.eq.0) THEN
       DO K=0,NZP-1
         DO I=0,NXM
           DO J=1,NY
             TH(I,K,J,N)=sin(2.d0*PI*GX(I)/LX)/(4.d0*PI**2.d0)
           END DO
         END DO
       END DO
       ELSE IF ((IC_TYPE.eq.1).or.(IC_TYPE.eq.2)) THEN
       DO K=0,NZP-1
         DO I=0,NXM
           IF ((TH_BC_YMIN(N).EQ.0).AND.(TH_BC_YMAX(N).EQ.0)) THEN
               DO J=1,NY
! Initialize with a linear profile using the bcs
               IF (GYF(J).LE.2.0) THEN
                 TH(I,K,J,N)=(TH_BC_YMAX_C1(N)-TH_BC_YMIN_C1(N))
     &                *(GYF(J)+1.)/2.0+TH_BC_YMIN_C1(N)
               ELSE
                 TH(I,K,J,N)=TH_BC_YMAX_C1(N)
               END IF
             END DO
           ELSE IF ((TH_BC_YMIN(N).EQ.1)
     &            .AND.(TH_BC_YMAX(N).EQ.1)) THEN
             DO J=1,NY
! Linear profile with slope corresponding to upper value
                TH(I,K,J,N)=TH_BC_YMAX_C1(N)*GYF(J)
              END DO
           ELSE
             IF (RANK.EQ.0) then
                WRITE(*,*) 'WARNING, THETA INITIALIZED TO ZERO ...'
                WRITE(*,*) 'CREATE AN INITIAL VALUE IN CREATE_FLOW_CHAN'
             end if
           END IF
         END DO
        END DO
      ELSE IF (IC_TYPE.eq.3) THEN
! Shear layer
       DO J=0,NY
         DO K=0,NZP-1
           DO I=0,NXM
             TH(I,K,J,N)=TANH(GYF(J)*15.d0)+1.d0
            END DO
          END DO
        END DO
      END IF
      
      CALL FFT_XZ_TO_FOURIER(TH(0,0,0,n),CTH(0,0,0,n),0,NY+1)

      END IF
      END DO


      RETURN
      END SUBROUTINE CREATE_TH_CHAN


C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE CREATE_FLOW_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      INTEGER I,J,K
      REAL*8 RNUM1,RNUM2,RNUM3
      INTEGER, DIMENSION(:), ALLOCATABLE :: seed

C Initialize the random number generator
      CALL RANDOM_SEED(SIZE = K)
      Allocate (seed(1:K))
      seed(1:K)=10
      CALL RANDOM_SEED(PUT = seed)

C UBULK0 and KICK should be set in input.dat

C Set the laminar velocity profile in physical space
       IF (RANK.EQ.0) 
     &     write(*,*) 'UBULK0: ',UBULK0

       IF (IC_TYPE.eq.0) then
C For closed channel flow
       DO J=0,NY
         DO K=0,NZP-1
           DO I=0,NXM
             U1(I,K,J)=(3./2.)*UBULK0*(1.d0-GYF(J)**2.)
             U2(I,K,J)=0.
             U3(I,K,J)=0.
           END DO
         END DO
      END DO
      else if (IC_TYPE.eq.1) then 
C For open channel flow :
       DO K=0,NZP-1
         DO I=0,NXM
           DO J=1,NY
             U1(I,K,J)=-(3./2.)*UBULK0*GYF(J)**2.+3.*UBULK0*GYF(J)
             U2(I,K,J)=0.
             U3(I,K,J)=0.
           END DO
           U1(I,K,0)=0.
           U3(I,K,0)=0.
           U1(I,K,NY+1)=0.
           U3(I,K,NY+1)=0.
         END DO
      END DO
      else if (IC_TYPE.eq.2) then
C For Couette flow:
       DO J=0,NY
         DO K=0,NZP-1
           DO I=0,NXM
             U1(I,K,J)=gyf(j)
             U2(I,K,J)=0.
             U3(I,K,J)=0.
           END DO
         END DO
      END DO
      else if (IC_TYPE.eq.3) then
! Shear layer
       DO J=0,NY+1
         DO K=0,NZ+1
           DO I=0,NX+1
             U1(I,K,J)=TANH(GYF(J)*20.d0)
             U2(I,K,J)=0.d0
             U3(I,K,J)=0.d0
            END DO
          END DO
        END DO
      end if

C Zero the ghost cells
       IF (.NOT.USE_MPI) THEN
       DO K=0,NZM
         DO I=0,NXM
           U1(I,K,0)=0.
           U2(I,K,0)=0.
           U3(I,K,0)=0.
           U1(I,K,NY+1)=0.
           U2(I,K,NY+1)=0.
           U3(I,K,NY+1)=0.
         END DO
      END DO
      END IF
      
      CALL FFT_XZ_TO_FOURIER(U1,CU1,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(U2,CU2,0,NY+1)
      CALL FFT_XZ_TO_FOURIER(U3,CU3,0,NY+1)

      CALL SAVE_STATS_CHAN(.FALSE.)

      IF (RANK.EQ.0) 
     &     WRITE(*,*) 'KICK: ',KICK
      IF (RANK.EQ.0) 
     &     write(*,*) 'NKX,NY,TNKZ: ',NKX,NY,TNKZ
 
      DO I=1,NXP-1
        DO J=1,NY
          DO K=1,TNKZ
C Now, give the velocity field a random perturbation
            CALL RANDOM_NUMBER(RNUM1)
            CALL RANDOM_NUMBER(RNUM2)
            CALL RANDOM_NUMBER(RNUM3)

            IF (IC_TYPE.eq.3) THEN
C If we are initializing with a shear layer 
              CU1(I,K,J)=CU1(I,K,J)
     &             +(RNUM1-0.5)*KICK*EXP(-(GYF(J)*20.d0)**2.d0)
              CU2(I,K,J)=CU2(I,K,J)
     &             +(RNUM1-0.5)*KICK*EXP(-(GY(J)*20.d0)**2.d0)
              CU3(I,K,J)=CU3(I,K,J)
     &             +(RNUM1-0.5)*KICK*EXP(-(GYF(J)*20.d0)**2.d0)
            ELSE 
              CU1(I,K,J)=CU1(I,K,J)+(RNUM1-0.5)*KICK
              CU2(I,K,J)=CU2(I,K,J)+(RNUM2-0.5)*KICK
              CU3(I,K,J)=CU3(I,K,J)+(RNUM3-0.5)*KICK
            END IF
          END DO
          IF (TNKZ.EQ.0) THEN
! Here, In the 2d case we want to add a kick to the mean in z
            K=0         
            CALL RANDOM_NUMBER(RNUM1)
            CALL RANDOM_NUMBER(RNUM2)
            CALL RANDOM_NUMBER(RNUM3)

            IF (IC_TYPE.eq.3) THEN
              CU1(I,K,J)=CU1(I,K,J)
     &             +(RNUM1-0.5)*KICK*EXP(-(GYF(J)*20.d0)**2.d0)
              CU2(I,K,J)=CU2(I,K,J)
     &             +(RNUM1-0.5)*KICK*EXP(-(GYF(J)*20.d0)**2.d0)
              CU3(I,K,J)=CU3(I,K,J)
     &             +(RNUM1-0.5)*KICK*EXP(-(GYF(J)*20.d0)**2.d0)
            ELSE
              CU1(I,K,J)=CU1(I,K,J)+(RNUM1-0.5)*KICK
              CU2(I,K,J)=CU2(I,K,J)+(RNUM2-0.5)*KICK
              CU3(I,K,J)=CU3(I,K,J)+(RNUM3-0.5)*KICK
            END IF
          END IF 

        END DO
      END DO

      IF (USE_MPI) THEN
        CALL GHOST_CHAN_MPI
      END IF

C Apply Boundary conditions to velocity field
      IF (USE_MPI) THEN
        CALL APPLY_BC_VEL_MPI
      ELSE
        CALL APPLY_BC_VEL_LOWER
        CALL APPLY_BC_VEL_UPPER
      END IF

C Remove the divergence of the velocity field


      CALL REM_DIV_CHAN

      IF (USE_MPI) THEN
        CALL GHOST_CHAN_MPI
      END IF

C Get the pressure from the poisson equation
      CALL POISSON_P_CHAN

! Fix for the pressure
      IF (USE_MPI) THEN
        CALL GHOST_CHAN_MPI
      END IF

      CALL SAVE_STATS_CHAN(.FALSE.)

      RETURN
      END SUBROUTINE CREATE_FLOW_CHAN

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE VIS_FLOW_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C Convert to physical space and output the velocity and pressure 
C to be read for visualization

      CHARACTER*35 FNAME,FNAME_TH
      INTEGER      I, J, K, n,yind

      FNAME='diablo.vis'
      IF (RANK.EQ.0) 
     &     WRITE(6,*) 'Writing flow to ',FNAME
      OPEN(UNIT=10,FILE=FNAME,STATUS="UNKNOWN",FORM="UNFORMATTED")

      FNAME_TH='diablo_th.vis'
      IF (RANK.EQ.0) 
     &     WRITE(6,*) 'Writing flow to ',FNAME_TH
      OPEN(UNIT=20,FILE=FNAME_TH,STATUS="UNKNOWN",FORM="UNFORMATTED")

C Write out grid at GYF points
      open(unit=11,file='ygrid_out.txt'
     &       ,status='unknown',form='formatted')
      write(11,111) (GYF(j),j=2,NYM)
      close(11)
111   format(64(F16.8,' '))
      open(unit=12,file='xgrid_out.txt'
     &       ,status='unknown',form='formatted')
      write(12,112) (GX(i),i=0,NXM)
      close(12)
112   format(64(F16.8,' '))
      open(unit=13,file='zgrid_out.txt'
     &      ,status='unknown',form='formatted')
      write(13,113) (GZ(k),k=0,NZM)
      close(13)
113   format(64(F16.8,' '))

      IF (NUM_PER_DIR.EQ.3) THEN
C        Enter write statment here
      ELSEIF (NUM_PER_DIR.EQ.2) THEN
C Convert to physical space
      call fft_xz_to_physical(CU1,U1,0,NY+1)
      call fft_xz_to_physical(CU2,U2,0,NY+1)
      call fft_xz_to_physical(CU3,U3,0,NY+1)
      call fft_xz_to_physical(CP,P,0,NY+1)
      do n=1,N_TH
        call fft_xz_to_physical(CTH(0,0,0,n),TH(0,0,0,n),0,NY+1)
      end do
C Interpolate the vertical velocity to GYF gridpoints
        WRITE(10) ((( 
     &    REAL(U1(I,K,J)),REAL(0.5*(U2(I,K,J)+U2(I,K,J+1)))
     &    ,REAL(U3(I,K,J)),REAL(U1(I,K,J)),
     &    REAL(0.5*(U2(I,K,J)+U2(I,K,J+1))),REAL(U3(I,K,J)),
     &    REAL(P(I,K,J))
     &    ,K=0,NZM),J=2,NYM),I=0,NXM)

        WRITE(20) ((((
     &     REAL(TH(I,K,J,n))
     &        ,n=1,N_TH),K=0,NZM),J=2,NYM), I=0,NXM)

C Output velocity field for input to LIC (Line integral convolution)
      open(61,file='lic_x.dat',form='formatted',status='unknown')
      write(61,*) NZ,NY
      open(62,file='lic_y.dat',form='formatted',status='unknown')
      write(62,*) NZ,NY
	
	yind = ny
	if (ny.ge.30) yind=30
      do j=1,NY
        write(61,161) ((real(U3(yind,k,j))),k=0,NZM)
        write(62,161) ((real(U2(yind,k,j))),k=0,NZM)
      end do
161   format(192(F8.3))
      close(61)
      close(62)

      ELSEIF (NUM_PER_DIR.EQ.1) THEN
C        Enter write statment here
      ELSEIF (NUM_PER_DIR.EQ.0) THEN
C        Enter write statment here

      END IF
101   format(10(F16.8,' '))
      CLOSE(10)
      CLOSE(20)


C Compute the discrimenant for visualization of vortices
C Note, the velocity field will be destroyed by this calculation,
C so it is important that this is only done at the end of a simulation

C NOTE:  THIS SECTION NEEDS TO BE CHECKED, THERE MAY BE AN ERROR
C IN THE CALCULATION OF THE DISCRIMINANT

      IF ((NUM_PER_DIR.EQ.2)) THEN
C First, convert back to Fourier space
      call fft_xz_to_fourier(U1,CU1,0,NY+1)
      call fft_xz_to_fourier(U2,CU2,0,NY+1)
      call fft_xz_to_fourier(U3,CU3,0,NY+1)
C First, calculate the velocity gradient tensor at GYF points
      do j=2,NYM
        do k=0,TNKZ
          do i=0,NKX
            CA21(i,k,j)=CIKX(i)*0.5*(CU2(i,k,j+1)+CU2(i,k,j))
            CA31(i,k,j)=CIKX(i)*CU3(i,k,j)
            CA12(i,k,j)=(0.5*(CU1(i,k,j+1)+CU1(i,k,j))
     &                 - 0.5*(CU1(i,k,j)+CU1(i,k,j-1)))/DYF(j) 
            CA32(i,k,j)=(0.5*(CU3(i,k,j+1)+CU3(i,k,j))
     &                 - 0.5*(CU3(i,k,j)+CU3(i,k,j-1)))/DYF(j) 
            CA13(i,k,j)=CIKZ(k)*CU1(i,k,j)
            CA23(i,k,j)=CIKZ(k)*0.5*(CU2(i,k,j+1)+CU2(i,k,j))
C Now, the following will overwrite CUi
            CA11(i,k,j)=CIKX(i)*CU1(i,k,j)
            CA22(i,k,j)=(CU2(i,k,j+1)-CU2(i,k,j))/DYF(j)
            CA33(i,k,j)=CIKZ(k)*CU3(i,k,j)
          end do
        end do
      end do
C Transform to physicl space
      call fft_xz_to_physical(CA11,A11,0,NY+1)
      call fft_xz_to_physical(CA21,A21,0,NY+1)
      call fft_xz_to_physical(CA31,A31,0,NY+1)
      call fft_xz_to_physical(CA12,A12,0,NY+1)
      call fft_xz_to_physical(CA22,A22,0,NY+1)
      call fft_xz_to_physical(CA32,A32,0,NY+1)
      call fft_xz_to_physical(CA13,A13,0,NY+1)
      call fft_xz_to_physical(CA23,A23,0,NY+1)
      call fft_xz_to_physical(CA33,A33,0,NY+1)

C Defining S_ij=(A_ij+A_ji)/2, compute Strain_rate = S_ij S_ji
      do j=2,NYM
        do k=0,NZM
          do i=0,NXM
            Strain_rate(i,k,j)=0.5*((A12(i,k,j)+A21(i,k,j))**2.
     &                             +(A13(i,k,j)+A31(i,k,j))**2. 
     &                             +(A23(i,k,j)+A32(i,k,j))**2.)
     &                    +A11(i,k,j)**2.+A22(i,k,j)**2.+A33(i,k,j)**2. 
          end do
C Compute third invariant = -det(A)
C Overwrites A11
          do i=0,NXM
            Third_ivar(I,K,J) =
     &  - A11(I,K,J)*(A22(I,K,J)*A33(I,K,J) - A23(I,K,J)*A32(I,K,J))
     &  + A12(I,K,J)*(A21(I,K,J)*A33(I,K,J) - A23(I,K,J)*A31(I,K,J))
     &  - A13(I,K,J)*(A21(I,K,J)*A32(I,K,J) - A22(I,K,J)*A31(I,K,J))
          end do

C Defining Omega_ij=(A_ij-A_ji)/2, compute Enstrophy=-(Omega_ij Omega_ji). 
C Note that this loop overwrites A22.
          do i=0,NXM
            Enstrophy  (I,K,J) = 0.5*((A12(I,K,J) - A21(I,K,J))**2+
     &        (A13(I,K,J) - A31(I,K,J))**2+ 
     &        (A23(I,K,J) - A32(I,K,J))**2)
          end do

C Compute Second_ivar.
C Note that this loop overwrites A33.
          do i=0,NXM
            Second_ivar(I,K,J)=0.5*(Enstrophy(I,K,J)-Strain_rate(I,K,J))
          end do

C Compute Discriminant.
C Note that this loop overwrites A12.
          do i=0,NXM
            Discriminant(I,K,J) = 6.75*(Third_ivar(I,K,J))**2
     &       + (Second_ivar(I,K,J))**3
          end do
        end do 
      end do 
           
      OPEN(UNIT=20,FILE='inv.vis',STATUS='UNKNOWN',FORM='UNFORMATTED')
        WRITE(20) ((( 
     &    REAL(GX(I)),REAL(GYF(J))
     &    ,REAL(GZ(K)),REAL(Discriminant(i,k,j)),
     &    REAL(Enstrophy(i,k,j)),REAL(Strain_rate(i,k,j))
     &    ,K=0,NZM),J=2,NYM),I=0,NXM)
      CLOSE(20)

      END IF

      RETURN
      END SUBROUTINE VIS_FLOW_CHAN


C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE INPUT_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL    VERSION, CURRENT_VERSION
      INTEGER I,J,K,N

! Read in input parameters specific for channel flow case
      OPEN (11,file='input_chan.dat',form='formatted',status='old')      
C Read input file.

      CURRENT_VERSION=1.0
      READ(11,*)
      READ(11,*)
      READ(11,*)
      READ(11,*)
      READ(11,*) VERSION
      IF (VERSION .NE. CURRENT_VERSION) 
     &         STOP 'Wrong input data format input_chan'
      READ(11,*)
      READ(11,*) TIME_AD_METH
      READ(11,*) 
      READ(11,*) LES_MODEL_TYPE
      READ(11,*)
      READ(11,*) IC_TYPE, KICK
      READ(11,*)
      READ(11,*) I_RO_TAU
      READ(11,*)
      READ(11,*) F_TYPE, UBULK0, PX0, OMEGA0, AMP_OMEGA0
      READ(11,*)
      READ(11,*) U_BC_YMIN, U_BC_YMIN_C1, U_BC_YMIN_C2, U_BC_YMIN_C3
      READ(11,*) 
      READ(11,*) V_BC_YMIN, V_BC_YMIN_C1, V_BC_YMIN_C2, V_BC_YMIN_C3
      READ(11,*)
      READ(11,*) W_BC_YMIN, W_BC_YMIN_C1, W_BC_YMIN_C2, W_BC_YMIN_C3
      READ(11,*)
      READ(11,*) U_BC_YMAX, U_BC_YMAX_C1, U_BC_YMAX_C2, U_BC_YMAX_C3
      READ(11,*)
      READ(11,*) V_BC_YMAX, V_BC_YMAX_C1, V_BC_YMAX_C2, V_BC_YMAX_C3
      READ(11,*)
      READ(11,*) W_BC_YMAX, W_BC_YMAX_C1, W_BC_YMAX_C2, W_BC_YMAX_C3
      READ(11,*)
! Give the exate value for the bulk velocity
      UBULK0=2.0D0/3.0D0
! Read in boundary conditions for the N_TH scalars
      DO N=1,N_TH
        READ(11,*)
        READ(11,*) TH_BC_YMIN(N),TH_BC_YMIN_C1(N),TH_BC_YMIN_C2(N)
     &             ,TH_BC_YMIN_C3(N)
        READ(11,*)
        READ(11,*) TH_BC_YMAX(N),TH_BC_YMAX_C1(N),TH_BC_YMAX_C2(N)
     &             ,TH_BC_YMAX_C3(N)
      END DO

! Create instantaneous statistics files
      IF(RANK.EQ.0) THEN
        OPEN(UNIT=1000,FILE='trace.d',STATUS='UNKNOWN')
        OPEN(UNIT=1001,FILE='Euvw_ypm50_all.d',STATUS='UNKNOWN')
        OPEN(UNIT=1002,FILE='Euvw_ypm50_tcore.d',STATUS='UNKNOWN')
        OPEN(UNIT=1003,FILE='Euvw_ypm50_bcore.d',STATUS='UNKNOWN')
      END IF
! Initialise spectra variable
      phiuu(:,:,:)=0.0D0
      phivv(:,:,:)=0.0D0
      phiww(:,:,:)=0.0D0
      phiuv(:,:,:)=0.0D0
      phipp(:,:,:)=0.0D0

! PI
! Initialise spectra variable
      pixsub(:,:,:)=(0.0D0,0.0D0)
      pix(:,:,:)=0.0D0
      piysub(:,:,:)=(0.0D0,0.0D0)
      piy(:,:,:)=0.0D0
      pizsub(:,:,:)=(0.0D0,0.0D0)
      piz(:,:,:)=0.0D0
      prod(:,:,:)=0.0D0
      prestrans(:,:,:)=0.0D0
      diffx(:,:,:)=0.0D0
      diffy(:,:,:)=0.0D0
      diffz(:,:,:)=0.0D0
      vistransx(:,:,:)=0.0D0
      vistransy(:,:,:)=0.0D0
      vistransz(:,:,:)=0.0D0
      turbtransx(:,:,:)=0.0D0
      turbtransy(:,:,:)=0.0D0
      turbtransz(:,:,:)=0.0D0
      tempu(:,:,:)=0.0D0
      sgsx(:,:,:)=0.0D0
      sgsy(:,:,:)=0.0D0
      sgsz(:,:,:)=0.0D0
      CSijc(:,:,:,:)=(0.0D0,0.0D0)
      sumx1(:,:,:)=(0.0D0,0.0D0)
      sumy1(:,:,:)=(0.0D0,0.0D0)
      sumz1(:,:,:)=(0.0D0,0.0D0)
      sumx2(:,:,:)=(0.0D0,0.0D0)
      sumy2(:,:,:)=(0.0D0,0.0D0)
      sumz2(:,:,:)=(0.0D0,0.0D0)
      sumx3(:,:,:)=(0.0D0,0.0D0)
      sumy3(:,:,:)=(0.0D0,0.0D0)
      sumz3(:,:,:)=(0.0D0,0.0D0)
      ta=0
      dai=0.0D0
      Sijc(:,:,:,:)=0.0D0
      Aijc(:,:,:,:)=0.0D0
      CAijc(:,:,:,:)=(0.0D0,0.0D0)
!      INTEGER U_BC_LOWER,W_BC_LOWER,U_BC_UPPER,W_BC_UPPER
      betaVc(:,:,:)=0.0D0
      beta11c(:,:,:)=0.0D0
      beta22c(:,:,:)=0.0D0
      beta33c(:,:,:)=0.0D0
      beta12c(:,:,:)=0.0D0
      beta13c(:,:,:)=0.0D0
      beta23c(:,:,:)=0.0D0
      TEMPYc(:,:,:)=0.0D0
      facc(:,:,:)=0.0D0
      del_xc=0.0D0
      del_zc=0.0D0
      NU_Tc(:,:,:)=0.0D0
      NU_T2c(:,:,:)=0.0D0
      RETURN
      END SUBROUTINE INPUT_CHAN

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE CREATE_GRID_CHAN
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      CHARACTER*55 FNAME
      INTEGER I,J,K

         IF (RANK.EQ.0) 
     &     WRITE (6,*) 'Fourier in X'
         DO I=0,NX
           GX(I)=(I*LX)/NX
           DX(I)=LX/NX
           IF (VERBOSITY .GT. 3 .AND. RANK.EQ.0) 
     &          WRITE(6,*) 'GX(',I,') = ',GX(I)
         END DO
         IF (RANK.EQ.0) 
     &        WRITE (6,*) 'Fourier in Z'
         DO K=0,NZ
           GZ(K)=(K*LZ)/NZ
           DZ(K)=LZ/NZ
           IF (RANK.EQ.0 .AND. VERBOSITY .GT. 3) 
     &          WRITE(6,*) 'GZ(',K,') = ',GZ(K)
         END DO
         IF (RANK.EQ.0) 
     &        WRITE (6,*) 'Finite-difference in Y'

         IF (RANK.EQ.0) 
     &        write(*,*) 'USE_MPI: ',USE_MPI

         FNAME='grid.h5'
         if (FNAME(len_trim(FNAME)-2:len_trim(FNAME)).eq.".h5") then
#ifdef HDF5
            call ReadGridHDF5(FNAME,2)
#else
            IF (RANK.EQ.0) THEN
            write(*,*) ' **** ERROR ******************************'
            write(*,*) ' Program not compiled with HDF5 libraries.'
            END IF
            stop 
#endif
         else 
         IF (USE_MPI) THEN
           FNAME='./ygrid'//trim(MPI_IO_NUM)//'.txt'
           IF (RANK.EQ.0) THEN
           write(*,*) 'FNAME: ',FNAME
           write(*,*) 'MPI_IO_NUM: ****',trim(MPI_IO_NUM),'*****'
           END IF
         END IF

         OPEN (30,file=FNAME,form='formatted',status='old')
         READ (30,*) NY_T
C Check to make sure that grid file is the correct dimensions
         IF (NY_T.ne.NY) THEN
           IF (RANK.EQ.0) 
     &           WRITE(6,*) 'NY, NY_T',NY,NY_T
           STOP 'Error: ygrid.txt wrong dimensions'
         END IF
         DO J=1,NY+1
           READ(30,*) GY(j)
           IF (VERBOSITY .GT. 3 .AND. RANK.EQ.0) 
     &          WRITE(6,*) 'GY(',J,') = ',GY(J)
         END DO
         DO J=1,NY
           READ(30,*) GYF(j)
           IF (RANK.EQ.0 .AND. VERBOSITY .GT. 3)
     &          WRITE(6,*) 'GYF(',J,') = ',GYF(J)
         END DO
         CLOSE(30)

         IF (USE_MPI) THEN
           CALL GHOST_GRID_MPI
         ELSE
C Define ghost cells
           GYF(0)=2.d0*GYF(1)-GYF(2)
           GYF(NY+1)=2.d0*GYF(NY)-GYF(NYM)
           GY(0)=2.d0*GY(1)-GY(2)
         END IF
         end if

C Define grid spacing
         DO J=1,NY+1
           DY(J)=(GYF(J)-GYF(J-1))
         END DO
         DO J=1,NY
           DYF(J)=(GY(J+1)-GY(J))
         END DO
!         DYF(NY+1)=DYF(NY)
C Define grid spacing of the ghost cells
         IF (USE_MPI)
     &      CALL GHOST_GRIDSPACING_MPI

!     Get parameter for the movie
         IF (MOVIE) THEN 
            OPEN(unit=650,file='MOVIE.dat',status='old',
     &           form='formatted')
            READ(650, *) XcMovie, YcMovie, ZcMovie
            CLOSE(650)
            
!     Get the indices
            NxMovie=int(XcMovie*NX/LX)
            NyMovie=nint((YcMovie+LY/2)*(((NY-1)*NPROCS)/LY))
            NzMovie=int(ZcMovie*NZ/LZ)
         
            RankZMovie = int(NzMovie/NZP)
            NzMovie    = NzMovie-RankZMovie*NZP

            RankYMovie=-1
            IF (GYF(1).LT.YcMovie .and. GYF(NY).GE.YcMovie) THEN
               RankYMovie=RANKY
               I=1
               do while(.not.
     &              (GYF(I).LT.YcMovie .and. GYF(I+1).GE.YcMovie))
                  I=I+1
               end do
               NyMovie=I;
            END IF

            if (RANKY.eq.RankYMovie .and. RANKZ.eq.RankZMovie) then
               write(*,*) ' Movie Parameters: '
c$$$               write(*,*) '    Xc: ', LX/NX*NxMovie, ' (NxMovie: ', 
c$$$     &              NxMovie, ')'
c$$$               write(*,*) '    Yc: ', LY/((NY-1)*NPROCS+1)*NyMovie-LY/2
c$$$     &              , ' (NyMovie: ', NyMovie, ')'
c$$$               write(*,*) '    Zc: ', LZ/NZ*NzMovie, ' (NzMovie: ', 
c$$$     &              NzMovie, ')'
               write(*,*) '    Xc: ', GX(NxMovie), ' (NxMovie: ', 
     &              NxMovie, ')'
               write(*,*) '    Yc: ', GYF(NyMovie)
     &              , ' (NyMovie: ', NyMovie, ')'
               write(*,*) '    Zc: ', GZ(RankZMovie*NZP+NzMovie), 
     &              ' (NzMovie: ', RankZMovie*NZP+NzMovie, ')'

            END IF
         end if

         RETURN 
         END SUBROUTINE CREATE_GRID_CHAN



C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE SAVE_STATS_CHAN(FINAL)
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|

      CHARACTER*35 FNAME
      CHARACTER*10 GNAME
      LOGICAL FINAL
      integer i,j,k,n
      real*8 uc, ubulk
      REAL*8 STAT(1:NY,1:18)
      COMPLEX*16 CVC,CVC2
      
      real*8 varxy(0:NXM,1:NY),varzy(0:NZP-1,1:NY),varxz(0:NXM,0:NZP-1)
      real*8 uxz(0:NXM,0:NZP-1),vxz(0:NXM,0:NZP-1),wxz(0:NXM,0:NZP-1)
            
      IF (RANK.EQ.0) 
     &     WRITE(6,*) 'Saving flow statistics.'

      IF (USE_MPI) THEN
        CALL GHOST_CHAN_MPI
      END IF

C Apply Boundary conditions to velocity field
      IF (USE_MPI) THEN
        CALL APPLY_BC_VEL_MPI
      ELSE
        CALL APPLY_BC_VEL_LOWER
        CALL APPLY_BC_VEL_UPPER
      END IF

      if (FINAL) then
! We are done with the simulation
! Close the NetCDF file
!       CALL NETCDF_CLOSE_STATS_CHAN
#ifdef HDF5
! Save spectra
       CALL WriteHDF5_SPECTRA 

#else
        IF (RANKZ.EQ.0) THEN
        IF (USE_MPI) THEN
          FNAME='stats'//trim(MPI_IO_NUM)//'.txt'
        ELSE
          FNAME='stats.txt'
        END IF

        open(20,file=FNAME,form='formatted',status='unknown')
        do j=1,NY
          write(20,201) j,GYF(j),UBAR(j),VBAR(j),WBAR(j)
        end do
201     format(I3,',',F16.9,',',F16.9,',',F16.9,',',F16.9)
        do n=1,N_TH
        do j=1,NY
          write(20,202) j,GYF(j),THBAR(j,n)
        end do
        end do
202     format(I3,',',F16.9,',',F16.9)
        close(20)
        END IF
#endif
      else

!! Compute and write out the centerline velocity
!      IF (NPROCY.EQ.1) THEN
!      if (int(float(NY)/2.) .eq. float(NY)/2.) then
!! IF NY is even
!        uc=dble(CU1(0,0,int(float(NY)/2.))) 
!      else
!        uc=0.5*(dble(CU1(0,0,int(float(NY)/2.)-1))
!     +         +dble(CU1(0,0,int(float(NY)/2.))))
!      end if
!      write(*,*) 'Centerline velocity = ', uc 
!! Compute and write out bulk velocity
!      END IF

! We are in the middle of a run, compile statistics
! First get the number of samples taken so far
      IF (RANK.EQ.0) write(*,*) 'TIME, DELTA_T: ',TIME, DELTA_T
      NSAMPLES=NSAMPLES+1
      IF (RANKZ.EQ.0) THEN
! Get the mean velocity
         do j=1,NY
            UBAR(j)=(1./float(NSAMPLES))*dble(CU1(0,0,j))
     &           +((float(NSAMPLES)-1.)/float(NSAMPLES))*UBAR(j)
            VBAR(j)=(1./float(NSAMPLES))*dble(CU2(0,0,j))
     &           +((float(NSAMPLES)-1.)/float(NSAMPLES))*VBAR(j)
            WBAR(j)=(1./float(NSAMPLES))*dble(CU3(0,0,j))
     &           +((float(NSAMPLES)-1.)/float(NSAMPLES))*WBAR(j)
            do n=1,N_TH
               THBAR(j,n)=(1./float(NSAMPLES))*dble(CTH(0,0,j,n))
     &         +((float(NSAMPLES)-1.)/float(NSAMPLES))*THBAR(j,n)
            end do
         end do

! Integrat the instantaneous mean profile numerically at GY points
         UME=CU1(0,0,:)
      ELSE
         UME=0.d0
      END IF
      CALL INTEGRATE_Y_VAR(UME,UBULK,MPI_COMM_WORLD)
! Write out UBULK
      IF (RANK.EQ.0) write(*,*) 'UBULK: ',UBULK

! Save CUi
      do k=0,TNKZ
        do i=0,NXP-1 ! NKX
          do j=0,NY+1
            CR1(i,k,j)=CU1(i,k,j)
            CR2(i,k,j)=CU2(i,k,j)
            CR3(i,k,j)=CU3(i,k,j)
          end do
        end do
      end do

! Get the mean value of the velocities
      IF (RANKZ.EQ.0) THEN
         ume=dble(CU1(0,0,:))
         vme=dble(CU2(0,0,:))
         wme=dble(CU3(0,0,:)) 
         DO n=1,N_TH
            thme(:,n)=dble(CTH(0,0,:,n))
         END DO
      END IF
      CALL MPI_BCAST(ume,NY+2,MPI_DOUBLE_PRECISION,0,
     &     MPI_COMM_Z,ierror)
      CALL MPI_BCAST(vme,NY+2,MPI_DOUBLE_PRECISION,0,
     &     MPI_COMM_Z,ierror)
      CALL MPI_BCAST(wme,NY+2,MPI_DOUBLE_PRECISION,0,
     &     MPI_COMM_Z,ierror)
      IF (N_TH.GT.0) CALL MPI_BCAST(thme,(NY+2)*N_TH,
     &     MPI_DOUBLE_PRECISION,0,MPI_COMM_Z,ierror)

! Get the spectra
      DO J=1,NY; DO K=0,NZ+1; DO I=0,NXP
        phipp(I,K,J)=phipp(I,K,J)+CP(I,K,J)*CONJG(CP(I,K,J))
        phiuu(I,K,J)=phiuu(I,K,J)+CU1(I,K,J)*CONJG(CU1(I,K,J))
        CVC=0.5D0*(CU2(I,K,J)+CU2(I,K,J+1))
        phivv(I,K,J)=phivv(I,K,J)+CVC*CONJG(CVC)
        phiww(I,K,J)=phiww(I,K,J)+CU3(I,K,J)*CONJG(CU3(I,K,J))
        phiuv(I,K,J)=phiuv(I,K,J)+REAL(CVC*CONJG(CU1(I,K,J)))
      END DO; END DO; END DO

! PI

! PIX
      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    pixsub(I,K,J)=CONJG(CIKX(I)*CU1(I,K,J))
	    pix(I,K,J)=pix(I,K,J)+real(CP(I,K,J)*pixsub(I,K,J))
          END DO
        END DO
      END DO

! PIY
      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    piysub(I,K,J)=CONJG((CU2(I,K,J+1)-CU2(I,K,J))/DYF(J))
	    piy(I,K,J)=piy(I,K,J)+real(CP(I,K,J)*piysub(I,K,J))
          END DO
        END DO
      END DO

! 0.5d0*(((U2(I,K,J+1)-U2(I,K,J))/DYF(J)-((U2(I,K,J)-U2(I,K,J-1))/DYF(J-1))

! PIZ
      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    pizsub(I,K,J)=CONJG(CIKZ(K)*CU3(I,K,J))
	    piz(I,K,J)=piz(I,K,J)+real(CP(I,K,J)*pizsub(I,K,J))
          END DO
        END DO
      END DO

! Production
      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    CVC=0.5D0*(CU2(I,K,J)+CU2(I,K,J+1))
	    prod(I,K,J)=prod(I,K,J)+real(-CONJG(CU1(I,K,J))*CVC*DUDY(J))
          END DO
        END DO
      END DO

! Pressure transport
      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    CVC=0.5D0*(CU2(I,K,J-1)+CU2(I,K,J))
            CVC2=0.5D0*(CU2(I,K,J)+CU2(I,K,J+1))
	    prestrans(I,K,J)=prestrans(I,K,J)-real((CP(I,K,J)*CONJG(CVC2)-
     &                   CP(I,K,J-1)*CONJG(CVC))/DYF(J-1))
          END DO
        END DO
      END DO

! Dissipation

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    diffx(I,K,J)=diffx(I,K,J)-nu*(CIKX(I)*CU1(I,K,J)*CONJG(CIKX(I)*
     &                   CU1(I,K,J))+(0.5*(CU1(i,k,j+1)+CU1(i,k,j))-0.5*
     &                   (CU1(i,k,j)+CU1(i,k,j-1)))/DYF(j)*CONJG(0.5*
     &                   (CU1(i,k,j+1)+CU1(i,k,j))-0.5*(CU1(i,k,j)+
     &                   CU1(i,k,j-1)))/DYF(j)+(CIKZ(K)*CU1(I,K,J))*
     &                   CONJG(CIKZ(K)*CU1(I,K,J)))
          END DO
        END DO
      END DO

!      write(*,*) 'diffx(10,10,30) ',diffx(10,10,30)

!      write(*,*) 'diffx(1st) ',nu*(CIKX(5)*CU1(5,5,30)*CONJG(CIKX(5)*
!     &                   CU1(5,5,30)))

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    diffy(I,K,J)=diffy(I,K,J)-nu*(CIKX(I)*CU2(I,K,J)*CONJG(CIKX(I)*
     &                   CU2(I,K,J))+(0.5*(CU2(i,k,j+1)+CU2(i,k,j))-0.5*
     &                   (CU2(i,k,j)+CU2(i,k,j-1)))/DYF(j)*CONJG(0.5*
     &                   (CU2(i,k,j+1)+CU2(i,k,j))-0.5*(CU2(i,k,j)+
     &                   CU2(i,k,j-1)))/DYF(j)+(CIKZ(K)*CU2(I,K,J))*
     &                   CONJG(CIKZ(K)*CU2(I,K,J)))
          END DO
        END DO
      END DO

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    diffz(I,K,J)=diffz(I,K,J)-nu*(CIKX(I)*CU3(I,K,J)*CONJG(CIKX(I)*
     &                   CU3(I,K,J))+(0.5*(CU3(i,k,j+1)+CU3(i,k,j))-0.5*
     &                   (CU3(i,k,j)+CU3(i,k,j-1)))/DYF(j)*CONJG(0.5*
     &                   (CU3(i,k,j+1)+CU3(i,k,j))-0.5*(CU3(i,k,j)+
     &                   CU3(i,k,j-1)))/DYF(j)+(CIKZ(K)*CU3(I,K,J))*
     &                   CONJG(CIKZ(K)*CU3(I,K,J)))
          END DO
        END DO
      END DO

! Viscous transport

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    vistransx(I,K,J)=vistransx(I,K,J)+nu*((((0.5*CONJG(CU1(I,K,J+1))
     &                   *CU1(I,K,J+1))-(0.5*CONJG(CU1(I,K,J))*
     &                   CU1(I,K,J)))/DYF(J)-((0.5*CONJG(CU1(I,K,J))*
     &                   CU1(I,K,J))-(0.5*CONJG(CU1(I,K,J-1))*
     &                   CU1(I,K,J-1)))/DYF(J-1))/(0.5*(DYF(J)+DYF(J-1))
     &                   ))
          END DO
        END DO
      END DO

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    vistransy(I,K,J)=vistransy(I,K,J)+nu*((((0.5*CONJG(CU2(I,K,J+1))
     &                   *CU2(I,K,J+1))-(0.5*CONJG(CU2(I,K,J))*
     &                   CU2(I,K,J)))/DYF(J)-((0.5*CONJG(CU2(I,K,J))*
     &                   CU2(I,K,J))-(0.5*CONJG(CU2(I,K,J-1))*
     &                   CU2(I,K,J-1)))/DYF(J-1))/(0.5*(DYF(J)+DYF(J-1))
     &                   ))
          END DO
        END DO
      END DO

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    vistransz(I,K,J)=vistransz(I,K,J)+nu*((((0.5*CONJG(CU3(I,K,J+1))
     &                   *CU3(I,K,J+1))-(0.5*CONJG(CU3(I,K,J))*
     &                   CU3(I,K,J)))/DYF(J)-((0.5*CONJG(CU3(I,K,J))*
     &                   CU3(I,K,J))-(0.5*CONJG(CU3(I,K,J-1))*
     &                   CU3(I,K,J-1)))/DYF(J-1))/(0.5*(DYF(J)+DYF(J-1))
     &                   ))
          END DO
        END DO
      END DO

! SGS 
      
      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CSijc(I,K,J,1)=CIKX(I)*CU1(I,K,J)
            CSijc(I,K,J,2)=(CU2(I,K,J+1)-CU2(I,K,J))/DYF(j) 
            CSijc(I,K,J,3)=CIKZ(K)*CU3(I,K,J)  
            CSijc(I,K,J,5)=0.5d0*(CIKZ(K)*CU1(I,K,J)+CIKX(I)*CU3(I,K,J))
          END DO
        END DO
      END DO
      DO J=1,NY+1
        DO K=0,TNKZ
          DO I=0,NXP-1
            CSijc(I,K,J,4)=0.5d0*( (CU1(I,K,J)-CU1(I,K,J-1))/DY(j)
     &                          +CIKX(I)*CU2(I,K,J) ) 
            CSijc(I,K,J,6)=0.5d0*( CIKZ(K)*CU2(I,K,J)
     &                         +(CU3(I,K,J)-CU3(I,K,J-1))/DY(j) )
          END DO
        END DO
      END DO  

      DO ij=1,6
        CALL HALO_ARRAY_COMPLEX(CSijc(:,:,:,ij))
      END DO

! Convert rate of strain tensor to physical space
      do ij=1,6
        call FFT_XZ_TO_PHYSICAL(CSijc(0,0,0,ij),Sijc(0,0,0,ij),0,NY+1)
      end do 

      DO J=1,NY
        DO K=0,TNKZ
          DO I=0,NXP-1
            CAijc(I,K,J,1)=CIKX(I)*CU1(I,K,J)
            CAijc(I,K,J,2)=(CU2(I,K,J+1)-CU2(I,K,J))/DYF(j) 
            CAijc(I,K,J,3)=CIKZ(K)*CU3(I,K,J)  
            CAijc(I,K,J,5)=CIKX(I)*CU3(I,K,J)
            CAijc(I,K,J,7)=CIKZ(K)*CU1(I,K,J)
          END DO
        END DO
      END DO
      DO J=1,NY+1
        DO K=0,TNKZ
          DO I=0,NXP-1
            CAijc(I,K,J,4)=CIKX(I)*CU2(I,K,J) 
            CAijc(I,K,J,6)=CIKZ(K)*CU2(I,K,J)
            CAijc(I,K,J,8)=(CU3(I,K,J)-CU3(I,K,J-1))/DY(j)  
            CAijc(I,K,J,9)=(CU1(I,K,J)-CU1(I,K,J-1))/DY(j)                
          END DO
        END DO
      END DO  

      DO ij=1,9
        CALL HALO_ARRAY_COMPLEX(CAijc(:,:,:,ij))
      END DO

! Convert rate of strain tensor to physical space
      do ij=1,9
        call FFT_XZ_TO_PHYSICAL(CAijc(0,0,0,ij),Aijc(0,0,0,ij),0,NY+1)
      end do 

!!!!!!!!!!!!!!

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
            Sijc(I,K,J,4)=
     & (Sijc(I,K,J,4)-0.5*(U1(I,K,J)-U1(I,K,J-1))/DY(j))
            Sijc(I,K,J,6)=
     & (Sijc(I,K,J,6)-0.5*(U3(I,K,J)-U3(I,K,J-1))/DY(j))
          
          END DO
        END DO
      END DO

! Convert |S|*S_ij to Fourier space
      do ij=1,6   
      CALL FFT_XZ_TO_FOURIER(Sijc(0,0,0,ij),CSijc(0,0,0,ij),0,NY+1)      
      end do

! Define filter width considering dealiasing.
      del_xc=(DX(1)*3./2.)**2.d0 
      del_zc=(DZ(1)*3./2.)**2.d0
      
     
! Extend to ghost cells 

!!!!!!!!!!!!!!!!!!!!!!!!!!VREMAN PART

       DO J=1,NY
       DO K=0,NZP-1
       DO I=0,NXM          
         beta11c(I,K,J)=
     &del_xc*(aijc(i,k,j,1)**2.0d0)+
     &(DYF(j)**2.d0)*(0.5d0*(aijc(i,k,j,9)+aijc(i,k,j+1,9)))
     &**2.0d0+
     &del_zc*(aijc(i,k,j,7)**2.0d0)

            beta12c(I,K,J)=
     &del_xc*aijc(i,k,j,1)*(0.5d0*(aijc(i,k,j,4)+aijc(i,k,j+1,4)))+
     &(dYF(j)**2.d0)*0.5d0*(aijc(i,k,j,9)+aijc(i,k,j+1,9))
     &*aijc(i,k,j,2)+
     &del_zc*aijc(i,k,j,7)*(0.5d0*(aijc(i,k,j,6)+aijc(i,k,j+1,6)))

            beta13c(I,K,J)=
     &del_xc*aijc(i,k,j,1)*aijc(i,k,j,5)+
     &(dYF(j)**2.d0)*(0.5d0*(aijc(i,k,j,2)+aijc(i,k,j+1,2)))*
     &         (0.5d0*(aijc(i,k,j,8)+aijc(i,k,j+1,8)))+
     &del_zc*aijc(i,k,j,7)*aijc(i,k,j,3)

        beta23c(I,K,J)=
     &del_xc*(0.5d0*(aijc(i,k,j,4)+aijc(i,k,j+1,4)))*aijc(i,k,j,5)+
     &(dYF(j)**2.d0)*aijc(i,k,j,2)*0.5*(aijc(i,k,j,8)
     &+aijc(i,k,j+1,8))+
     &+del_zc*0.5d0*(aijc(i,k,j,6)+aijc(i,k,j+1,6))*aijc(i,k,j,3)

        beta22c(I,K,J)=
     &del_xc*(0.5d0*(aijc(i,k,j,4)+aijc(i,k,j+1,4)))**2.0d0+
     &(dYF(j)**2.d0)*(aijc(i,k,j,2)**2.0d0)+
     &del_zc*(0.5d0*(aijc(i,k,j,6)+aijc(i,k,j+1,6))**2.0d0)

        beta33c(I,K,J)=
     &del_xc*aijc(i,k,j,5)**2.0d0+
     &(dYf(j)**2.d0)*(0.5d0*(aijc(i,k,j,8)+aijc(i,k,j+1,8))**2.d0)+
     &del_zc*aijc(i,k,j,3)**2.0d0

       END DO
       END DO
       END DO

       
       DO K=0,NZP-1
       DO I=0,NXM 
       DO J=1,NY
         betaVc(I,K,J)=
     &   dmax1(0.d0,
     &   beta11c(i,k,j)*beta22c(i,k,j)
     & -(beta12c(i,k,j)**2.0d0)
     & +beta11c(i,k,j)*beta33c(i,k,j)
     & -(beta13c(i,k,j)**2.0d0)
     & +beta22c(i,k,j)*beta33c(i,k,j)
     & -(beta23c(i,k,j)**2.0d0)             
     & )
     
     
        facc(i,k,j)=max(0.d0,sqrt(
     &  betaVc(i,k,j)/(
     &   aijc(i,k,j,1)*aijc(i,k,j,1)+
     &  aijc(i,k,j,2)*aijc(i,k,j,2)+    
     &  aijc(i,k,j,3)*aijc(i,k,j,3)+ 
     &  aijc(i,k,j,5)*aijc(i,k,j,5)+
     &  aijc(i,k,j,7)*aijc(i,k,j,7)+
     &  (0.5d0*(aijc(i,k,j,4)+aijc(i,k,j+1,4))**2.d0)+
     &  (0.5d0*(aijc(i,k,j,6)+aijc(i,k,j+1,6))**2.d0)+
     &  (0.5d0*(aijc(i,k,j,8)+aijc(i,k,j+1,8))**2.d0)+
     &  (0.5d0*(aijc(i,k,j,9)+aijc(i,k,j+1,9))**2.d0)
     &  )))


       END DO
       END DO
       END DO   
       
       DO K=0,NZP-1
         DO I=0,NXM
          DO J=1,NY+1
             TempYc(i,k,j)=
     &       (facc(I,K,J)*DYF(j-1)+facc(I,K,J-1)*DYF(j))
     &        /(2.d0*DY(j))

             NU_Tc(i,k,j)=C_Vreman*
     &       TempYc(i,k,j)

         END DO
!        NU_Tc(i,k,1)=0.d0
!        NU_Tc(i,k,NY)=0
        END DO
       END DO

       CALL HALO_ARRAY_REAL(NU_Tc)

       DO K=0,NZP-1
         DO I=0,NXM
          DO J=1,NY
             NU_T2c(i,k,j)=-2.d0*
     &       (NU_Tc(I,K,J)*DY(j+1)+NU_Tc(I,K,J+1)*DY(j))
     &        /(2.d0*DYF(j))

         END DO
!        NU_T2c(i,k,1)=0.d0
        IF(RANKY.EQ.NPROCY-1) then
         NU_T2c(i,k,NY+1)=0
        ELSEIF (RANKY.EQ.0) then
         NU_T2c(i,k,0)=0.d0
        END IF
!       NU_T2c(i,k,NY)=0
       END DO
      END DO

      CALL HALO_ARRAY_REAL(NU_T2c)

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
              CSijc(I,K,J,ij)=0.d0
            END DO
          END DO
        END DO
      END DO

      call FFT_XZ_TO_FOURIER(U1,CU1,0,NY+1)
      call FFT_XZ_TO_FOURIER(U2,CU2,0,NY+1)
      call FFT_XZ_TO_FOURIER(U3,CU3,0,NY+1)
           

!!!!!!!!!!!!!!!!!!!!!!!!!!

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    sgsx(I,K,J)=sgsx(I,K,J)-real(CONJG(CU1(I,K,J))
     &                   *(CIKX(I)*CSijc(I,K,J,1)*nu_t2c(I,K,J)+(0.5
     &                   *(CSijc(i,k,j+1,4)*nu_t2c(I,K,J+1)
     &                   +CSijc(i,k,j,4)*nu_t2c(I,K,J))-0.5
     &                   *(CSijc(i,k,j,4)*nu_t2c(I,K,J)+
     &                   CSijc(i,k,j-1,4)*nu_t2c(I,K,J-1)))
     &                   /DYF(j)+CIKZ(K)*CSijc(I,K,J,5)
     &                   *nu_t2c(I,K,J)))
          END DO
        END DO
      END DO

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    sgsy(I,K,J)=sgsy(I,K,J)-real(CONJG(CU2(I,K,J))
     &                   *(CIKX(I)*CSijc(I,K,J,4)*nu_t2c(I,K,J)+(0.5
     &                   *(CSijc(i,k,j+1,2)*nu_t2c(I,K,J+1)
     &                   +CSijc(i,k,j,2)*nu_t2c(I,K,J))-0.5
     &                   *(CSijc(i,k,j,2)*nu_t2c(I,K,J)+
     &                   CSijc(i,k,j-1,2)*nu_t2c(I,K,J-1)))
     &                   /DYF(j)+CIKZ(K)*CSijc(I,K,J,6)
     &                   *nu_t2c(I,K,J)))
          END DO
        END DO
      END DO

      DO j=1,NY
        DO k=0,TNKZ
          DO i=0,NXP-1 !NKX
	    sgsz(I,K,J)=sgsz(I,K,J)-real(CONJG(CU3(I,K,J))
     &                   *(CIKX(I)*CSijc(I,K,J,5)*nu_t2c(I,K,J)+(0.5
     &                   *(CSijc(i,k,j+1,6)*nu_t2c(I,K,J+1)
     &                   +CSijc(i,k,j,6)*nu_t2c(I,K,J))-0.5
     &                   *(CSijc(i,k,j,6)*nu_t2c(I,K,J)+
     &                   CSijc(i,k,j-1,6)*nu_t2c(I,K,J-1)))
     &                   /DYF(j)+CIKZ(K)*CSijc(I,K,J,3)
     &                   *nu_t2c(I,K,J)))
          END DO
        END DO
      END DO


! Turbulent transport

      DO j=1,NY
        DO k=0,NKZ
          DO i=0,NXP-1 !NKX
            DO l=-NKZ,NKZ
              CALL CONVERTERK(K,tt)
              uu=TT
              m=uu-l
              CALL CONVERTER(l,tt)
              LT=TT
              CALL CONVERTER(m,tt)
              MT=TT
!              write(*,*) ' ( , )',k,LT,MT
              IF(lt.gt.NKZ .and. lt.gt.NKZ) THEN
              ELSE
               IF(lt.gt.NKZ .or. mt.gt.NKZ) THEN
                sumx1(I,K,J)=sumx1(I,K,J)+CU1(I,lt,J)*CU1(I,mt,J)
                sumy1(I,K,J)=sumy1(I,K,J)+CU1(I,lt,J)*CU2(I,mt,J)
                sumz1(I,K,J)=sumz1(I,K,J)+CU1(I,lt,J)*CU3(I,mt,J)
                sumx2(I,K,J)=sumx2(I,K,J)+CU2(I,lt,J)*CU1(I,mt,J)
                sumy2(I,K,J)=sumy2(I,K,J)+CU2(I,lt,J)*CU2(I,mt,J)
                sumz2(I,K,J)=sumz2(I,K,J)+CU2(I,lt,J)*CU3(I,mt,J)
                sumx3(I,K,J)=sumx3(I,K,J)+CU3(I,lt,J)*CU1(I,mt,J)
                sumy3(I,K,J)=sumy3(I,K,J)+CU3(I,lt,J)*CU2(I,mt,J)
                sumz3(I,K,J)=sumz3(I,K,J)+CU3(I,lt,J)*CU3(I,mt,J)
!
                sumx1(I,K,J)=sumx1(I,K,J)+CU1(I,mt,J)*CU1(I,lt,J)
                sumy1(I,K,J)=sumy1(I,K,J)+CU1(I,mt,J)*CU2(I,lt,J)
                sumz1(I,K,J)=sumz1(I,K,J)+CU1(I,mt,J)*CU3(I,lt,J)
                sumx2(I,K,J)=sumx2(I,K,J)+CU2(I,mt,J)*CU1(I,lt,J)
                sumy2(I,K,J)=sumy2(I,K,J)+CU2(I,mt,J)*CU2(I,lt,J)
                sumz2(I,K,J)=sumz2(I,K,J)+CU2(I,mt,J)*CU3(I,lt,J)
                sumx3(I,K,J)=sumx3(I,K,J)+CU3(I,mt,J)*CU1(I,lt,J)
                sumy3(I,K,J)=sumy3(I,K,J)+CU3(I,mt,J)*CU2(I,lt,J)
                sumz3(I,K,J)=sumz3(I,K,J)+CU3(I,mt,J)*CU3(I,lt,J)
                ELSE
!                write(*,*) ' ( , )',k,LT,MT
                 sumx1(I,K,J)=sumx1(I,K,J)+CU1(I,lt,J)*CU1(I,mt,J)
                 sumy1(I,K,J)=sumy1(I,K,J)+CU1(I,lt,J)*CU2(I,mt,J)
                 sumz1(I,K,J)=sumz1(I,K,J)+CU1(I,lt,J)*CU3(I,mt,J)
                 sumx2(I,K,J)=sumx2(I,K,J)+CU2(I,lt,J)*CU1(I,mt,J)
                 sumy2(I,K,J)=sumy2(I,K,J)+CU2(I,lt,J)*CU2(I,mt,J)
                 sumz2(I,K,J)=sumz2(I,K,J)+CU2(I,lt,J)*CU3(I,mt,J)
                 sumx3(I,K,J)=sumx3(I,K,J)+CU3(I,lt,J)*CU1(I,mt,J)
                 sumy3(I,K,J)=sumy3(I,K,J)+CU3(I,lt,J)*CU2(I,mt,J)
                 sumz3(I,K,J)=sumz3(I,K,J)+CU3(I,lt,J)*CU3(I,mt,J)
                END IF
              END IF
            END DO
          END DO
        END DO
      END DO

      DO j=1,NY
        DO k=0,NKZ
          DO i=0,NXP-1 
            turbtransx(I,K,J)=turbtransx(I,K,J)-real(CONJG(CU1(I,K,J))*
     &                    (CIKX(I)*sumx1(I,K,J)+(sumy1(I,K,J+1)-
     &                    sumy1(I,K,J))/DYF(J)+CIKZ(K)*sumz1(I,K,J)))

            turbtransy(I,K,J)=turbtransy(I,K,J)-real(CONJG(CU2(I,K,J))*
     &                    (CIKX(I)*sumx2(I,K,J)+(sumy2(I,K,J+1)-
     &                    sumy2(I,K,J))/DYF(J)+CIKZ(K)*sumz2(I,K,J)))

            turbtransz(I,K,J)=turbtransz(I,K,J)-real(CONJG(CU3(I,K,J))*
     &                    (CIKX(I)*sumx3(I,K,J)+(sumy3(I,K,J+1)-
     &                    sumy3(I,K,J))/DYF(J)+CIKZ(K)*sumz3(I,K,J)))
          END DO
        END DO
      END DO


      DO j=1,NY
        DO k=NKZ+1,TNKZ
          DO i=0,NXP-1 
            turbtransx(I,K,J)=turbtransx(I,K,J)+turbtransx(I,TNKZ-k+1,J)
            turbtransy(I,K,J)=turbtransy(I,K,J)+turbtransy(I,TNKZ-k+1,J)
            turbtransz(I,K,J)=turbtransz(I,K,J)+turbtransz(I,TNKZ-k+1,J)
          END DO
        END DO
      END DO


! Convert to physical space
      call fft_xz_to_physical(CU1,U1,0,NY+1)
      call fft_xz_to_physical(CU2,U2,0,NY+1)
      call fft_xz_to_physical(CU3,U3,0,NY+1)

! Get the turbulent kinetic energy at each level 
      do j=1,NY
        urms(j)=0.
        vrms(j)=0.
        wrms(j)=0.
      do k=0,NZP-1
      do i=0,NXM 
        urms(j)=urms(j)+(U1(i,k,j)-ume(j))**2.
        vrms(j)=vrms(j)+0.5*((U2(i,k,j  )-vme(j  ))**2. +
     &                       (U2(i,k,j+1)-vme(j+1))**2. )
        wrms(j)=wrms(j)+(U3(i,k,j)-wme(j))**2.
      end do
      end do
      end do

      call mpi_allreduce(mpi_in_place,urms,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      call mpi_allreduce(mpi_in_place,vrms,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      call mpi_allreduce(mpi_in_place,wrms,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)

      do j=1,NY
        urms(j)=sqrt(urms(j)/(float(NZ)*float(NX)))
        vrms(j)=sqrt(vrms(j)/(float(NZ)*float(NX)))
        wrms(j)=sqrt(wrms(j)/(float(NZ)*float(NX)))
      end do 

      ! Get the bulk rms value
      CALL INTEGRATE_Y_VAR(urms,urms_b,MPI_COMM_Y)
      CALL INTEGRATE_Y_VAR(vrms,vrms_b,MPI_COMM_Y)
      CALL INTEGRATE_Y_VAR(wrms,wrms_b,MPI_COMM_Y)

! Compute the Reynolds stress and mean velocity gradient
      do j=1,NY
        uv(j)=0. 
        uw(j)=0.
        wv(j)=0.
      do k=0,NZP-1
      do i=0,NXM
        uv(j)=uv(j)+(U1(i,k,j)-ume(j))
     +    *(0.5*(U2(i,k,j)+U2(i,k,j+1))
     &    -0.5*(vme(j)+vme(j+1)))
        wv(j)=wv(j)+(U3(i,k,j)-wme(j))
     +    *(0.5*(U2(i,k,j)+U2(i,k,j+1))
     &    -0.5*(vme(j)+vme(j+1)))
        uw(j)=uw(j)+(U1(i,k,j)-ume(j))
     +    *(U3(i,k,j)-wme(j))
      end do
      end do
      end do

      call mpi_allreduce(mpi_in_place,uv,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      call mpi_allreduce(mpi_in_place,uw,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      call mpi_allreduce(mpi_in_place,wv,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      
      do j=1,NY
        uv(j)=uv(j)/(float(NZ)*float(NX))
        uw(j)=uw(j)/(float(NZ)*float(NX))
        wv(j)=wv(j)/(float(NZ)*float(NX))
      end do

! Calculate the correlation with the reference point at the channel
! centre (corry)
! Get the xz-plane data at channel centre
      uxz(0:NXM,0:NZP-1)=U1(0:NXM,0:NZP-1,NY)-ume(NY)
      vxz(0:NXM,0:NZP-1)=U2(0:NXM,0:NZP-1,NY)
      wxz(0:NXM,0:NZP-1)=U3(0:NXM,0:NZP-1,NY)

!      IF(RANKZ.EQ.0) WRITE(*,*) 'ranky,rankz==',ranky,rankz,uxz(3,3)
      CALL MPI_BCAST(uxz,(NXM+1)*NZP,MPI_DOUBLE_PRECISION,NPROCY/2-1,
     &     MPI_COMM_Y,ierror)
      CALL MPI_BCAST(vxz,(NXM+1)*NZP,MPI_DOUBLE_PRECISION,NPROCY/2-1,
     &     MPI_COMM_Y,ierror)
      CALL MPI_BCAST(wxz,(NXM+1)*NZP,MPI_DOUBLE_PRECISION,NPROCY/2-1,
     &     MPI_COMM_Y,ierror)

      Cuu=0.0D0; Cvv=0.0D0; Cww=0.0D0
      Cuv=0.0D0; Cuw=0.0D0; Cwv=0.0D0
      Do J=1,NY
        DO K=0,NZP-1; DO I=0,NXM
          Cuu(J)=Cuu(J)+uxz(I,K)*(U1(I,K,J)-ume(J))
          Cvv(J)=Cvv(J)+vxz(I,K)*U2(I,K,J)
          Cww(J)=Cww(J)+wxz(I,K)*U3(I,K,J)
!          Cuv(J)=Cuv(J)+(uxz(I,K)-ume(J))*(0.5D0*(U2(I,K,J)+U2(I,K,J+1)))
        END DO; END DO
      END DO
      CALL mpi_allreduce(mpi_in_place,Cuu,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      CALL mpi_allreduce(mpi_in_place,Cvv,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      CALL mpi_allreduce(mpi_in_place,Cww,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)

      Cuu(:)=Cuu(:)/(FLOAT(NZ)*FLOAT(NX))
      Cvv(:)=Cvv(:)/(FLOAT(NZ)*FLOAT(NX))
      Cww(:)=Cww(:)/(FLOAT(NZ)*FLOAT(NX))
              
! Get the y-derivative of the mean velocity at GYF points
!      do j=1,NY
!        dudy(j)=dble(CR1(0,0,j+1)-CR1(0,0,j-1))/(2.*DYF(j))
!        dwdy(j)=dble(CR3(0,0,j+1)-CR3(0,0,j-1))/(2.*DYF(j))
!      end do
! Get the y-derivative of the mean velocity at GY points
      do j=1,NY
!        dudy(j)=(ume(j)-ume(j-1))/(GYF(j)-GYF(j-1))
!        dwdy(j)=(wme(j)-wme(j-1))/(GYF(j)-GYF(j-1))
         dudy(j)=(DY(J)**2*UME(J+1)-DY(J+1)**2*UME(J-1)-
     &           (DY(J)**2-DY(J+1)**2)*UME(J))
     &           /(DY(J)*DY(J+1)*(DY(J)+DY(J+1)))
         dwdy(j)=(DY(J)**2*WME(J+1)-DY(J+1)**2*WME(J-1)-
     &           (DY(J)**2-DY(J+1)**2)*WME(J))
     &           /(DY(J)*DY(J+1)*(DY(J)+DY(J+1)))
      end do

! Calculate the mean square shear
      do j=1,NY
        shear(j)=0.d0
        do k=0,NZP-1
          do i=0,NXM
            shear(j)=shear(j)
     &            +((U1(i,k,j+1)-U1(i,k,j-1))/(2.d0*DYF(j)))**2.d0
     &            +((U3(i,k,j+1)-U3(i,k,j-1))/(2.d0*DYF(j)))**2.d0
          end do
        end do
      end do
      call mpi_allreduce(mpi_in_place,shear,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
        shear(j)=shear(j)/dble(NX*NZ)
      end do

! Write out the bulk rms velocity
      if (RANK.eq.0) then
         write(*,*) '<U_rms>: ',urms_b
         write(*,*) '<V_rms>: ',vrms_b
         write(*,*) '<W_rms>: ',wrms_b
      end if

! Get the rms vorticity
! First, get the x-component in fourier space
      do j=1,NY
      do k=0,TNKZ
      do i=0,NXP-1 !NKX
        CS1(i,k,j)=(CR3(i,k,j+1)-CR3(i,k,j-1))/(2.d0*DYF(j))
     &            -CIKZ(K)*0.5d0*(CR2(i,k,j+1)+CR2(i,k,j))
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
! Get the rms value
      do j=1,NY
      omega_x(j)=0.d0
      do k=0,NZP-1
      do i=0,NXM
        omega_x(j)=omega_x(j)+S1(i,k,j)**2.d0
      end do
      end do
      end do
      call mpi_allreduce(mpi_in_place,omega_x,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
      omega_x(j)=sqrt(omega_x(j)/(dble(NX)*dble(NZ)))
      end do

! Now, get the y-component in fourier space
      do j=1,NY
      do k=0,TNKZ
      do i=0,NXP-1 !NKX
        CS1(i,k,j)=CIKZ(k)*CR1(i,k,j)-CIKX(i)*CR3(i,k,j)
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
! Get the rms value
      do j=1,NY
      omega_y(j)=0.d0
      do k=0,NZP-1
      do i=0,NXM
        omega_y(j)=omega_y(j)+S1(i,k,j)**2.d0
      end do
      end do
      end do
      call mpi_allreduce(mpi_in_place,omega_y,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
      omega_y(j)=sqrt(omega_y(j)/(dble(NX)*dble(NZ)))
      end do

! Now, get the y-component in fourier space
      do j=1,NY
      do k=0,TNKZ
      do i=0,NXP-1 ! NKX
        CS1(i,k,j)=CIKX(i)*0.5d0*(CR2(i,k,j+1)+CR2(i,k,j))
     &             -(CR1(i,k,j+1)-CR1(i,k,j-1))/(2.d0*DYF(j))
      end do
      end do
      ! if (RANKZ.eq.0) CS1(0,0,j)=CS1(0,0,j)+dudy(j)
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
! Get the rms value
      do j=1,NY
      omega_z(j)=0.d0
      do k=0,NZP-1
      do i=0,NXM
        omega_z(j)=omega_z(j)+S1(i,k,j)**2.d0
      end do
      end do
      end do
      call mpi_allreduce(mpi_in_place,omega_z,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
      omega_z(j)=sqrt(omega_z(j)/(dble(NX)*dble(NZ)))
      end do


! Write out the mean statistics at each time
#ifdef HDF5
      FNAME='mean.h5'
      GNAME='statistics'
      IF(RANKZ.EQ.0) THEN
        DO J=1,NY
          STAT(J,1)=REAL(J)
        END DO
        STAT(1:NY,2)=GYF(1:NY)
        STAT(1:NY,3)=UME(1:NY)
        STAT(1:NY,4)=VME(1:NY)
        STAT(1:NY,5)=WME(1:NY)        
        STAT(1:NY,6)=urms(1:NY)
        STAT(1:NY,7)=vrms(1:NY)
        STAT(1:NY,8)=wrms(1:NY)
        STAT(1:NY,9)=uv(1:NY)
        STAT(1:NY,10)=uw(1:NY)
        STAT(1:NY,11)=wv(1:NY)
        STAT(1:NY,12)=dudy(1:NY)
        STAT(1:NY,13)=dwdy(1:NY)
        STAT(1:NY,14)=dble(cp(0,0,1:NY))
        STAT(1:NY,15)=shear(1:NY)
        STAT(1:NY,16)=omega_x(1:NY)
        STAT(1:NY,17)=omega_y(1:NY)
        STAT(1:NY,18)=omega_z(1:NY)
!
        CALL WriteHDF5_STATISTICS(FNAME,GNAME,STAT,18)
      END IF
! calculate the correlction in y direction on fly
      FNAME='corry.h5'
      GNAME='statistics'
      IF(RANKZ.EQ.0) THEN
        DO J=1,NY
          STAT(J,1)=REAL(J)
        END DO
        STAT(1:NY,2)=GYF(1:NY)
        STAT(1:NY,3)=Cuu(1:NY)
        STAT(1:NY,4)=Cvv(1:NY)
        STAT(1:NY,5)=Cww(1:NY)
        STAT(1:NY,6)=Cuv(1:NY)
        STAT(1:NY,7)=Cuw(1:NY)
        STAT(1:NY,8)=Cwv(1:NY)
!
        CALL WriteHDF5_STATISTICS(FNAME,GNAME,STAT(:,1:8),8)
      END IF
#else
      IF (RANKZ.EQ.0) THEN
      IF (USE_MPI) THEN
        FNAME='mean'//trim(MPI_IO_NUM)//'.txt'
      ELSE
        FNAME='mean.txt'
      END IF
      open(40,file=FNAME,form='formatted',status='unknown')
      write(40,*) TIME_STEP,TIME,DELTA_T
      write(40,*) UBULK
      do j=1,NY
        write(40,401) j,GYF(J),ume(j)
     +      ,0.5*(vme(j+1)+vme(j))
     +      ,wme(j),urms(j),vrms(j),wrms(j)
     +      ,uv(j),uw(j),wv(j),dudy(j),dwdy(j),dble(cp(0,0,j)),shear(j)
     &      ,omega_x(j),omega_y(j),omega_z(j)
      end do
      END IF
#endif
401   format(I3,' ',17(F20.9,' '))

C Call netcdf
!      CALL NETCDF_WRITE_STATS_CHAN

! Do over the number of passive scalars
      do n=1,N_TH

! Save CTH
      do k=0,TNKZ
        do i=0,NXP-1 ! NKX
          do j=0,NY+1
            CRTH(i,k,j,n)=CTH(i,k,j,n)
          end do
        end do
      end do

! Compute the scalar gradient and store in CRi
      do j=1,NY
        do k=0,TNKZ
          do i=0,NXP-1 ! NKX
! Store gradients of TH(:,:,:,n) (if it is used) in CRi
          CR1(i,k,j)=CIKX(i)*CTH(i,k,j,n)
          CR2(i,k,j)=(CTH(i,k,j+1,n)-CTH(i,k,j-1,n))/(GYF(j+1)-GYF(j-1))
          CR3(i,k,j)=CIKZ(k)*CTH(i,k,j,n)
          end do
        end do
      end do
! Convert gradients to physical space
      CALL FFT_XZ_TO_PHYSICAL(CR1,R1,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CR2,R2,0,NY+1)
      CALL FFT_XZ_TO_PHYSICAL(CR3,R3,0,NY+1)


! Convert to physical space
      call fft_xz_to_physical(CTH(0,0,0,n),TH(0,0,0,n),0,NY+1)
      do j=1,NY
        thrms(j,n)=0.
      do k=0,NZP-1
      do i=0,NXM
        thrms(j,n)=thrms(j,n)+(abs(TH(i,k,j,n)-thme(j,n)))**2.
      end do
      end do
      end do
      call mpi_allreduce(mpi_in_place,thrms(0,n),(NY+2),
     &     MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
        thrms(j,n)=sqrt(thrms(j,n)/(float(NZ)*float(NX)))
      end do
! Compute the Reynolds stress and mean velocity gradient
      do j=1,NY
        thv(j,n)=0.
      do k=0,NZP-1
      do i=0,NXM
       thv(j,n)=thv(j,n)+(TH(i,k,j,n)-thme(j,n))
     +    *(0.5*(U2(i,k,j)+U2(i,k,j+1))
     &      -0.5*(vme(j)+vme(j+1)))
      end do
      end do
      end do
      call mpi_allreduce(mpi_in_place,thv(0,n),(NY+2)*N_TH,
     &     MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
      thv(j,n)=thv(j,n)/(float(NZ)*float(NX))
      end do

! Get the y-derivative of the mean scalar at GYF points
      do j=1,NY
        dthdy(j,n)=(thme(j+1,n)-thme(j-1,n))/(2.*DYF(j))
      end do

! Compute the potential energy dissipation, grad(TH) \cdot grad(TH)
      do j=1,NY
        pe_diss(j,n)=0.d0
        do k=0,NZP-1
          do i=0,NXM
            pe_diss(j,n)=pe_diss(j,n)
     &          +R1(i,k,j)**2.d0+R2(i,k,j)**2.d0+R3(i,k,j)**2.d0
          end do
        end do
      end do
      call mpi_allreduce(mpi_in_place,pe_diss(0,n),(NY+2),
     &       MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_Z,ierror)
      do j=1,NY
        pe_diss(j,n)=pe_diss(j,n)/dble(NX*NZ)
      end do
 

      if (n.eq.1 .and. MOVIE) then
         FNAME='movie.h5'

         IF (RANKZ.EQ.RANKZMOVIE) THEN
            do I=0,NXM
            do J=1,NY
               varxy(i,j)=TH(i,NzMovie,j,n)
            end do
            end do
            GNAME='th_xy'
            call writeHDF5_xyplane(FNAME,GNAME,varxy)
         END IF
         
         IF (RANKY.EQ.RANKYMOVIE) THEN
            do I=0,NXM
            do J=0,NZP-1
               varxz(i,j)=TH(i,j,NyMovie,n)
            end do
            end do
            GNAME='th_xz'
            call writeHDF5_xzplane(FNAME,GNAME,varxz)
         END IF


         do I=0,NZP-1
         do J=1,NY
            varzy(i,j)=TH(NxMovie,i,j,n)
         end do
         end do
         GNAME='th_zy'
         call writeHDF5_zyplane(FNAME,GNAME,varzy)
      END IF

c$$$        if (n.eq.1) then
c$$$        open(88,file='movie_th.txt',status='unknown',form='formatted')
c$$$        do I=0,NXM
c$$$        do J=1,NY
c$$$          write(88,*) TH(I,0,J,n)
c$$$        end do
c$$$        end do
c$$$
c$$$! This file will contain a single plane and is used in conjunction with
c$$$! the matlab script 'realtime_movie' to visualize data during
c$$$! simulation
c$$$        open (76,file='temp.txt',status='unknown',form='formatted')
c$$$        do J=1,NY
c$$$          write(76,*) gyf(j)
c$$$        end do
c$$$        do I=0,NXM
c$$$        do J=1,NY
c$$$          write(76,*) TH(I,0,J,n)
c$$$        end do
c$$$        end do
c$$$        close (76)
c$$$        CALL SYSTEM('mv temp.txt ../post_process/matlab/latest_slice.txt
c$$$     & ')
c$$$        end if

! Convert back to Fourier space
      call FFT_XZ_TO_FOURIER(TH(0,0,0,n),CTH(0,0,0,n),0,NY+1)

! End do over number of passive scalars, n
      end do


! Write out the mean statistics at each time
#ifdef HDF5

#else
      IF (RANKZ.EQ.0) THEN
      IF (USE_MPI) THEN
        FNAME='mean_th'//trim(MPI_IO_NUM)//'.txt'
      ELSE
        FNAME='mean_th.txt'
      END IF
      open(41,file=FNAME,form='formatted',status='unknown')
      write(41,*) TIME_STEP,TIME,DELTA_T
      write(41,*) UBULK
      do n=1,N_TH 
      do j=1,NY
        write(41,402) j,GYF(J),thme(j,n)
     +      ,dthdy(j,n),thrms(j,n),thv(j,n),pe_diss(j,n)
      end do
      end do
      END IF
#endif
402   format(I3,' ',6(F20.9,' '))

      IF (RANK.EQ.0) 
     &     write(*,*) 'VERBOSITY: ',VERBOSITY
      if (VERBOSITY.gt.4) then 
      IF (RANK.EQ.0) 
     &        write(*,*) 'Outputting info for gnuplot...'
      open (unit=10, file="solution")
      do i=2,NXM
        do j=2,NYM
          write (10,*) i, j, U1(i,0,j)
        end do
        write (10,*) ""
      end do
      close (10)
      call system ('gnuplot <gnuplot.in') 
      end if

      if (MOVIE) then

         FNAME='movie.h5'

         IF (RANKZ.EQ.RANKZMOVIE) THEN
            do I=0,NXM
            do J=1,NY
               varxy(i,j)=U1(i,NzMovie,j)
            end do
            end do
            GNAME='u_xy'
            call writeHDF5_xyplane(FNAME,GNAME,varxy)

            do I=0,NXM
            do J=1,NY
               varxy(i,j)=0.5*(U2(i,NzMovie,j)+U2(i,NzMovie,j+1))
            end do
            end do
            GNAME='v_xy'
            call writeHDF5_xyplane(FNAME,GNAME,varxy)

            do I=0,NXM
            do J=1,NY
               varxy(i,j)=U3(i,NzMovie,j)
            end do
            end do
            GNAME='w_xy'
            call writeHDF5_xyplane(FNAME,GNAME,varxy)
         END IF

         IF (RANKY.EQ.RANKYMOVIE) THEN
            do I=0,NXM
            do J=0,NZP-1
               varxz(i,j)=U1(i,j,NyMovie)
            end do
            end do
            GNAME='u_xz'
            call writeHDF5_xzplane(FNAME,GNAME,varxz)
            do I=0,NXM
            do J=0,NZP-1
               varxz(i,j)=0.5*(U2(i,j,NyMovie)+U2(i,j,NyMovie+1))
            end do
            end do
            GNAME='v_xz'
            call writeHDF5_xzplane(FNAME,GNAME,varxz)
            do I=0,NXM
            do J=0,NZP-1
               varxz(i,j)=U3(i,j,NyMovie)
            end do
            end do
            GNAME='w_xz'
            call writeHDF5_xzplane(FNAME,GNAME,varxz)
         END IF

         do I=0,NZP-1
         do J=1,NY
            varzy(i,j)=U1(NxMovie,i,j)
         end do
         end do
         GNAME='u_zy'
         call writeHDF5_zyplane(FNAME,GNAME,varzy)

         do I=0,NZP-1
         do J=1,NY
            varzy(i,j)=0.5*(U2(NxMovie,i,j)+U2(NxMovie,i,j+1))
         end do
         end do
         GNAME='v_zy'
         call writeHDF5_zyplane(FNAME,GNAME,varzy)

         do I=0,NZP-1
         do J=1,NY
            varzy(i,j)=U3(NxMovie,i,j)
         end do
         end do
         GNAME='w_zy'
         call writeHDF5_zyplane(FNAME,GNAME,varzy)

c$$$! This file will contain a time history over a plane
c$$$        open(85,file='movie_u.txt',status='unknown',form='formatted')
c$$$        do I=0,NXM
c$$$        do J=1,NY
c$$$          write(85,*) U1(I,0,J)
c$$$        end do
c$$$        end do
c$$$        open(86,file='movie_v.txt',status='unknown',form='formatted')
c$$$        do I=0,NXM
c$$$        do J=1,NY
c$$$          write(86,*) U2(I,0,J)
c$$$        end do
c$$$        end do
c$$$        open(87,file='movie_w.txt',status='unknown',form='formatted')
c$$$        do I=0,NXM
c$$$        do J=1,NY
c$$$          write(87,*) U3(I,0,J)
c$$$        end do
c$$$        end do

        END IF

C Convert velocity back to Fourier space
      call fft_xz_to_fourier(U1,CU1,0,NY+1)
      call fft_xz_to_fourier(U2,CU2,0,NY+1)
      call fft_xz_to_fourier(U3,CU3,0,NY+1)

      end if

      IF (RANK.EQ.0) 
     &     write(*,*) 'done save_stats chan' 

      call mpi_barrier(MPI_COMM_WORLD,ierror)

      RETURN
      END SUBROUTINE SAVE_STATS_CHAN

      subroutine tkebudget_chan
! NOte, it is important to only run this routine after complete R-K
!  time advancement since F1 is overwritten which is needed between R-K steps

      character*35 FNAME
      integer i,j,k

! Compute the turbulent dissipation rate, epsilon=nu*<du_i/dx_j du_i/dx_j>
      do j=2,NYM
        epsilon(j)=0.
      end do
! Store du/dx in CS1
      do j=2,NYM
      do k=0,TNKZ
      do i=0,NXP-1
        CS1(i,k,j)=CIKX(i)*CR1(i,k,j)
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+(S1(i,k,j)**2.0)
      end do
      end do
      end do
! Store dv/dx in CS1
      do j=2,NYM
      do k=0,TNKZ
      do i=0,NXP-1
        CS1(i,k,j)=CIKX(i)*(CR2(i,k,j)+CR2(i,k,j+1))/2.0
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+0.5*(S1(i,k,j)**2.0)
      end do
      end do
      end do
! Compute du/dy at GYF gridpoints, note remove mean
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        F1(i,k,j)=((U1(i,k,j+1)-CR1(0,0,j+1))
     &      -(U1(i,k,j-1)-CR1(0,0,j-1)))/(GY(j)+GY(j+1))
      end do
      end do
      end do
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+0.5*(F1(i,k,j)**2.0)
! Cross term dvdx*dudy
        epsilon(j)=epsilon(j)+(S1(i,k,j)*F1(i,k,j))
      end do
      end do
      end do
! Store dw/dx in CS1
      do j=2,NYM
      do k=0,TNKZ
      do i=0,NXP-1
        CS1(i,k,j)=CIKX(i)*CR3(i,k,j)
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+0.5*(S1(i,k,j)**2.0)
      end do
      end do
      end do
! Compute du/dz at GYF gridpoints, note remove mean
! Store du/dz in CS1
      do j=2,NYM
      do k=0,TNKZ
      do i=0,NXP-1
        CF1(i,k,j)=CIKZ(k)*CR1(i,k,j)
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CF1,F1,0,NY+1)
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+0.5*(F1(i,k,j)**2.0)
! Cross term dudz*dwdx
        epsilon(j)=epsilon(j)+S1(i,k,j)*F1(i,k,j)
      end do
      end do
      end do
! Compute dv/dy at GYF gridpoints, note remove mean
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        S1(i,k,j)=((U2(i,k,j+1)-CR2(0,0,j+1))-(U2(i,k,j)-CR2(0,0,j)))
     &            /GYF(j)
      end do
      end do
      end do
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+(S1(i,k,j)**2.0)
      end do
      end do
      end do
! Compute dw/dy at GYF gridpoints, note remove mean
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        S1(i,k,j)=((U3(i,k,j+1)-CR3(0,0,j+1))
     &      -(U3(i,k,j-1)-CR3(0,0,j-1)))/(GY(j)+GY(j+1))
      end do
      end do
      end do
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+0.5*(S1(i,k,j)**2.0)
      end do
      end do
      end do
! Store dv/dz in CF1
      do j=2,NYM
      do k=0,TNKZ
      do i=0,NXP-1
        CF1(i,k,j)=CIKZ(k)*(CR2(i,k,j)+CR2(i,k,j+1))/2.0
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CF1,F1,0,NY+1)
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+0.5*(F1(i,k,j)**2.0)
! Cross term dvdz*dwdy
        epsilon(j)=epsilon(j)+S1(i,k,j)*F1(i,k,j)
      end do
      end do
      end do
! Store dw/dz in CS1
      do j=2,NYM
      do k=0,TNKZ
      do i=0,NXP-1
        CS1(i,k,j)=CIKZ(k)*CR3(i,k,j)
      end do
      end do
      end do
! Convert to physical space
      call fft_xz_to_physical(CS1,S1,0,NY+1)
      do j=2,NYM
      do k=0,NZP-1
      do i=0,NXM
        epsilon(j)=epsilon(j)+(S1(i,k,j)**2.0)
      end do
      end do
      end do
      do j=2,NYM
        epsilon(j)=epsilon(j)/float(NX*NZ)
      end do
      call mpi_allreduce(mpi_in_place,epsilon,NY+2,MPI_DOUBLE_PRECISION,
     &     MPI_SUM,MPI_COMM_Z,ierror)

      IF (RANKZ.EQ.0) THEN
! Write out the mean statistics at each time
      IF (USE_MPI) THEN
        FNAME='tke'//trim(MPI_IO_NUM)//'.txt'
      ELSE
        FNAME='tke.txt'
      END IF
      open(45,file=FNAME,form='formatted',status='unknown')
      write(45,*) TIME_STEP,TIME,DELTA_T
      do j=2,NYM
        write(45,401) j,GYF(J),epsilon(j)
      end do
401   format(I3,' ',2(F20.9,' '))
      end if

      return 
      end subroutine tkebudget_chan
 
      subroutine filter_chan(N)
C This subroutine applies a filter to the highest wavenumbers
C It should be applied to the scalars in Fourier space
C The filter used is a sharpened raised cosine filter in the horizontal
C and a fourth order implicit compact filter in the vertical, with the
C parameter alpha determining the width of the vertical filtering window


      integer I,J,K,js,je,N

! Variables for horizontal filtering
      real*8 sigma(0:NKX,0:TNKZ),sigma0

! Variables for vertical filtering
      real*8 alpha
      parameter (alpha=0.0d0)
! Parameters for a larger stencil filter
      real*8 f_a,f_b,f_c

      js=0
      je=NY+1

C Set the filtering constants for the horizontal direction
      DO i=0,NKX
       DO k=0,TNKZ
        sigma0=0.5d0*(1.d0+
     &       cos(sqrt((KX(i)*LX*1.d0/float(NX))**2.d0
     &            +(KZ(k)*LZ*1.d0/float(NZ))**2.d0)))
! Apply a sharpened raised cosine filter
        sigma(i,k)=sigma0**4.d0*(35.d0-84.d0*sigma0
     &        +70.d0*sigma0**2.d0-20.d0*sigma0**3.d0)
       END DO
      END DO

      DO N=1,N_TH
C Do the spectral filtering in the horizontal
        DO K=0,TNKZ
          DO I=0,NKX
            DO J=js+1,je-1
              CTH(I,K,J,N)=CTH(I,K,J,N)*sigma(i,k)
            END DO
          END DO
        END DO
      END DO
C Set the filtering constants
      f_a=(1.d0/8.d0)*(5.d0+6.d0*alpha)
      f_b=0.5d0*(1.d0+2.d0*alpha)
      f_c=(-1.d0/8.d0)*(1.d0-2.d0*alpha)


      DO N=1,N_TH
C First, zero the tridiagonal matrix components
      DO I=0,NKX
        DO J=0,NY+1
          MATD_C(I,J)=1.d0
          MATL_C(I,J)=0.d0
          MATU_C(I,J)=0.d0
          VEC_C(I,J)=0.d0
        END DO
      END DO


C Filter the passive scalar, TH in the vertical direction
      DO K=1,TNKZ
        DO I=1,NKX
C Construct the centered difference terms
          DO J=2,NY-1
            MATL_C(I,J)=alpha
            MATD_C(I,J)=1.d0
            MATU_C(I,J)=alpha
            VEC_C(I,J)=f_a*CTH(I,K,J,N)
     &                +(f_b/2.d0)*(CTH(I,K,J+1,N)+CTH(I,K,J-1,N))
     &                +(f_c/2.d0)*(CTH(I,K,J+2,N)+CTH(I,K,J-2,N))
          END DO
C Now, construct the equations for the boundary nodes
          J=1
            MATL_C(I,J)=0.d0
            MATD_C(I,J)=1.d0
            MATU_C(I,J)=0.d0
            VEC_C(I,J)=CTH(I,K,J,N)
          J=NY
            MATL_C(I,J)=0.d0
            MATD_C(I,J)=1.d0
            MATU_C(I,J)=0.d0
            VEC_C(I,J)=CTH(I,K,J,N)
         END DO
C Now, solve the tridiagonal system
         CALL THOMAS_COMPLEX(MATL_C,MATD_C,MATU_C,VEC_C,NY,NKX)
         DO I=1,NKX
           DO J=js+1,je-1
             CTH(I,K,J,N)=VEC_C(I,J)
           END DO
         END DO
C END DO K  
       END DO

C END DO N 
       END DO
       return
       end subroutine filter_chan

      subroutine courant
! This subroutine sets the timestep based on the specified CFL number
! The subroutine should be called with the velocity in physical space


      real*8 vel
      real*8 dt
      real*8 dt_x,dt_y,dt_z
      integer i,j,k,n
      integer imin,jmin,kmin

! Set the initial dt to some arbitrary large number
      dt=999.d0

      dt=min(dt,0.5d0*min(dx(1),dy(1))/NU)
      do n=1,N_TH
        dt=min(dt,dt*NU/(NU/PR(n)))
      end do
! Make sure that we capture the inertial period (for rotating flows)
      if (I_RO_TAU.ne.0.d0) then
        dt=min(dt,2.d0*PI/I_RO_TAU/20.d0)
      end if

      do j=1,NY
        do k=0,NZP-1
          do i=0,NXM
            dt_x=cfl*dx(i)/abs(U1(i,k,j))
            dt_y=cfl*dy(j)/abs(U2(i,k,j))
            dt_z=cfl*dz(k)/abs(U3(i,k,j))
            dt=min(dt,dt_x,dt_y,dt_z)
          end do
        end do
      end do

      if (USE_MPI) then
         call get_minimum_mpi(dt)
      end if

      if (dt.le.0) then
        IF (RANK.EQ.0) 
     &        write(*,*) 'Error: dt<=0 in courant'
! Set DELTA_T to some small default value
        DELTA_T=0.0001d0
      else if (dt.ge.999.) then
!        write(*,*) 'WARNING: DELTA_T > 999, value capped at 999'
        DELTA_T=999.d0
      else
        DELTA_T=dt
      end if
      H_BAR(1)=DELTA_T*(8.0/15.0)
      H_BAR(2)=DELTA_T*(2.0/15.0)
      H_BAR(3)=DELTA_T*(5.0/15.0)

      return
      end subroutine courant

      SUBROUTINE TRACE
!----*|--.---------.---------.---------.---------.---------.---------.-|--
! This subroutine is to track trace of drag and kinetic energy
! The data is saved 
      REAL*8 TAUW1,TAUW2,UBULK
      REAL*8 Euua,Evva,Ewwa,E00a,E10a,E11a
      REAL*8 Euut,Evvt,Ewwt,E00t,E10t,E11t
      REAL*8 Euub,Evvb,Ewwb,E00b,E10b,E11b

      INTEGER J
      REAL*8 Euu(0:NY+1),Evv(0:NY+1),Eww(0:NY+1),E00(0:NY+1),E11(0:NY+1)
     &         , E10(0:NY+1)

! Calculate the wall shear stress on the top and bottom walls
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
      TAUW1=SQRT(TAUW1*NU)/NU; TAUW2=SQRT(DABS(TAUW2)*NU)/NU

! Calculate the bulk mean velocity
      CALL INTEGRATE_Y_VAR(UME,UBULK,MPI_COMM_WORLD)
!
! Calculate turbulent fluctuation energy Euu, Evv, Eww
      Euu(:)=0.0D0; Evv(:)=0.0D0; Eww(:)=0.0D0
      E00(:)=0.0D0; E10(:)=0.0D0; E11(:)=0.0D0

      DO J=2,NY
        DO K=0,TNKZ; DO I=0,NXP-1
          IF(I+RANKZ*NXP.LE.NKX) THEN
           Euu(J)=Euu(J)+0.5D0*(ABS(CU1(I,K,J))**2+ABS(CU1(I,K,J-1))**2) 
     & *DY(J)*2.0D0
           Evv(J)=Evv(J)+0.5D0*(ABS(CU2(I,K,J))**2+ABS(CU2(I,K,J-1))**2)
     & *DYF(J)*2.0D0
           Eww(J)=Eww(J)+0.5D0*(ABS(CU3(I,K,J))**2+ABS(CU3(I,K,J-1))**2)
     & *DY(J)*2.0D0
          END IF
        END DO; END DO
        DO I=1,NXP-1 ! K=0 modes
          IF(I+RANKZ*NXP.LE.NKX) THEN
           Euu(J)=Euu(J)-0.5D0*(ABS(CU1(I,0,J))**2+ABS(CU1(I,0,J-1))**2)
     & *DY(J)
           Evv(J)=Evv(J)-0.5D0*(ABS(CU2(I,0,J))**2+ABS(CU2(I,0,J-1))**2)
     & *DYF(J)
           Eww(J)=Eww(J)-0.5D0*(ABS(CU3(I,0,J))**2+ABS(CU3(I,0,J-1))**2)
     & *DY(J)
          END IF
        END DO
        IF(RANKZ.EQ.0) THEN
          DO K=1,TNKZ ! I=0 modes
           Euu(J)=Euu(J)-0.5D0*(ABS(CU1(0,K,J))**2+ABS(CU1(0,K,J-1))**2)
     & *DY(J)
           Evv(J)=Evv(J)-0.5D0*(ABS(CU2(0,K,J))**2+ABS(CU2(0,K,J-1))**2)
     & *DYF(J)
           Eww(J)=Eww(J)-0.5D0*(ABS(CU3(0,K,J))**2+ABS(CU3(0,K,J-1))**2)
     & *DY(J)
          END DO
          Euu(J)=Euu(J)-0.5D0*(ABS(CU1(0,0,J))**2+ABS(CU1(0,0,J-1))**2)
     & *DY(J)*2.0D0
!          Evv=Evv-0.5D0*(ABS(CU2(0,0,J))**2+ABS(CU2(0,0,J-1))**2)
!     & *DYF(J)*2.0D0
!          Eww=Eww-0.5D0*(ABS(CU3(0,0,J))**2+ABS(CU3(0,0,J-1))**2)
!     & *DY(J)*2.0D0
        END IF
! Quantify the streaks and mendery motion
        IF(RANKZ.EQ.0) THEN
          E00(J)=E00(J)+0.5D0*(ABS(CU1(0,1,J))**2+ABS(CU1(0,1,J-1))**2)
     &     *DY(J)+0.5D0*(ABS(CU2(0,1,J))**2+ABS(CU2(0,1,J-1))**2)*DYF(J)
     &           +0.5D0*(ABS(CU3(0,1,J))**2+ABS(CU3(0,1,J-1))**2)*DY(J)
          E10(J)=E10(J)+0.5D0*(ABS(CU1(1,0,J))**2+ABS(CU1(1,0,J-1))**2)
     &     *DY(J)+0.5D0*(ABS(CU2(1,0,J))**2+ABS(CU2(1,0,J-1))**2)*DYF(J)
     &           +0.5D0*(ABS(CU3(1,0,J))**2+ABS(CU3(1,0,J-1))**2)*DY(J)
          E11(J)=E11(J)+0.5D0*(ABS(CU1(1,1,J))**2+ABS(CU1(1,1,J-1))**2)
     &     *DY(J)+0.5D0*(ABS(CU2(1,1,J))**2+ABS(CU2(1,1,J-1))**2)*DYF(J)
     &           +0.5D0*(ABS(CU3(1,1,J))**2+ABS(CU3(1,1,J-1))**2)*DY(J)
        END IF
      END DO ! end of J
! Consider the whole channel domain, or the top core, or bottom core
      Euua=0.0D0; Evva=0.0D0; Ewwa=0.0D0; E00a=0.0D0; E10a=0.0D0; E11a=0.0D0
      Euut=0.0D0; Evvt=0.0D0; Ewwt=0.0D0; E00t=0.0D0; E10t=0.0D0; E11t=0.0D0
      Euub=0.0D0; Evvb=0.0D0; Ewwb=0.0D0; E00b=0.0D0; E10b=0.0D0; E11b=0.0D0
      DO J=2,NY
        IF(GYF(J).LE.ycore.AND.GYF(J).GE.-ycore) THEN
          Euua=Euua+Euu(J); Evva=Evva+Evv(J); Ewwa=Ewwa+Eww(J)
          E00a=E00a+E00(J); E10a=E10a+E10(J); E11a=E11a+E11(J)
        END IF
        IF(GYF(J).LE.ycore.AND.GYF(J).GE.0.0D0) THEN
          Euut=Euut+Euu(J); Evvt=Evvt+Evv(J); Ewwt=Ewwt+Eww(J)
          E00t=E00t+E00(J); E10t=E10t+E10(J); E11t=E11t+E11(J)
        END IF
        IF(GYF(J).GE.-ycore.AND.GYF(J).LE.0.0D0) THEN
          Euub=Euub+Euu(J); Evvb=Evvb+Evv(J); Ewwb=Ewwb+Eww(J)
          E00b=E00b+E00(J); E10b=E10b+E10(J); E11b=E11b+E11(J)
        END IF
      END DO
! 
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Euua,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Evva,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Ewwa,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E00a,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E10a,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E11a,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Euut,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Evvt,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Ewwt,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E00t,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E10t,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E11t,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Euub,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Evvb,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,Ewwb,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E00b,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E10b,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
      CALL MPI_ALLREDUCE(MPI_IN_PLACE,E11b,1,
     &        MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)
!
      Euua=Euua/ycore/2.0D0
      Evva=Evva/ycore/2.0D0
      Ewwa=Ewwa/ycore/2.0D0
      E00a=E00a/ycore/2.0D0
      E10a=E10a/ycore/2.0D0
      E11a=E11a/ycore/2.0D0
      Euut=Euut/ycore
      Evvt=Evvt/ycore
      Ewwt=Ewwt/ycore
      E00t=E00t/ycore
      E10t=E10t/ycore
      E11t=E11t/ycore
      Euub=Euub/ycore
      Evvb=Evvb/ycore
      Ewwb=Ewwb/ycore
      E00b=E00b/ycore
      E10b=E10b/ycore
      E11b=E11b/ycore
!      IF(RANK.EQ.0) THEN
!        WRITE(*,*) 'Euu=',Euu
!        WRITE(*,*) 'Evv=',Evv
!        WRITE(*,*) 'Eww=',Eww
!      END IF

! Write instantaneous statistics to files
      IF(RANK.EQ.0) THEN
        WRITE(1000,'(I10,5E20.10)') TIME_STEP,time,TAUW1,TAUW2,UBULK,PX0
        WRITE(1001,'(I10,7E20.10)') TIME_STEP,time,Euua,Evva,Ewwa,
     &                              E00a,E10a,E11a
        WRITE(1002,'(I10,7E20.10)') TIME_STEP,time,Euut,Evvt,Ewwt,
     &                              E00t,E10t,E11t
        WRITE(1003,'(I10,7E20.10)') TIME_STEP,time,Euub,Evvb,Ewwb,
     &                              E00b,E10b,E11b
      END IF

      END SUBROUTINE TRACE
      SUBROUTINE CONVERTER(K,TT)

      INTEGER K,TT

      IF(K.lt.0) THEN
        TT=2*NKZ+K+1
      ELSE IF(K.gt.0) THEN
        TT=K
      ELSE
        TT=K
      END IF


      END SUBROUTINE CONVERTER

      SUBROUTINE CONVERTERK(K,TT)

      INTEGER K,TT

      IF(K.gt.NKZ) THEN
        TT=-2*NKZ+K-1
      ELSE IF(K.lt.NKZ) THEN
        TT=K
      ELSE
        TT=K
      END IF


      END SUBROUTINE CONVERTERK
!
      END MODULE channel_module
