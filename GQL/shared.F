      MODULE shared_module
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! header, the header file for diablo.                              VERSION 0.3f
! This file contains definitions of global parameters and global variables.
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      IMPLICIT NONE
      INCLUDE 'mpif.h'

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! Spatial resolution parameters
! (We hardwire these into the code so that the compiler may perform
!  optimizations based on the grid size at compile time).
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      INTEGER   NX, NY, NZ, N_TH
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! grid_def, the portion of the header that defines the grid size for diablo.
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
       INCLUDE 'grid_def'
!      PARAMETER (NX=48)
!      PARAMETER(NY=11)
!      PARAMETER (NZ=48)
!      PARAMETER (N_TH=0)
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|

      INTEGER :: NPROCS, RANK
      INTEGER :: RANKY,RANKZ,MPI_COMM_Y,MPI_COMM_Z
      INTEGER :: status(MPI_STATUS_SIZE), IERROR
      INTEGER :: PNY,JOFF,NPROCY,NPROCZ,NXP,NZP, NXPP
      INTEGER :: XY2ZY_1,XY2ZY_2

! grid_mpi
      INCLUDE 'grid_mpi'
!      PARAMETER(NPROCS=16)
!      PARAMETER(NPROCY=4)
!      PARAMETER(NPROCZ=NPROCS/NPROCY)

      PARAMETER(NXP=NX/(2*NPROCZ))
      PARAMETER(NZP=NZ/(  NPROCZ))

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! Derived constants
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|

      INTEGER NKX, NKY, NKZ
      PARAMETER(NKX=NX/3,NKY=NY/3,NKZ=NZ/3)
      INTEGER  NXM, NYM, NZM, TNKY,TNKZ
      PARAMETER(NXM=NX-1)
      PARAMETER(NYM=NY-1)
      PARAMETER(NZM=NZ-1)
      PARAMETER(TNKY=2*NKY)
      PARAMETER(TNKZ=2*NKZ)

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! Input parameters and runtime variables
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      LOGICAL USE_MPI,GUSTS,LES
      CHARACTER*35 FLAVOR
      CHARACTER*10 MPI_IO_NUM
      REAL*8  NU, LX, LY, LZ, DELTA_T, KICK, UBULK0, PX0
      REAL*8  U_BC_XMIN_C1, U_BC_XMIN_C2, U_BC_XMIN_C3
      REAL*8  V_BC_XMIN_C1, V_BC_XMIN_C2, V_BC_XMIN_C3
      REAL*8  W_BC_XMIN_C1, W_BC_XMIN_C2, W_BC_XMIN_C3
      REAL*8  U_BC_YMIN_C1, U_BC_YMIN_C2, U_BC_YMIN_C3
      REAL*8  V_BC_YMIN_C1, V_BC_YMIN_C2, V_BC_YMIN_C3
      REAL*8  W_BC_YMIN_C1, W_BC_YMIN_C2, W_BC_YMIN_C3
      REAL*8  U_BC_ZMIN_C1, U_BC_ZMIN_C2, U_BC_ZMIN_C3
      REAL*8  V_BC_ZMIN_C1, V_BC_ZMIN_C2, V_BC_ZMIN_C3
      REAL*8  W_BC_ZMIN_C1, W_BC_ZMIN_C2, W_BC_ZMIN_C3
      REAL*8  TH_BC_XMIN_C1(1:N_TH), TH_BC_XMIN_C2(1:N_TH)
     &         , TH_BC_XMIN_C3(1:N_TH)
      REAL*8  TH_BC_YMIN_C1(1:N_TH), TH_BC_YMIN_C2(1:N_TH)
     &         , TH_BC_YMIN_C3(1:N_TH)
      REAL*8  TH_BC_ZMIN_C1(1:N_TH), TH_BC_ZMIN_C2(1:N_TH)
     &         , TH_BC_ZMIN_C3(1:N_TH)
      REAL*8  U_BC_XMAX_C1, U_BC_XMAX_C2, U_BC_XMAX_C3
      REAL*8  V_BC_XMAX_C1, V_BC_XMAX_C2, V_BC_XMAX_C3
      REAL*8  W_BC_XMAX_C1, W_BC_XMAX_C2, W_BC_XMAX_C3
      REAL*8  U_BC_YMAX_C1, U_BC_YMAX_C2, U_BC_YMAX_C3
      REAL*8  V_BC_YMAX_C1, V_BC_YMAX_C2, V_BC_YMAX_C3
      REAL*8  W_BC_YMAX_C1, W_BC_YMAX_C2, W_BC_YMAX_C3
      REAL*8  U_BC_ZMAX_C1, U_BC_ZMAX_C2, U_BC_ZMAX_C3
      REAL*8  V_BC_ZMAX_C1, V_BC_ZMAX_C2, V_BC_ZMAX_C3
      REAL*8  W_BC_ZMAX_C1, W_BC_ZMAX_C2, W_BC_ZMAX_C3
      REAL*8  TH_BC_XMAX_C1(1:N_TH), TH_BC_XMAX_C2(1:N_TH)
     &         , TH_BC_XMAX_C3(1:N_TH)
      REAL*8  TH_BC_YMAX_C1(1:N_TH), TH_BC_YMAX_C2(1:N_TH)
     &         , TH_BC_YMAX_C3(1:N_TH)
      REAL*8  TH_BC_ZMAX_C1(1:N_TH), TH_BC_ZMAX_C2(1:N_TH)
     &         , TH_BC_ZMAX_C3(1:N_TH)
      REAL*8  CFL 

      INTEGER NX_T,NY_T,NZ_T
      INTEGER N_TIME_STEPS, NUM_PER_DIR, TIME_AD_METH, VERBOSITY  
      INTEGER SAVE_FLOW_INT, SAVE_STATS_INT, IC_TYPE, F_TYPE
      INTEGER U_BC_XMIN, V_BC_XMIN, W_BC_XMIN, TH_BC_XMIN(1:N_TH)
      INTEGER U_BC_XMAX, V_BC_XMAX, W_BC_XMAX, TH_BC_XMAX(1:N_TH)
      INTEGER U_BC_YMIN, V_BC_YMIN, W_BC_YMIN, TH_BC_YMIN(1:N_TH)
      INTEGER U_BC_YMAX, V_BC_YMAX, W_BC_YMAX, TH_BC_YMAX(1:N_TH)
      INTEGER U_BC_ZMIN, V_BC_ZMIN, W_BC_ZMIN, TH_BC_ZMIN(1:N_TH)
      INTEGER U_BC_ZMAX, V_BC_ZMAX, W_BC_ZMAX, TH_BC_ZMAX(1:N_TH)
      INTEGER PREVIOUS_TIME_STEP
      INTEGER UPDATE_DT
      LOGICAL VARIABLE_DT,FIRST_TIME 
      LOGICAL MOVIE,CREATE_NEW_FLOW

      REAL*8  TIME
      INTEGER TIME_STEP, RK_STEP

      REAL*8  XcMovie, YcMovie, ZcMovie
      INTEGER NxMovie, NyMovie, NzMovie, RankYMovie, RankZMovie

      INTEGER TIME_ARRAY(8)
      REAL*8 START_TIME,END_TIME,TIME_LIMIT

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for scalar advection
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8  RI_TAU(1:N_TH), PR(1:N_TH), REACTION(1:N_TH)
      LOGICAL CREATE_NEW_TH(1:N_TH), BACKGROUND_GRAD(1:N_TH)
      INTEGER NUM_READ_TH
      INTEGER READ_TH_INDEX(1:N_TH)
      LOGICAL FILTER_TH(1:N_TH)
      INTEGER FILTER_INT(1:N_TH)
      INTEGER JSTART_TH(1:N_TH),JEND_TH(1:N_TH)

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for Chemotaxis simulations 
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8 CHI(1:N_TH),THTH(1:N_TH),THVAR(1:N_TH)

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for oscillatory pressure forcing
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8 OMEGA0, AMP_OMEGA0

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for frontogensis
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8 I_RO_TAU, I_FR

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for Low Reynolds number periodic advection
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8 EK0,EK,EPSILON_TARGET

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for Large Eddy Simulation
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      INTEGER LES_MODEL_TYPE
      REAL*8 NU_T (0:NX+1,0:NZP+1,0:NY+1)
      REAL*8 KAPPA_T (0:NX+1,0:NZP+1,0:NY+1,1:N_TH)

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! RKW3 parameters
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8  H_BAR(3), BETA_BAR(3), ZETA_BAR(3)

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! FFT parameters
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8  KX  (0:NXP), KY  (0:2*NKY), KZ  (0:2*NKZ),
     &        KX2 (0:NXP), KY2 (0:2*NKY), KZ2 (0:2*NKZ),
     &        PI, EPS, RNX, RNY, RNZ
      COMPLEX*16 CIKX(0:NXP), CIKY(0:2*NKY), CIKZ(0:2*NKZ), CI,
     &           CZX_PLANE(0:NZ,0:NKX), CYZ_PLANE(0:NY,0:2*NKZ)
      INTEGER*8  FFTW_X_TO_P_PLAN, FFTW_X_TO_F_PLAN,
     &           FFTW_Y_TO_P_PLAN, FFTW_Y_TO_F_PLAN,
     &           FFTW_Z_TO_P_PLAN, FFTW_Z_TO_F_PLAN

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! Grid parameters
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8  GX(0:NX+1), GY(0:NY+1), GZ(0:NZ+1),
     &        DX(0:NX+1), DY(0:NY+1), DZ(0:NZ+1),
     &        GXF(0:NX+1),  GYF(0:NY+1),  GZF(0:NZ+1),
     &        DXF(0:NX+1),  DYF(0:NY+1),  DZF(0:NZ+1)
      INTEGER JSTART,JEND

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
! Global variables
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      REAL*8  U1 (0:NX+1,0:NZP+1,0:NY+1), U2 (0:NX+1,0:NZP+1,0:NY+1),
     &        U3 (0:NX+1,0:NZP+1,0:NY+1), P  (0:NX+1,0:NZP+1,0:NY+1),
     &        R1 (0:NX+1,0:NZP+1,0:NY+1), R2 (0:NX+1,0:NZP+1,0:NY+1),
     &        R3 (0:NX+1,0:NZP+1,0:NY+1), F1 (0:NX+1,0:NZP+1,0:NY+1),
     &        F2 (0:NX+1,0:NZP+1,0:NY+1), F3 (0:NX+1,0:NZP+1,0:NY+1),
     &        S1 (0:NX+1,0:NZP+1,0:NY+1),
     &        SUU(0:NX+1,0:NZP+1,0:NY+1),
     &        SUV(0:NX+1,0:NZP+1,0:NY+1),
     &        SVV(0:NX+1,0:NZP+1,0:NY+1),
     &        SVW(0:NX+1,0:NZP+1,0:NY+1),
     &        SUW(0:NX+1,0:NZP+1,0:NY+1),
     &        SWW(0:NX+1,0:NZP+1,0:NY+1),
     &        TH (0:NX+1,0:NZP+1,0:NY+1,1:N_TH),
     &        FTH (0:NX+1,0:NZP+1,0:NY+1,1:N_TH), 
     &        RTH (0:NX+1,0:NZP+1,0:NY+1,1:N_TH)

      REAL*8  U1P(0:NX+1,0:NZP+1,0:NY+1), 
     &        U2P(0:NX+1,0:NZP+1,0:NY+1), U3P(0:NX+1,0:NZP+1,0:NY+1)
     
      REAL*8  Q1(0:NX+1,0:NZP+1,0:NY+1), Q2(0:NX+1,0:NZP+1,0:NY+1),
     &        Q1A(0:NX+1,0:NZP+1,0:NY+1), Q1B(0:NX+1,0:NZP+1,0:NY+1)

      COMPLEX*16 CQ1(0:NXP,0:NZ+1,0:NY+1), CQ2(0:NXP,0:NZ+1,0:NY+1),
     &        CQ1A(0:NXP,0:NZ+1,0:NY+1), CQ1B(0:NXP,0:NZ+1,0:NY+1)

      COMPLEX*16 CU1P(0:NXP,0:NZ+1,0:NY+1), CU2P(0:NXP,0:NZ+1,0:NY+1), 
     &           CU3P(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 CU1M(0:NXP,0:NZ+1,0:NY+1), CU2M(0:NXP,0:NZ+1,0:NY+1), 
     &           CU3M(0:NXP,0:NZ+1,0:NY+1)
     
      REAL*8 U1M(0:NX+1,0:NZP+1,0:NY+1), U2M(0:NX+1,0:NZP+1,0:NY+1), 
     &       U3M(0:NX+1,0:NZP+1,0:NY+1)


! The following variables are used to compute the discrimenant
! Note, that they are equivalenced to other variables, so don't
! use extra memory
      REAL*8     A11(0:NX+1,0:NZP+1,0:NY+1), A22(0:NX+1,0:NZP+1,0:NY+1),
     &           A33(0:NX+1,0:NZP+1,0:NY+1), A12(0:NX+1,0:NZP+1,0:NY+1),
     &           A13(0:NX+1,0:NZP+1,0:NY+1), A23(0:NX+1,0:NZP+1,0:NY+1),
     &           A21(0:NX+1,0:NZP+1,0:NY+1), A31(0:NX+1,0:NZP+1,0:NY+1),
     &           A32(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8     Third_ivar(0:NX+1,0:NZP+1,0:NY+1)
     &          ,Enstrophy(0:NX+1,0:NZP+1,0:NY+1)
     &          ,Second_ivar(0:NX+1,0:NZP+1,0:NY+1)
     &          ,Strain_rate(0:NX+1,0:NZP+1,0:NY+1)
     &          ,Discriminant(0:NX+1,0:NZP+1,0:NY+1)

      REAL*8     MATL (0:NX-1,0:NY+1), MATD(0:NX-1,0:NY+1),
     &     MATU(0:NX-1,0:NY+1), VEC(0:NX-1,0:NY+1)
      REAL*8     MATL_t (0:NX-1,0:NY+1), MATD_t(0:NX-1,0:NY+1),
     &     MATU_t(0:NX-1,0:NY+1)
      COMPLEX*16 VEC_C(0:NXP,0:NY+1)
      REAL*8     MATL_C(0:NXP,0:NY+1),MATD_C(0:NXP,0:NY+1),
     &     MATU_C(0:NXP,0:NY+1)

      COMPLEX*16 CU1(0:NXP,0:NZ+1,0:NY+1), CU2(0:NXP,0:NZ+1,0:NY+1), 
     &           CU3(0:NXP,0:NZ+1,0:NY+1), CP (0:NXP,0:NZ+1,0:NY+1),
     &           CR1(0:NXP,0:NZ+1,0:NY+1), CR2(0:NXP,0:NZ+1,0:NY+1),
     &           CR3(0:NXP,0:NZ+1,0:NY+1), CF1(0:NXP,0:NZ+1,0:NY+1),
     &           CF2(0:NXP,0:NZ+1,0:NY+1), CF3(0:NXP,0:NZ+1,0:NY+1),
     &           CS1(0:NXP,0:NZ+1,0:NY+1),
     &           CSUU(0:NXP,0:NZ+1,0:NY+1),
     &           CSUV(0:NXP,0:NZ+1,0:NY+1),
     &           CSVV(0:NXP,0:NZ+1,0:NY+1),
     &           CSVW(0:NXP,0:NZ+1,0:NY+1),
     &           CSUW(0:NXP,0:NZ+1,0:NY+1),
     &           CSWW(0:NXP,0:NZ+1,0:NY+1),
     &           CTH(0:NXP,0:NZ+1,0:NY+1,1:N_TH),
     &           CFTH(0:NXP,0:NZ+1,0:NY+1,1:N_TH), 
     &           CRTH(0:NXP,0:NZ+1,0:NY+1,1:N_TH)


! The following variables are used to compute the discrimenant
! Note, that they are equivalenced to other variables, so don't
! use extra memory
      COMPLEX*16 CA11(0:NXP,0:NZ+1,0:NY+1), CA22(0:NXP,0:NZ+1,0:NY+1),
     &           CA33(0:NXP,0:NZ+1,0:NY+1), CA12(0:NXP,0:NZ+1,0:NY+1),
     &           CA13(0:NXP,0:NZ+1,0:NY+1), CA23(0:NXP,0:NZ+1,0:NY+1),
     &           CA21(0:NXP,0:NZ+1,0:NY+1), CA31(0:NXP,0:NZ+1,0:NY+1),
     &           CA32(0:NXP,0:NZ+1,0:NY+1)
                
      INTEGER DIMALLOC
      PARAMETER(DIMALLOC=(MAX(NX,NZ)+2)*(MIN(NX,NZ)/NPROCZ+2)*(NY+2))
      REAL*8 AU1(DIMALLOC),AU2(DIMALLOC),AU3(DIMALLOC),
     &       AR1(DIMALLOC),AR2(DIMALLOC),AR3(DIMALLOC),
     &	     AF1(DIMALLOC),AF2(DIMALLOC),AF3(DIMALLOC),
     &       AP (DIMALLOC),AS1(DIMALLOC)
      REAL*8 ATH(DIMALLOC*N_TH),ARTH(DIMALLOC*N_TH),AFTH(DIMALLOC*N_TH)

      EQUIVALENCE (AU1,U1,CU1,A11,CA11,Third_ivar)
     &          , (AU2,U2,CU2,A22,CA22,Enstrophy)
     &          , (AU3,U3,CU3,A33,CA33,Second_ivar)
     &          , (AR1,R1,CR1,A12,CA12,Discriminant)
     &          , (AR2,R2,CR2,A13,CA13)
     &          , (AR3,R3,CR3,A23,CA23)
     &          , (AF1,F1,CF1,A21,CA21)
     &          , (AF2,F2,CF2,A31,CA31)
     &          , (AF3,F3,CF3,A32,CA32)
     &          , (AP,P,CP,Strain_rate)
     &          , (AS1,S1,CS1)
     &          , (ARTH,RTH,CRTH)
     &          , (ATH,TH,CTH)
     &          , (AFTH,FTH, CFTH)
     &          , (U1P,CU1P)
     &          , (U2P,CU2P)
     &          , (U3P,CU3P)
     &          , (U1M,CU1M)
     &          , (U2M,CU2M)
     &          , (U3M,CU3M)
     &          , (SUU,CSUU)
     &          , (SVV,CSVV)
     &          , (SWW,CSWW)
     &          , (SUV,CSUV)
     &          , (SUW,CSUW)
     &          , (SVW,CSVW)
     &          , (Q1,CQ1)
     &          , (Q2,CQ2)

ccccccccccccccccccccccccccccc
      INTEGER l,m,lt,mt,tt,ta,uu,dai

      COMPLEX*16 pixsub(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 pix(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 piysub(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 piy(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 pizsub(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 piz(0:NXP,0:NZ+1,0:NY+1)

      REAL*8 prod(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 prestrans(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 diffx(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 diffy(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 diffz(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 vistransx(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 vistransy(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 vistransz(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 turbtransx(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 turbtransy(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 turbtransz(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 tempu(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 sgsx(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 sgsy(0:NXP,0:NZ+1,0:NY+1)
      REAL*8 sgsz(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumx1(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumy1(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumz1(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumx2(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumy2(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumz2(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumx3(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumy3(0:NXP,0:NZ+1,0:NY+1)
      COMPLEX*16 sumz3(0:NXP,0:NZ+1,0:NY+1)
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|! Parameters for truck forcing
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      LOGICAL TRUCK
      REAL*8 AF(DIMALLOC),AS2(DIMALLOC)
      REAL*8 F(0:NX+1,0:NZP+1,0:NY+1), D(0:NX+1,0:NZP+1,0:NY+1),
     &       S2(0:NX+1,0:NZP+1,0:NY+1),
     &       POS(3),RPOS(3),FPOS(3),THETA,RTHETA,FTHETA,V0,
     &       MAG_VAR, THETA_VAR, WIDTH
      COMPLEX*16 CF(0:NXP,0:NZ+1,0:NY+1),
     &           CS2(0:NXP,0:NZ+1,0:NY+1)
      EQUIVALENCE (AF,F,CF),(AS2,S2,CS2)

! Variables for outputting statistics
      REAL*8 UBAR(0:NY+1),VBAR(0:NY+1),WBAR(0:NY+1)
      REAL*8 URMS(0:NY+1),VRMS(0:NY+1),WRMS(0:NY+1)
      REAL*8 UME (0:NY+1),VME (0:NY+1),WME (0:NY+1)
      REAL*8 UV(0:NY+1),UW(0:NY+1),WV(0:NY+1)
      REAL*8 DUDY(0:NY+1),DWDY(0:NY+1)
      REAL*8 URMS_B,VRMS_B,WRMS_B,TKE_B
      REAL*8 SHEAR(0:NY+1)
      REAL*8 OMEGA_X(0:NY+1),OMEGA_Y(0:NY+1),OMEGA_Z(0:NY+1)

! Vertical correlation
      REAL*8 Cuu(0:NY+1),Cvv(0:NY+1),Cww(0:NY+1),
     &       Cuv(0:NY+1),Cuw(0:NY+1),Cwv(0:NY+1)
      REAL*8 ycore
      PARAMETER (ycore=50.0/525.6)

! Variables for NetCDF files
      integer ncid_stats,ncid_vis,T_VARID
      integer urms_varid, vrms_varid, wrms_varid
      integer ubar_varid, vbar_varid, wbar_varid
      integer uv_varid, wv_varid, uw_varid
      integer u_varid, v_varid, w_varid, p_varid, th_varid
      integer nc_start(2),nc_count(2),nc_index

! Variables needed for SAVE_STATS_TH
      REAL*8 THBAR(0:NY+1,1:N_TH),THRMS_B(1:N_TH)
     &       ,THRMS(0:NY+1,1:N_TH),THME(0:NY+1,1:N_TH)
      REAL*8 THV(0:NY+1,1:N_TH),DTHDY(0:NY+1,1:N_TH)
      REAL*8 PE_DISS(0:NY+1,1:N_TH)

! Variables for tkebudget
      REAL*8 EPSILON(0:NY+1)

      INTEGER NSAMPLES

! LES
      REAL*8 C_SMAG,C_VREMAN
      REAL*8  Sijc(0:NX+1,0:NZP+1,0:NY+1,1:6)
      COMPLEX*16 CSijc(0:NXP,0:NZ+1,0:NY+1,1:6)
      REAL*8  Aijc(0:NX+1,0:NZP+1,0:NY+1,1:9)
      COMPLEX*16 CAijc(0:NXP,0:NZ+1,0:NY+1,1:9)
!      INTEGER U_BC_LOWER,W_BC_LOWER,U_BC_UPPER,W_BC_UPPER
      REAL*8  betaVc(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  beta11c(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  beta22c(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  beta33c(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  beta12c(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  beta13c(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  beta23c(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  TEMPYc(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  facc(0:NX+1,0:NZP+1,0:NY+1)
      REAL*8  NU_T2c(0:NX+1,0:NZP+1,0:NY+1)
      real*8 del_xc,del_zc
!      parameter (C_VREMAN=0.03d0)
      REAL*8 NU_Tc (0:NX+1,0:NZP+1,0:NY+1)

!      EQUIVALENCE (SijC,CSijC)
!      EQUIVALENCE (AijC,CAijC)

! Variable for spectra
      REAL*8 phiuu(0:NXP,0:NZ+1,0:NY+1),phivv(0:NXP,0:NZ+1,0:NY+1),
     &       phiww(0:NXP,0:NZ+1,0:NY+1),phiuv(0:NXP,0:NZ+1,0:NY+1),
     &       phipp(0:NXP,0:NZ+1,0:NY+1)

      END MODULE shared_module
