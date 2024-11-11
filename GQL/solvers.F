      MODULE solvers_module
      USE shared_module

      CONTAINS
 
C----*|--.---------.---------.---------.---------.---------.---------.-|------
      SUBROUTINE APPLY_BC_1_LOWER(MATL1,MATD1,MATU1,VEC1)
C----*|--.---------.---------.---------.---------.---------.---------.-|-----
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      INTEGER I

C Bottom Wall:
      IF (U_BC_YMIN.EQ.0) THEN
C Dirichlet
        DO I=0,NXM
          MATL1(I,0)=0. 
          MATD1(I,0)=1.
          MATU1(I,0)=0.                   
          VEC1(I,0)=0.

          MATL1(I,1)=0. 
          MATD1(I,1)=1.
          MATU1(I,1)=0.                   
          VEC1(I,1)=U_BC_YMIN_C1 
        END DO
      ELSE
C Neumann
        DO I=0,NXM
          MATL1(I,0)=0.
          MATD1(I,0)=-1.
          MATU1(I,0)=1.
          VEC1(I,0)=DY(1)*U_BC_YMIN_C1
        END DO
      END IF

      RETURN 
      END SUBROUTINE APPLY_BC_1_LOWER

C----*|--.---------.---------.---------.---------.---------.---------.-|------
      SUBROUTINE APPLY_BC_1_LOWER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1)
C----*|--.---------.---------.---------.---------.---------.---------.-|-----
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      INTEGER I

C Bottom Wall:
      IF (U_BC_YMIN.EQ.0) THEN
C Dirichlet
        DO I=0,NKX
          MATL_C1(I,0)=0. 
          MATD_C1(I,0)=1.
          MATU_C1(I,0)=0.                   
          VEC_C1(I,0)=0.

          MATL_C1(I,1)=0. 
          MATD_C1(I,1)=1.
          MATU_C1(I,1)=0.                   
          VEC_C1(I,1)=U_BC_YMIN_C1 
        END DO
      ELSE
C Neumann
        DO I=0,NKX
          MATL_C1(I,0)=0.
          MATD_C1(I,0)=-1.
          MATU_C1(I,0)=1.
          VEC_C1(I,0)=DY(1)*U_BC_YMIN_C1
        END DO
      END IF

      RETURN 
      END SUBROUTINE APPLY_BC_1_LOWER_C


C----*|--.---------.---------.---------.---------.---------.---------.-|----
      SUBROUTINE APPLY_BC_1_UPPER(MATL1,MATD1,MATU1,VEC1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      INTEGER I

C Top wall
      IF (U_BC_YMAX.EQ.0) THEN
C Dirichlet
        DO I=0,NXM
          MATL1(I,NY+1)=0.
          MATD1(I,NY+1)=1.
          MATU1(I,NY+1)=0.
          VEC1(I,NY+1)=0.

          MATL1(I,NY)=0.
          MATD1(I,NY)=1.
          MATU1(I,NY)=0.
          VEC1(I,NY)=U_BC_YMAX_C1
        END DO
      ELSE
C Neumann
        DO I=0,NXM
          MATL1(I,NY)=-1.
          MATD1(I,NY)=1.
          MATU1(I,NY)=0.
          VEC1(I,NY)=DY(NY)*U_BC_YMAX_C1
        END DO
        DO I=0,NXM
          MATL1(I,NY+1)=0.
          MATD1(I,NY+1)=1.
          MATU1(I,NY+1)=0.
          VEC1(I,NY+1)=0.
        END DO

      END IF

      RETURN
      END SUBROUTINE APPLY_BC_1_UPPER

C----*|--.---------.---------.---------.---------.---------.---------.-|----
      SUBROUTINE APPLY_BC_1_UPPER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      INTEGER I

C Top wall
      IF (U_BC_YMAX.EQ.0) THEN
C Dirichlet
        DO I=0,NKX
          MATL_C1(I,NY+1)=0.
          MATD_C1(I,NY+1)=1.
          MATU_C1(I,NY+1)=0.
          VEC_C1(I,NY+1)=0.

          MATL_C1(I,NY)=0.
          MATD_C1(I,NY)=1.
          MATU_C1(I,NY)=0.
          VEC_C1(I,NY)=U_BC_YMAX_C1
        END DO
      ELSE
C Neumann
        DO I=0,NKX
          MATL_C1(I,NY+1)=-1.
          MATD_C1(I,NY+1)=1.
          MATU_C1(I,NY+1)=0.
          VEC_C1(I,NY+1)=DY(NY+1)*U_BC_YMAX_C1
        END DO      
      END IF

      RETURN
      END SUBROUTINE APPLY_BC_1_UPPER_C


C----*|--.---------.---------.---------.---------.---------.---------.-|---
      SUBROUTINE APPLY_BC_2_LOWER(MATL1,MATD1,MATU1,VEC1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      INTEGER I

C Bottom Wall:
      IF (V_BC_YMIN.EQ.0) THEN
C Dirichlet
        DO I=0,NXM
          MATL1(I,1)=0.d0 
          MATD1(I,1)=1.d0
          MATU1(I,1)=0.d0                   
          VEC1(I,1)=V_BC_YMIN_C1 

          MATL1(I,2)=0.d0 
          MATD1(I,2)=1.d0
          MATU1(I,2)=0.d0                   
          VEC1(I,2)=V_BC_YMIN_C1 
        END DO
      ELSE IF (V_BC_YMIN.EQ.1) THEN
C Neumann
        DO I=0,NXM
          MATD1(I,1)=-1.d0
          MATU1(I,1)=1.d0
          MATL1(I,1)=0.d0
          VEC1(I,1)=DYF(1)*V_BC_YMIN_C1
        END DO
      END IF

C The following is only a placeholder, this row is used for U1 and U3
      DO I=0,NXM
        MATL1(I,0) = 0.
        MATD1(I,0) = 1.
        MATU1(I,0) = 0.
        VEC1(I,0) = 0.
      END DO

      RETURN
      END SUBROUTINE APPLY_BC_2_LOWER

C----*|--.---------.---------.---------.---------.---------.---------.-|---
      SUBROUTINE APPLY_BC_2_LOWER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      INTEGER I

C Bottom Wall:
      IF (V_BC_YMIN.EQ.0) THEN
C Dirichlet
        DO I=0,NKX
          MATL_C1(I,1)=0.d0 
          MATD_C1(I,1)=1.d0
          MATU_C1(I,1)=0.d0                   
          VEC_C1(I,1)=V_BC_YMIN_C1 

          MATL_C1(I,2)=0.d0 
          MATD_C1(I,2)=1.d0
          MATU_C1(I,2)=0.d0                   
          VEC_C1(I,2)=V_BC_YMIN_C1 
        END DO
      ELSE IF (V_BC_YMIN.EQ.1) THEN
C Neumann
        DO I=0,NKX
          MATD_C1(I,1)=-1.d0
          MATU_C1(I,1)=1.d0
          MATL_C1(I,1)=0.d0
          VEC_C1(I,1)=DYF(1)*V_BC_YMIN_C1
        END DO
      END IF

C The following is only a placeholder, this row is used for U1 and U3
      DO I=0,NKX
        MATL_C1(I,0) = 0.
        MATD_C1(I,0) = 1.
        MATU_C1(I,0) = 0.
        VEC_C1(I,0) = 0.
      END DO

      RETURN
      END SUBROUTINE APPLY_BC_2_LOWER_C

 
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_2_UPPER(MATL1,MATD1,MATU1,VEC1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      INTEGER I
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
C Top wall
      IF (V_BC_YMAX.EQ.0) THEN
C Dirichlet
        DO I=0,NXM
          MATL1(I,NY+1)=0.
          MATD1(I,NY+1)=1.
          MATU1(I,NY+1)=0.
          VEC1(I,NY+1)=V_BC_YMAX_C1
          
          MATL1(I,NY)=0.
          MATD1(I,NY)=1.
          MATU1(I,NY)=0.
          VEC1(I,NY)=V_BC_YMAX_C1
        END DO
      ELSE IF (V_BC_YMAX.EQ.1) THEN
C Neumann
        DO I=0,NXM
          MATL1(I,NY+1)=-1.
          MATD1(I,NY+1)=1.
          MATU1(I,NY+1)=0.
          VEC1(I,NY+1)=DYF(NY)*V_BC_YMAX_C1
        END DO
      END IF      
      RETURN
      END SUBROUTINE APPLY_BC_2_UPPER

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_2_UPPER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      INTEGER I
C Top wall
      IF (V_BC_YMAX.EQ.0) THEN
C Dirichlet
        DO I=0,NKX
          MATL_C1(I,NY+1)=0.
          MATD_C1(I,NY+1)=1.
          MATU_C1(I,NY+1)=0.
          VEC_C1(I,NY+1)=V_BC_YMAX_C1
          
          MATL_C1(I,NY)=0.
          MATD_C1(I,NY)=1.
          MATU_C1(I,NY)=0.
          VEC_C1(I,NY)=V_BC_YMAX_C1
        END DO
      ELSE IF (V_BC_YMAX.EQ.1) THEN
C Neumann
        DO I=0,NKX
          MATL_C1(I,NY+1)=-1.
          MATD_C1(I,NY+1)=1.
          VEC_C1(I,NY+1)=DYF(NY)*V_BC_YMAX_C1
        END DO      
      END IF
      RETURN
      END SUBROUTINE APPLY_BC_2_UPPER_C

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_3_LOWER(MATL1,MATD1,MATU1,VEC1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      INTEGER I

C Bottom Wall:
      IF (W_BC_YMIN.EQ.0) THEN
C Dirichlet
        DO I=0,NXM
          MATL1(I,0)=0. 
          MATD1(I,0)=1.
          MATU1(I,0)=0.                   
          VEC1(I,0)=0.

          MATL1(I,1)=0. 
          MATD1(I,1)=1.
          MATU1(I,1)=0.                   
          VEC1(I,1)=W_BC_YMIN_C1
        END DO
      ELSE
C Neumann
        DO I=0,NXM
          MATL1(I,0)=0.
          MATD1(I,0)=-1.
          MATU1(I,0)=1.
          VEC1(I,0)=DY(1)*W_BC_YMIN_C1
        END DO
      END IF

      RETURN
      END SUBROUTINE APPLY_BC_3_LOWER

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_3_LOWER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      INTEGER I

C Bottom Wall:
      IF (W_BC_YMIN.EQ.0) THEN
C Dirichlet
        DO I=0,NKX
          MATL_C1(I,0)=0. 
          MATD_C1(I,0)=1.
          MATU_C1(I,0)=0.                   
          VEC_C1(I,0)=0.

          MATL_C1(I,1)=0. 
          MATD_C1(I,1)=1.
          MATU_C1(I,1)=0.                   
          VEC_C1(I,1)=W_BC_YMIN_C1
        END DO
      ELSE
C Neumann
        DO I=0,NKX
          MATL_C1(I,0)=0.
          MATD_C1(I,0)=-1.
          MATU_C1(I,0)=1.
          VEC_C1(I,0)=DY(1)*W_BC_YMIN_C1
        END DO
      END IF

      RETURN
      END SUBROUTINE APPLY_BC_3_LOWER_C

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_3_UPPER(MATL1,MATD1,MATU1,VEC1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      INTEGER I

C Top wall
      IF (W_BC_YMAX.EQ.0) THEN
C Dirichlet
        DO I=0,NXM
          MATL1(I,NY+1)=0.
          MATD1(I,NY+1)=1.
          MATU1(I,NY+1)=0.
          VEC1(I,NY+1)=0.

          MATL1(I,NY)=0.
          MATD1(I,NY)=1.
          MATU1(I,NY)=0.
          VEC1(I,NY)=W_BC_YMAX_C1
        END DO
      ELSE
C Neumann
        DO I=0,NXM
          MATL1(I,NY)=-1.
          MATD1(I,NY)=1.
          MATU1(I,NY)=0.
          VEC1(I,NY)=DY(NY)*W_BC_YMAX_C1
        END DO
        DO I=0,NXM
          MATL1(I,NY+1)=0.
          MATD1(I,NY+1)=1.
          MATU1(I,NY+1)=0.
          VEC1(I,NY+1)=0.
        END DO

      END IF

      RETURN 
      END SUBROUTINE APPLY_BC_3_UPPER


C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_3_UPPER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      INTEGER I

C Top wall
      IF (W_BC_YMAX.EQ.0) THEN
C Dirichlet
        DO I=0,NKX
          MATL_C1(I,NY+1)=0.
          MATD_C1(I,NY+1)=1.
          MATU_C1(I,NY+1)=0.
          VEC_C1(I,NY+1)=0.

          MATL_C1(I,NY)=0.
          MATD_C1(I,NY)=1.
          MATU_C1(I,NY)=0.
          VEC_C1(I,NY)=W_BC_YMAX_C1
        END DO
      ELSE
C Neumann
        DO I=0,NKX
          MATL_C1(I,NY+1)=-1.
          MATD_C1(I,NY+1)=1.
          MATU_C1(I,NY+1)=0.
          VEC_C1(I,NY+1)=DY(NY+1)*W_BC_YMAX_C1
        END DO      
      END IF

      RETURN
      END SUBROUTINE APPLY_BC_3_UPPER_C

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      subroutine APPLY_BC_TH_LOWER(MATL1,MATD1,MATU1,VEC1,N)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      integer i,N
! Bottom Wall:
      if (TH_BC_YMIN(N).eq.0) then
! Dirichlet
        do i=0,NXM
          MATL1(i,0)=0. 
          MATD1(i,0)=1.
          MATU1(i,0)=0.                   
          VEC1(i,0)=0.

          MATL1(i,1)=0. 
          MATD1(i,1)=1.
          MATU1(i,1)=0.                   
          VEC1(i,1)=TH_BC_YMIN_C1(N)
        end do
      else
! Neumann
! NOTE: BC enforced at GY(2)
        do i=0,NXM
          MATL1(i,1)=0.
          MATD1(i,1)=-1.
          MATU1(i,1)=1.
          VEC1(i,1)=DY(2)*TH_BC_YMIN_C1(N)
        end do
        do i=0,NXM
          MATL1(i,0)=0.
          MATD1(i,0)=-1.
          MATU1(i,0)=1.
          VEC1(i,0)=DY(1)*TH_BC_YMIN_C1(N)
        end do
      end if
      RETURN
      end subroutine APPLY_BC_TH_LOWER

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      subroutine APPLY_BC_TH_LOWER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1,N)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      integer i,N
! Bottom Wall:
      if (TH_BC_YMIN(N).eq.0) then
! Dirichlet
        do i=0,NKX
          MATL_C1(i,0)=0. 
          MATD_C1(i,0)=1.
          MATU_C1(i,0)=0.                   
          VEC_C1(i,0)=0.

          MATL_C1(i,1)=0. 
          MATD_C1(i,1)=1.
          MATU_C1(i,1)=0.                   
          VEC_C1(i,1)=TH_BC_YMIN_C1(N)
        end do
      else
! Neumann
! NOTE: BC enforced at GY(2)
        do i=0,NXM
          MATL_C1(i,1)=0.
          MATD_C1(i,1)=-1.
          MATU_C1(i,1)=1.
          VEC_C1(i,1)=DY(2)*TH_BC_YMIN_C1(N)
        end do
        do i=0,NXM
          MATL_C1(i,0)=0.
          MATD_C1(i,0)=-1.
          MATU_C1(i,0)=1.
          VEC_C1(i,0)=DY(1)*TH_BC_YMIN_C1(N)
        end do

      end if
      RETURN
      END subroutine APPLY_BC_TH_LOWER_C

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      subroutine APPLY_BC_TH_UPPER(MATL1,MATD1,MATU1,VEC1,N)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      REAL*8 :: MATL1(0:NX-1,0:NY+1), MATD1(0:NX-1,0:NY+1),
     &     MATU1(0:NX-1,0:NY+1), VEC1(0:NX-1,0:NY+1)
      integer i,N
! Top wall
      if (TH_BC_YMAX(N).eq.0) then
! Dirichlet
        do i=0,NXM
          MATL1(i,NY+1)=0.
          MATD1(i,NY+1)=1.
          MATU1(i,NY+1)=0.
          VEC1(i,NY+1)=0.

          MATL1(i,NY)=0.
          MATD1(i,NY)=1.
          MATU1(i,NY)=0.
          VEC1(i,NY)=TH_BC_YMAX_C1(N)
        end do
      else
! Neumann
! NOTE: BC enforced at GY(NY)
        do i=0,NXM
          MATL1(i,NY)=-1.
          MATD1(i,NY)=1.
          MATU1(i,NY)=0.
          VEC1(i,NY)=DY(NY)*TH_BC_YMAX_C1(N)
        end do
        do i=0,NXM
          MATL1(i,NY+1)=-1.
          MATD1(i,NY+1)=1.
          MATU1(i,NY+1)=0.
          VEC1(i,NY+1)=DY(NY+1)*TH_BC_YMAX_C1(N)
        end do

      end if
      return
      end subroutine APPLY_BC_TH_UPPER

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      subroutine APPLY_BC_TH_UPPER_C(MATL_C1,MATD_C1,MATU_C1,VEC_C1,N)
C----*|--.---------.---------.---------.---------.---------.---------.-|--
      COMPLEX*16 :: VEC_C1(0:NXP,0:NY+1)
      REAL*8 :: MATL_C1(0:NXP,0:NY+1),MATD_C1(0:NXP,0:NY+1),
     &     MATU_C1(0:NXP,0:NY+1)
      integer i,N
! Top wall
      if (TH_BC_YMAX(N).eq.0) then
! Dirichlet
        do i=0,NKX
          MATL_C1(i,NY+1)=0.
          MATD_C1(i,NY+1)=1.
          MATU_C1(i,NY+1)=0.
          VEC_C1(i,NY+1)=0.

          MATL_C1(i,NY)=0.
          MATD_C1(i,NY)=1.
          MATU_C1(i,NY)=0.
          VEC_C1(i,NY)=TH_BC_YMAX_C1(N)
        end do
      else
! Neumann
! NOTE: BC enforced at GY(NY)
        do i=0,NKX
          MATL_C1(i,NY)=-1.
          MATD_C1(i,NY)=1.
          MATU_C1(i,NY)=0.
          VEC_C1(i,NY)=DY(NY)*TH_BC_YMAX_C1(N)
        end do      
        do i=0,NKX
          MATL_C1(i,NY+1)=-1.
          MATD_C1(i,NY+1)=1.
          MATU_C1(i,NY+1)=0.
          VEC_C1(i,NY+1)=DY(NY+1)*TH_BC_YMAX_C1(N)
        end do      
      end if
      return
      end subroutine APPLY_BC_TH_UPPER_C

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_VEL_LOWER
C----*|--.---------.---------.---------.---------.---------.---------.-|--
C This subroutine is called after initializing the flow
C It sets the appropriate boundary conditions including ghost cell values
C  on the velocity field in Fourier space
      INTEGER I,K      

C Now, apply the boundary conditions depending on the type specified 
      IF (U_BC_YMIN.EQ.0) THEN
C Dirichlet 
C Start with zero
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU1(I,K,1)=0.d0
           END DO
         END DO
C Now, set only the mean
         IF (RANKZ.EQ.0) CU1(0,0,1)=U_BC_YMIN_C1
C Ghost cell not used
         CU1(0,0,0)=0.d0
      ELSE IF (U_BC_YMIN.EQ.1) THEN
C Neumann
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU1(I,K,0)=CU1(I,K,1)-DY(1)*U_BC_YMIN_C1 
           END DO
         END DO
      ELSE
         STOP 'Error: U_BC_YMIN must be 0, or 1'
      END IF

      IF (W_BC_YMIN.EQ.0) THEN
C Dirichlet
C Start with zero
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU3(I,K,1)=0.d0
           END DO
         END DO
C Now, set only the mean
         IF (RANKZ.EQ.0) CU3(0,0,1)=W_BC_YMIN_C1
C Ghost cell not used
         CU3(0,0,0)=0.d0
      ELSE IF (W_BC_YMIN.EQ.1) THEN
C Neumann
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU3(I,K,0)=CU3(I,K,1)-DY(1)*W_BC_YMIN_C1 
           END DO
         END DO
      ELSE
         STOP 'Error: W_BC_YMIN must be 0, or 1' 
      END IF

      IF (V_BC_YMIN.EQ.0) THEN
C Dirichlet
C Set the vertical velocity at GYF(1) (halfway between GY(2) and GY(1))
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU2(I,K,1)=2.d0*V_BC_YMIN_C1-CU2(I,K,2)  
           END DO
         END DO
      ELSE IF (V_BC_YMIN.EQ.1) THEN
C Neumann
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU2(I,K,1)=CU2(I,K,2)-DYF(1)*V_BC_YMIN_C1 
           END DO
         END DO
      ELSE IF (V_BC_YMIN.EQ.2) THEN
C Upstream-travelling wave proposed by Speyer/Kim
C (initialize as zero)
         IF (RANKZ.EQ.0) CU2(0,0,1)=-CU2(0,0,2)
      ELSE
         STOP 'Error: V_BC_YMIN must be 0, 1, or 2'
      END IF

      RETURN
      END SUBROUTINE APPLY_BC_VEL_LOWER

C----*|--.---------.---------.---------.---------.---------.---------.-|--
      SUBROUTINE APPLY_BC_VEL_UPPER
C----*|--.---------.---------.---------.---------.---------.---------.-|--
C This subroutine is called after initializing the flow
C It sets the appropriate boundary conditions including ghost cell values
C  on the velocity field in Fourier space
      INTEGER I,K      

! Now, apply boundary conditions to the top of the domain
      IF (U_BC_YMAX.EQ.0) THEN
C Dirichlet 
C Start with zero
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU1(I,K,NY)=0.d0
           END DO
         END DO
C Now, set only the mean
         IF (RANKZ.EQ.0) CU1(0,0,NY)=U_BC_YMAX_C1
C Ghost cell not used
         CU1(0,0,NY+1)=0.d0
      ELSE IF (U_BC_YMAX.EQ.1) THEN
C Neumann
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU1(I,K,NY+1)=CU1(I,K,NY)+DY(NY+1)*U_BC_YMAX_C1
           END DO
         END DO
      ELSE
         STOP 'Error: U_BC_YMAX must be 0, or 1'
      END IF

      IF (W_BC_YMAX.EQ.0) THEN
C Dirichlet
C Start with zero
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU3(I,K,NY)=0.d0
           END DO
         END DO
C Now, set only the mean
         IF (RANKZ.EQ.0) CU3(0,0,NY)=W_BC_YMAX_C1
C Ghost cell not used
         CU3(0,0,NY+1)=0.d0
      ELSE IF (W_BC_YMAX.EQ.1) THEN
C Neumann
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU3(I,K,NY+1)=CU3(I,K,NY)+DY(NY+1)*W_BC_YMAX_C1
           END DO
         END DO
      ELSE
        STOP 'Error: W_BC_YMAX must be 0, or 1'
      END IF

      IF (V_BC_YMAX.EQ.0) THEN
C Dirichlet
C Set the vertical velocity at GYF(NY) (halfway between GY(NY) and GY(NY+1))
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU2(0,0,NY+1)=2.d0*V_BC_YMAX_C1-CU2(0,0,NY)
           END DO
         END DO
      ELSE IF (V_BC_YMAX.EQ.1) THEN
C Neumann
         DO K=0,TNKZ
           DO I=0,NXP-1
             CU2(I,K,NY+1)=CU2(I,K,NY)+DYF(NY)*V_BC_YMAX_C1
           END DO
         END DO
      ELSE IF (V_BC_YMAX.EQ.2) THEN
C Upstream-travelling wave proposed by Speyer/Kim
C (initialize as zero gradient)
         IF (RANKZ.EQ.0) CU2(0,0,NY+1)=-CU2(0,0,NY)
      ELSE
         STOP 'Error: V_BC_YMAX must be 0, 1, or 2'
      END IF

      RETURN
      END SUBROUTINE APPLY_BC_VEL_UPPER


C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
      SUBROUTINE THOMAS_REAL(A,B,C,G,NY,NX)
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|
C Uses the Thomas algorithm to solve Ax=b for tridiagonal A
C The RHS vector and solution are real
C Input lower, main, and upper diagonals, ld, md, ud, and rhs x
C Returns solution in x
C The indexing should be done by ROW, ie.
C [ b1  c1   0   0   0 ...
C [ a2  b2  c2   0   0 ...
C [  0  a3  b3   c3  0 ...

      INTEGER I, J, NX, NY
      REAL*8 A(0:NX,0:NY), B(0:NX,0:NY), C(0:NX,0:NY), G(0:NX,0:NY)

      DO J=0,NY-1
        DO I=0,NX
          A(I,J+1)=-A(I,J+1)/B(I,J)
          B(I,J+1)=B(I,J+1)+A(I,J+1)*C(I,J)
          G(I,J+1)=G(I,J+1)+A(I,J+1)*G(I,J)
        END DO
      END DO
      DO I=0,NX
        G(I,NY)=G(I,NY)/B(I,NY)
      END DO
      DO J=NY-1,0,-1
        DO I=0,NX
          G(I,J)=(G(I,J)-C(I,J)*G(I,J+1))/B(I,J)
        END DO
      END DO

      RETURN
      END SUBROUTINE THOMAS_REAL

C----*|--.---------.---------.---------.---------.---------.---------.-|-------|    
      SUBROUTINE THOMAS_COMPLEX(A,B,C,G,NY,NX)
C----*|--.---------.---------.---------.---------.---------.---------.-|-------|

C Uses the Thomas algorithm to solve Ax=b for tridiagonal A
C The RHS vector and solution is complex
C Input lower, main, and upper diagonals, ld, md, ud, and rhs x
C Returns solution in x
C The indexing should be done by ROW, ie.
C [ b1  c1   0   0   0 ...
C [ a2  b2  c2   0   0 ...
C [  0  a3  b3   c3  0 ...

      INTEGER I, J, K, NY, NX
      REAL*8 A(0:NX,0:NY), B(0:NX,0:NY), C(0:NX,0:NY)
      COMPLEX*16 G(0:NX,0:NY)

      DO J=0,NY-1
        DO I=0,NX
          A(I,J+1)=-A(I,J+1)/B(I,J)
          B(I,J+1)=B(I,J+1)+A(I,J+1)*C(I,J)
          G(I,J+1)=G(I,J+1)+A(I,J+1)*G(I,J)
        END DO
      END DO
      DO I=0,NX
        G(I,NY)=G(I,NY)/B(I,NY)
      END DO
      DO I=0,NX
        DO J=NY-1,0,-1
          G(I,J)=(G(I,J)-C(I,J)*G(I,J+1))/B(I,J)
        END DO
      END DO

      RETURN
      END SUBROUTINE THOMAS_COMPLEX

!
      END MODULE solvers_module
