# This is the makefile for diablo.
# To compile the code, just type make.  Such an approach makes
# recompilation of the code easy, recompiling only as necessary
# to account for recent changes to the code.
#
# As the user, set the following definitions:

#**********   User Defined Variables Below *********
# Fortran 90 complier to use:
COMPILER = ifort

# Any desired options for the compiler (e.g. -O2, -g, etc.)
#USEROPTS = -O3 -ftree-vectorize
USEROPTS = -O3 -fpp -mcmodel=large
FCOMP = -fpic -i_dynamic -mcmodel=large
LINK = -fpic -i_dynamic -mcmodel=large8.15-parallel
# Location where fftw and netcdf (optional) libraries are installed
LINKDIR = -L/apps/fftw/2.1.5-double/lib -L/apps/hdf5/1.8.15/parallel/lib
#
# # Location where the optional netcdf include file (netcdf.inc) is installed
INCLUDEDIR = -I/apps/fftw/2.1.5-double/include -I/apps/hdf5/1.8.15/parallel/include

# Option to compile with MPI libraries
PARALLEL = TRUE

# Option to enable the LES model (loads the required variables into memory)
LES = TRUE

# Option to compile with the NetCDF libraries
NETCDF = FALSE
HDF5 = TRUE

# Option to compile with newton solver
NEWTON = FALSE

# Option to run different flavors (basic, ensemble, etc.)
ENSEM = FALSE
BATCH = FALSE
# **********    END of user definitions ************

ifeq ($(ARCH),ifort)
COMPILER = ifort
USEROPTS = -O3 -fpp
else ifeq ($(ARCH),gfortran)
COMPILER = gfortran
USEROPTS = -O3 -cpp
endif

ifeq ($(LES),TRUE)
LES_o = les.o
else
LES_o = no_les.o
endif

ALL2ALL=1

ifeq ($(PARALLEL),TRUE)
COMPILER = mpif90 
MPI = mpi.o
MPIF = mpi.alltoall$(ALL2ALL).F
ifeq ($(HDF5),TRUE)
HDF5_o = hdf5.o 
COMPILER = h5pfc
HDF5OPTS=-DHDF5
endif
ifeq ($(NEWTON),TRUE)
NEWTONOPTS=-DNEWTON
endif
else
ifeq ($(HDF5),TRUE)
HDF5_o = hdf5s.o 
COMPILER = h5pfc
HDF5OPTS=-DHDF5
endif
MPI = mpi_serial.o
endif

MAIN = diablo.f
HEADER = header
ENSEM_HOOKS = dummy_code/ensem_dummy.f
BATCH_HOOKS = dummy_code/batch_dummy.f
HOOKS = batch_hooks.o ensem_hooks.o
ADJOINT = 

ifeq ($(ENSEM),TRUE)
MAIN = ensemble.f
HEADER = header header_ensem
COMPILER = mpif90
ENSEM_HOOKS = ensem_hooks.f
endif

ifeq ($(BATCH),TRUE)
MAIN = batch.f
HEADER = header header_batch
BATCH_HOOKS = batch_hooks.f
#ADJOINT = adj_chan.o adj_per.o
ADJOINT = adj_per.o
endif

# Use the parameters to set flags
ifeq ($(NETCDF),TRUE)
COMPOPTS = $(USEROPTS) $(HDF5OPTS) $(NEWTONOPTS) $(INCLUDEDIR)
LINKOPTS = $(LINKDIR) -ldrfftw -ldfftw -lnetcdf
NETCDF_o = netcdf.o
else
COMPOPTS = $(USEROPTS) $(HDF5OPTS) $(NEWTONOPTS)
LINKOPTS = $(LINKDIR) -lrfftw -lfftw \
	./lib/liblapack_ifort64.a \
	./lib/libblas_ifort64.a
NETCDF_o = no_netcdf.o
endif



diablo: $(MAIN) diablo_io.o channel.o $(LES_o) \
	fft.o newton.o shared.o solvers.o $(MPI) \
	$(HDF5_o)
	$(COMPILER) $(COMPOPTS) $(MAIN) -o diablo \
	diablo_io.o channel.o $(LES_o) \
	fft.o newton.o shared.o solvers.o \
	$(MPI) $(LINKOPTS) $(HDF5_o)

shared.o: shared.F grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c shared.F

solvers.o: solvers.F shared.o grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c solvers.F

diablo_io.o: diablo_io.F shared.o channel.o solvers.o fft.o $(HDF5_o)
	$(COMPILER) $(COMPOPTS) -c diablo_io.F

#periodic.o: periodic.f fft.o header grid_def
#	$(COMPILER) $(COMPOPTS) -c periodic.f

channel.o: channel.F solvers.o fft.o $(MPI) shared.o $(LES_o) grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c channel.F

ifeq ($(LES),TRUE) 
les.o: les.F fft.o shared.o $(MPI) $(HDF5_o) grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c les.F
else
no_les.o: dummy_code/no_les.f fft.o shared.o $(MPI) $(HDF5_o) grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c dummy_code/no_les.f
endif

ifeq ($(NETCDF),TRUE)
netcdf.o: netcdf.f header
	$(COMPILER) $(COMPOPTS) -c netcdf.f
else
no_netcdf.o: dummy_code/no_netcdf.f 
	$(COMPILER) $(COMPOPTS) -c dummy_code/no_netcdf.f
endif

ifeq ($(PARALLEL),TRUE)
mpi.o: $(MPIF) shared.o solvers.o grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c -o mpi.o $(MPIF) 
else
mpi_serial.o: dummy_code/mpi_serial.f shared.o solvers.o grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c dummy_code/mpi_serial.f
endif

header : header_mpi grid_def

header_mpi : grid_mpi

hdf5.o : hdf5.F shared.o fft.o $(MPI) grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c hdf5.F

hdf5s.o : hdf5s.f shared.o fft.o $(MPI) grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c hdf5s.f

#duct.o: duct.f header grid_def
#	$(COMPILER) $(COMPOPTS) -c duct.f

#cavity.o: cavity.f header grid_def
#	$(COMPILER) $(COMPOPTS) -c cavity.f

#ensem_hooks.o: $(ENSEM_HOOKS) header header_ensem grid_def
#	$(COMPILER) $(COMPOPTS) -c $(ENSEM_HOOKS) -o ensem_hooks.o

#batch_hooks.o: $(BATCH_HOOKS) header header_batch grid_def
#	$(COMPILER) $(COMPOPTS) -c $(BATCH_HOOKS) -o batch_hooks.o

#ifeq ($(BATCH),TRUE)
#adj_chan.o: adj_chan.f header header_batch grid_def
#	$(COMPILER) $(COMPOPTS) -c adj_chan.f

#adj_per.o: adj_chan.f header header_batch grid_def
#	$(COMPILER) $(COMPOPTS) -c adj_per.f
#endif

fft.o:  fft.F shared.o $(MPI) grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c fft.F

#rand.o:  rand.f grid_def grid_mpi
#	$(COMPILER) $(COMPOPTS) -c rand.f

newton.o: newton.F90 $(LES_o) $(HDF5_o) $(MPI) fft.o shared.o solvers.o \
	diablo_io.o grid_def grid_mpi NewtonHook.o GMRESm.o
	$(COMPILER) $(COMPOPTS) -c newton.F90

NewtonHook.o: NewtonHook.F90 $(LES_o) $(HDF5_o) $(MPI) fft.o shared.o solvers.o \
	diablo_io.o grid_def grid_mpi GMRESm.o
	$(COMPILER) $(COMPOPTS) -c NewtonHook.F90

GMRESm.o: GMRESm.F90 $(LES_o) $(HDF5_o) $(MPI) fft.o shared.o solvers.o \
	diablo_io.o grid_def grid_mpi
	$(COMPILER) $(COMPOPTS) -c GMRESm.F90

clean:
	rm -f *.o fort.* *~ diablo core *.mod

# Compiler specific notes:
#
# Compilation with Absoft Linux Fortran 77 appears to be impossible, as it
# cannot handle the INTEGER*8 option required by FFTW.  If someone finds
# a way around this, please let me know.
# 
# Compilation with Absoft Linux Fortran 90 is possible, but the option
# -YEXT_NAMES=LCS must be used as one of the link options so the compiler
# can find the lowercase external library function names.
#
# Compilation with Lahey Fortran 95 (lf95) is possible, but there is an
# underscore incompatability with the FFTW libraries, which are compiled
# with g77.  To get around this, you need to go into fft.f and add 
# trailing underscores to the name of every fftw function where they
# appear throughout the code.

