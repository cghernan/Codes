import numpy as np
import h5py
from dedalus.extras import flow_tools
import time
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
lx=16.49
lz=30
nx=192
nz=576
Re=1e3
Pr=8
Aspect = 2.82842712475
Rib = 0.16
rho_ampl = 1e-3

dealias = 3/2
stop_sim_time = 500
timestepper = d3.RK222
max_timestep = 1.
dtype = np.float64

# Domain and Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, lx), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=nz, bounds=(-lz/2, lz/2), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
rho = dist.Field(name='rho', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_rho1 = dist.Field(name='tau_rho1', bases=xbasis)
tau_rho2 = dist.Field(name='tau_rho2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1
integ = lambda A: d3.Integrate(d3.Integrate(A, 'x'), 'z')
#lift_basis = zbasis.derivative_basis(1) # First derivative basis
lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_rho = d3.grad(rho) + ez*lift(tau_rho1) # First-order reduction
e_ij = d3.grad(u) + d3.transpose(d3.grad(u))

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, u, rho, tau_p, tau_rho1, tau_rho2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("dt(u) + grad(p) - div(grad_u)/Re + Rib*rho*ez + lift(tau_u2) =  -dot(u,grad(u))")
problem.add_equation("dt(rho) - div(grad_rho)/(Re*Pr) + lift(tau_rho2) = - dot(u,grad(rho))")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Boundary conditions
problem.add_equation("dot(ez, dot(ex,e_ij))(z=-lz/2) = 0")
problem.add_equation("dot(ez, u)(z=-lz/2) = 0")
problem.add_equation("rho(z=-lz/2) = 2")

problem.add_equation("dot(ez, dot(ex,e_ij))(z=lz/2) = 0")
problem.add_equation("dot(ez, u)(z=lz/2) = 0")
problem.add_equation("rho(z=lz/2) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
logger.info('Solver built')

# Initial conditions
u['g'][0] = np.tanh(z)
rho['g'] = (1-np.tanh(Aspect*z)) 

# Add small velocity perturbations localized to the shear layers
# use vector potential to ensure div(u) = 0
A = dist.Field(name='A', bases=(xbasis,zbasis))
A.fill_random('g', seed=42, distribution='normal')
A.low_pass_filter(scales=(0.2, 0.2, 0.2))
A['g'] *= (1 - (2*z/lz)**2) *np.exp(-z**2) # Damp noise at walls

up = d3.skew(d3.grad(A)).evaluate()
up.change_scales(1)
u['g'] += 1e-3*up['g'] 

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1., max_writes=10)
snapshots.add_task(rho, name='density')
snapshots.add_task(u, name='velocity')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=1e-3, cadence=10, safety=0.2, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
if Aspect != 0:
    CFL.add_frequency((Rib*((d3.grad(rho)*ez)**2)**0.5)**0.5)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(0.5*d3.dot(u,u)), name='KE')
flow.add_property(d3.dot(u,ez)**2, name='w2')
flow.add_property(d3.div(u), name='divu')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    start_time = time.time()
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            max_divu = flow.max('divu')
            max_KE = flow.max('KE')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f, max(div(u))=%e, max(KE)=%f' %(solver.iteration, solver.sim_time, timestep, max_w, max_divu, max_KE))
            
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
    end_time = time.time()

# Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Final KE: %f' %(flow.max('KE')))
