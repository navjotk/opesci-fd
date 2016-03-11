from opesci import *
from os import path
from argparse import ArgumentParser, RawTextHelpFormatter
from sympy.abc import *
import numpy as np
from sympy.functions.special.delta_functions import Heaviside
from sympy import floor
_test_dir = path.join(path.dirname(__file__), "src")


def simplewave3d(domain_size, grid_size, dt, tmax, output_vts=False, o_converge=True,
                 accuracy_order=[1, 2, 2, 2],
                 omp=True, simd=False, ivdep=True, double=False, pluto=False,
                 filename='test.cpp', expand=True, eval_const=True,
                 fission=False):
    """
    create 3D eigen waves and run FD simulation
    :param domain_size: define size of domain
    e.g. (1.0, 1.0, 1.0) for unit cube
    :param grid_size: define grid size, e.g. (10, 10, 10)
    :param dt: define time step
    :param tmax: define simulation time
    :param output_vts: switch for code to output at each time step
    (such as save field as vtk file), default False (no output)
    :param o_converge: switch for code to calculate L2 norms
    for convergence test, default True (output)
    :param omp: swtich for inserting #pragma omp for before outer loop
    default True (use omp)
    :param simd: switch for inserting #pragma simd before inner loop
    default False (do not use simd)
    :param ivdep: switch for inserting #praga ivdep before inner loop
    default True (use ivdep)
    default False (not include vtk header files)
    :param double: switch for using double as real number variables
    default False (use float)
    :param expand: expand kernel fully (no factorisation), default True
    :param eval_const: evaluate all constants in kernel in generated code default True
    :param filename: output source code file name as string
    :param pluto: switch for inserting #pragma scop and #pragma endscop for
    pluto optimisation
    :param fission: switch for doing loop fission optimisation
    """

    print 'domain size: ' + str(domain_size)
    print 'grid size: ' + str(grid_size)
    print 'approximation order: ' + str(accuracy_order)
    print 'dt: ' + str(dt)
    print 'tmax: ' + str(tmax)

    # Declare fields
    

    grid = RegularGrid(dimension=3, domain_size=domain_size, grid_size=grid_size, pluto=pluto, fission=fission)
    grid.set_time_step(dt, tmax)
    grid.set_switches(omp=omp, simd=simd, ivdep=ivdep, double=double,
                      expand=expand, eval_const=eval_const,
                      output_vts=output_vts, converge=o_converge)
    # define parameters
    # const_c: The constant from the wave equation
    t, x, y, z, const_c = symbols('_t x y z c')
    grid.set_index([x, y, z])
    # Set the values of the constant C from the wave equation, and the initial velocity (single scalar value for all dimensions=0)
    grid.set_params(c=2, v=1)

    print 'require dt < ' + str(grid.get_time_step_limit())
    dtt = grid.get_time_derivative(2, 2)
    dt = grid.get_time_derivative(1, 2)
    dxx, dyy, dzz = grid.get_space_derivatives(2, 2)
    global M, Q, D, E
    m,s,h = symbols('m s h')
    m=M(x,y)
    q=Q(x,y,t)
    d=D(x,y,t)
    e=E(x,y)
    wave_equation = m*dtt- (dxx+dyy) - q  + e*dt
    
    
    hstep=25 #space increment d  = minv/(10*f0);
    tstep=2 #time increment dt < .5 * hstep /maxv;
    tmin=0.0 #initial time
    tmax=600 #simulate until
    xmin=-500.0 - 10*hstep #left bound
    xmax=500.0 + 10*hstep #right bound...assume packet never reaches boundary
    ymin=-500.0 - 10*hstep #left bound
    ymax=500.0 + 10*hstep #right bound...assume packet never reaches boundary
    f0=.010
    t0=1/.010
    nbpml=10
    nx = int((xmax-xmin)/hstep) + 1 #number of points on x grid
    ny = int((ymax-ymin)/hstep) + 1 #number of points on x grid
    nt = int((tmax-tmin)/tstep) + 2 #number of points on t grid
    xsrc=-400
    ysrc=0.0
    xrec = nbpml+4
    

    grid.set_order(accuracy_order)

    # The equation involves second-order derivatives, so instruct the class not to drop second derivatives from the analysis
    grid.calc_derivatives(2)
    
    #velocity_function = 3 + Heaviside(x-floor(nx/2))*1.5
    #grid.set_velocity_initialisation_function(mt)
    grid.solve_fd([wave_equation], m, q, e)
    print 'Kernel AI'
    print '%.2f, %.2f (weighted), %d ADD, %d MUL, %d LOAD, %d STORE' % grid.get_kernel_ai()
    return grid


def default(compiler=None, execute=False, nthreads=1,
            accuracy_order=[2, 4, 4, 4],
            output=False, profiling=False, papi_events=[], pluto=False, tile=' ', fission=False):
    """
    Eigenwave test case on a unit cube grid (100 x 100 x 100)
    """
    if pluto:
        f = open('tile.sizes', 'w')
        f.write(str(tile))
        f.close()
    domain_size = (1.0, 1.0, 1.0)
    grid_size = (100, 100, 100)
    dt = 0.002
    tmax = 1.0
    filename = path.join(_test_dir, 'regular3d.cpp')
    grid = simplewave3d(domain_size, grid_size, dt, tmax,
                        accuracy_order=accuracy_order,
                        o_converge=True, omp=True, simd=False,
                        ivdep=True, filename=filename, pluto=pluto, fission=fission)
    grid.set_switches(output_vts=output, profiling=profiling)
    grid.set_papi_events(papi_events)
    out = None
    if pluto:
        grid.generate(filename)
        filename_p = grid.pluto_op(filename)
        if compiler in ['clang', 'clang++']:
            # an ugly fix, but pluto will always attack <omp.h> to the first line
            # which would fail clang
            with open(filename_p, 'r') as fin:
                data = fin.read().splitlines(True)
            with open(filename_p, 'w') as fout:
                fout.writelines(data[1:])
        grid.src_file = filename_p
        if compiler is None:
            share = True
        else:
            share = False
            out = grid.compile(filename_p, compiler=compiler, shared=share)
    else:
        filename_p = filename
        if compiler is None:
            grid.generate(filename_p)
        else:
            out = grid.compile(filename_p, compiler=compiler, shared=False)
    hstep=25 #space increment d  = minv/(10*f0);
    tstep=2 #time increment dt < .5 * hstep /maxv;
    tmin=0.0 #initial time
    tmax=600 #simulate until
    xmin=-500.0 - 10*hstep #left bound
    xmax=500.0 + 10*hstep #right bound...assume packet never reaches boundary
    ymin=-500.0 - 10*hstep #left bound
    ymax=500.0 + 10*hstep #right bound...assume packet never reaches boundary
    f0=.010
    t0=1/.010
    nbpml=10
    nx = int((xmax-xmin)/hstep) + 1 #number of points on x grid
    ny = int((ymax-ymin)/hstep) + 1 #number of points on x grid
    nt = int((tmax-tmin)/tstep) + 2 #number of points on t grid
    vel=np.ones((nx,ny)) + 2.0
    vel[floor(nx/2):nx,:]=4.5
    mt=vel**-2
    if execute:
        # Test Python-based execution for the base test
        grid.execute(filename_p, mt, compiler=compiler, nthreads=nthreads)
        grid.convergence()
    return out


def cx1():
    """
    test case for comparison between pragma simd and pragma ivdep on cx1
    """
    domain_size = (1.0, 1.0, 1.0)
    grid_size = (200, 200, 200)
    dt = 0.001
    tmax = 5.0
    simplewave3d(domain_size, grid_size, dt, tmax, output_vts=False, o_converge=False,
                 omp=True, simd=False, ivdep=True,
                 filename=path.join(_test_dir, 'simplewave3d_ivdep.cpp'))
    simplewave3d(domain_size, grid_size, dt, tmax, output_vts=False, o_converge=False,
                 omp=True, simd=True, ivdep=False,
                 filename=path.join(_test_dir, 'simplewave3d_simd.cpp'))


def converge_test():
    """
    - test case for convergence analysis of (2,4) scheme
    - eigen wave in unit cube
    - start with spacing 0.1, spacing halves for each test
    - dt reduce by factor of 4 for each test
    """
    domain_size = (1.0, 1.0, 1.0)
    s = 10
    c = 0.4*s
    dt = c/(s**2)
    tmax = 5.0
    simplewave3d(domain_size, (s, s, s), dt, tmax, output_vts=False, o_converge=True,
                 omp=True, simd=False, ivdep=True,
                 filename='tmp/test3d_'+str(s)+'.cpp')

    s = s*2
    dt = c/(s**2)
    simplewave3d(domain_size, (s, s, s), dt, tmax, output_vts=False, o_converge=True,
                 omp=True, simd=False, ivdep=True,
                 filename='tmp/test3d_'+str(s)+'.cpp')

    s = s*2
    dt = c/(s**2)
    simplewave3d(domain_size, (s, s, s), dt, tmax, output_vts=False, o_converge=True,
                 omp=True, simd=False, ivdep=True,
                 filename='tmp/test3d_'+str(s)+'.cpp')

    s = s*2
    dt = c/(s**2)
    simplewave3d(domain_size, (s, s, s), dt, tmax, output_vts=False, o_converge=True,
                 omp=True, simd=False, ivdep=True,
                 filename='tmp/test3d_'+str(s)+'.cpp')


def main():
    ModeHelp = """Avalable testing modes:
default:   Simple acoustic wave test case on a unit cube grid (100 x 100 x 100)

converge:  Convergence test of the (2,4) scheme, which is 2nd order
           in time and 4th order in space. The test halves spacing
           starting from 0.1 and reduces dt by a factor of 4 for
           each step
"""
    p = ArgumentParser(description="Standalone testing script for the Simple acoustic wave example",
                       formatter_class=RawTextHelpFormatter)
    p.add_argument('mode', choices=('default', 'converge', 'cx1'),
                   nargs='?', default='default', help=ModeHelp)
    p.add_argument('-so', '--spatial_order', default=4, type=int, dest='so',
                   help='order of the spatial discretisation to use for code generation * 2, eg. order=4 to use 4th order approximation in x,y,z')
    p.add_argument('-c', '--compiler', default=None,
                   help='C++ Compiler to use for model compilation, eg. g++ or icpc')
    p.add_argument('-x', '--execute', action='store_true', default=False,
                   help='Dynamically execute the generated model')
    p.add_argument('-n', '--nthreads', type=int, default=1,
                   help='Number of threads for dynamic execution')
    p.add_argument('-o', '--output', action='store_true', default=False,
                   help='Activate solution output in .vts format')
    p.add_argument('-p', '--profiling', action='store_true', default=False,
                   help='Activate performance profiling from PAPI')
    p.add_argument('--papi-events', dest='papi_events', nargs='+', default=[],
                   help='Specific PAPI events to measure')
    p.add_argument('--tile', default=None,
                   help="tile-size for pluto optimisation e.g. --tile '4 4 32'")
    p.add_argument('--pluto', action='store_true', default=False,
                   help="Apply pluto optimisation ")
    p.add_argument('--fission', action='store_true', default=False,
                   help="Apply loop fission optimisation")

    args = p.parse_args()
    print "Simple wave 3D example "

    if args.mode == 'default':
        default(compiler=args.compiler, execute=args.execute,
                nthreads=args.nthreads, output=args.output,
                accuracy_order=[2, args.so, args.so, args.so],
                profiling=args.profiling, papi_events=args.papi_events,
                pluto=args.pluto, tile=args.tile, fission=args.fission)
    elif args.mode == 'converge':
        converge_test()
    elif args.mode == 'cx1':
        cx1()
if __name__ == "__main__":
    main()
