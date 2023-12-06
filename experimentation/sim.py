import rebound
import numpy as np
import matplotlib.pyplot as plt


sim = rebound.Simulation()
np.random.seed(42)

# integrator options
sim.integrator = "mercurius"
sim.dt = 1
sim.testparticle_type = 1

# collision and boundary options
sim.collision = "direct"
sim.collision_resolve = "merge"
sim.collision_resolve_keep_sorted = 1
sim.boundary = "open"
boxsize = 200.0
sim.configure_box(boxsize)
sim.track_energy_offset = 1

# simulation time
tmax = 1e4

# massive bodies
sim.add(m=1.0, r=0.005)  # Sun

# sim.add(
#     m=3.3011e23 * 5.02785e-31,
#     r=2439.7 * 6.68459e-9,
#     a=0.307499,
#     e=0.205630,
#     omega=48.331 / 180 * np.pi,
#     theta=0,
# )  # Mercury

# sim.add(
#     m=4.8675e24 * 5.02785e-31,
#     r=6051.8 * 6.68459e-9,
#     a=0.723332,
#     e=0.006772,
#     omega=54.884 / 180 * np.pi,
#     theta=0,
# )  # Venus

sim.add(
    m=5.972168e24 * 5.02785e-31,
    r=6371.0 * 6.68459e-9,
    a=1.0003,
    e=0.0167086,
    omega=114.20783 / 180 * np.pi,
    theta=0,
)  # Earth

sim.add(
    m=6.4171e23 * 5.02785e-31,
    r=3389.5 * 6.68459e-9,
    a=1.52368055,
    e=0.0934,
    omega=286.5 / 180 * np.pi,
    theta=0,
)  # Mars

sim.add(
    m=1.8982e27 * 5.02785e-31,
    r=69911 * 6.68459e-9,
    a=5.2038,
    e=0.0489,
    omega=273.867 / 180 * np.pi,
    theta=0,
)  # Jupiter

sim.add(
    m=5.6834e26 * 5.02785e-31,
    r=58232 * 6.68459e-9,
    a=9.5826,
    e=0.0565,
    omega=339.392 / 180 * np.pi,
    theta=0,
)  # Saturn

sim.add(
    m=8.6810e25 * 5.02785e-31,
    r=25362 * 6.68459e-9,
    a=19.19126,
    e=0.04717,
    omega=96.998857 / 180 * np.pi,
    theta=0,
)  # Uranus

sim.add(
    m=1.02413e26 * 5.02785e-31,
    r=24622 * 6.68459e-9,
    a=30.07,
    e=0.008678,
    omega=273.187 / 180 * np.pi,
    theta=0,
)  # Neptune

sim.add(
    m=1.303e22 * 5.02785e-31,
    r=1188.3 * 6.68459e-9,
    a=39.482,
    e=0.2488,
    omega=113.834 / 180 * np.pi,
    theta=0,
)  # Pluto


# sim.add(m=5e-5, a=49.305, e=0.2488, omega=3.1415, theta=3.1415)  # Pluto

# simulation add particle notes
# m = mass,
# r = radius
# a = semi=major axis (average orbital radius)
# e = eccentricity (how elipitcal an orbit is)
# omega = argument of periapsis/perihelion (angle to the closest point of orbit, relative to +x)
# f = angle of the starting position of the satellite to the perihelion
# theta = starting angle of the satelitle relative to the +x axis (use either f or theta, not both)

spawnradius = 10

sim.N_active = sim.N


# exit()

# semi-active bodies
n_comets = 3
a = np.random.random(n_comets) * 10 + spawnradius
e = np.random.random(n_comets) * 0.009 + 0.4
inc = np.random.random(n_comets) * 0
m = 1e-10
r = 1e-7

for i in range(0, n_comets):
    rand = np.random.random() * 2 * np.pi
    sim.add(m=m, r=r, a=a[i], e=e[i], inc=inc[i], Omega=0, omega=rand, f=rand)

sim.move_to_com()
E0 = sim.energy()


op = rebound.OrbitPlot(sim, Narc=300)
# plt.show()

for i in range(300):
    sim.integrate(sim.t + 10)
    op.update()
    # op.fig.savefig("./anim/out_%03d.png" % i)

#
# sim.integrate(tmax)
# dE = abs((sim.energy() - E0) / E0)
# print(dE)

# op = rebound.OrbitPlot(sim, Narc=300)
# plt.show()

# make Anim,
# ffmpeg -framerate 10 -pattern_type glob -i 'anim/*.png'   -c:v libx264 -pix_fmt yuv420p out.mp4
#  convert -delay 6 -quality 95 anim/*.png movie.mp4s
