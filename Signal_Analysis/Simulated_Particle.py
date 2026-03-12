import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'



def simulated_particle(P_size, P_Speed, Inc_Angle, Laser_Lambda, Po, T_impact, S_l, Time_max, Adq_Freq, m0):
    """
    Simulate the signal of a single particle based on Optical Feedback Interferometry.

    Parameters
    -----------
    P_size      : Diameter of the particle [m]
    P_Speed     : Particle speed [m/s]
    Inc_Angle   : Incident angle [degrees]
    Laser_Lambda: Laser wavelength [m]
    Po          : Laser power [mV]
    T_impact    : Time when particle center crosses beam center [s]
    S_l         : Laser beam spot diameter [m]
    Time_max    : Number of samples
    Adq_Freq    : Acquisition frequency [Hz]
    m0          : Modulation index

    Returns
    -----------
    P_t, t      : Signal vector [mV], Time vector [s]
    """
    # Convert angle from degrees to radians
    Inc_Angle_rad = np.radians(Inc_Angle)
    Inc_Angle_rad_2 = np.radians(90-Inc_Angle)

    # Time vector in seconds
    t = np.linspace(0, Time_max / Adq_Freq, Time_max)

    # Doppler frequency: f_D = 2 * V * sin(theta) / lambda
    f_D = (2 * P_Speed * np.sin(Inc_Angle_rad_2)) / Laser_Lambda
    print('doppler:',f_D)

    print('P_Speed:', P_Speed)

    # Transit time:
    tau = (P_size + S_l) / (P_Speed * np.sin(Inc_Angle_rad))
    print('tau:', tau)
    #tau =  ((P_size + S_l*0.05) / (P_Speed) ) # adjustd factor trying to correalte the effective sensing volumen of the laser beam

    # Signal: P_f(t) = P_0 * [1 + m_0 * cos(2 * pi * f_D * t)] * exp(-((t - t_0)^2) / (2 * tau^2))
    modulation = 1 + m0 * np.cos(2 * np.pi * f_D * t)
    envelope = np.exp(-((t - T_impact)**2) / (2 * tau**2))
    P_t = Po * modulation * envelope

    return P_t, t

# Define parameters
D_Particle = 2e-6      # Particle diameter [m]
P_Speed = 0.026      # Particle speed [m/s], set for f_D ≈ 20 kHz
Theta = 80             # Incident angle [degrees]
Laser_lambda = 1550e-9 # Laser wavelength [m]
Po = 0.004134*4          # Laser power [mV]
T_impact = 0.0007       # Time of impact [s], centered at 6 ms / 2
S_l = 70e-6             # Laser spot diameter [m], adjusted to 6 µm
Time_max = 3000       # Number of samples, for 6 ms window
Adq_Freq = 2e6         # Acquisition frequency [Hz], fixed
M0_init = 10.0          # Initial modulation index 

# Initial signal calculation
P_t, t = simulated_particle(
    P_size=D_Particle,
    P_Speed=P_Speed,
    Inc_Angle=Theta,
    Laser_Lambda=Laser_lambda,
    Po=Po,
    T_impact=T_impact,
    S_l=S_l,
    Time_max=Time_max,
    Adq_Freq=Adq_Freq,
    m0=M0_init
)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.15, bottom=0.35)  # Space for three sliders
line, = ax.plot(t * 1000, P_t, label=r'$P_f(t)$', color='blue', linewidth=2)  # Time in ms

ax.set_xlabel('Time (ms)', fontsize=20)
ax.set_ylabel('Power (mV)', fontsize=20)
ax.set_title('(a)', fontsize=24, pad=15)
ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
ax.legend(fontsize=18)
ax.grid(True, linestyle='--', alpha=0.7)

# Add sliders
axcolor = 'lightgoldenrodyellow'
ax_speed = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_d_particle = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_m0 = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_sl = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

s_speed = Slider(ax_speed, 'Speed (m/s)', 0.01, 0.2, valinit=P_Speed, valstep=0.001)
s_d_particle = Slider(ax_d_particle, 'D Particle (µm)', 1, 20, valinit=D_Particle * 1e6, valstep=0.1)
s_m0 = Slider(ax_m0, 'm0', 5, 20, valinit=M0_init, valstep=0.1)
s_sl = Slider(ax_sl, 'sl', 0, 70e-6, valinit=S_l, valstep=1e-6)


# Update function for sliders
def update(val):
    P_speed_new = s_speed.val
    D_particle_new = s_d_particle.val * 1e-6  # Convert µm back to m
    m0_new = s_m0.val
    sl_new= s_sl.val
    P_t_new, t_new = simulated_particle(
        P_size=D_particle_new,
        P_Speed=P_speed_new,
        Inc_Angle=Theta,
        Laser_Lambda=Laser_lambda,
        Po=Po,
        T_impact=T_impact,
        S_l=sl_new,
        Time_max=Time_max,
        Adq_Freq=Adq_Freq,
        m0=m0_new
    )
    line.set_ydata(P_t_new)
    line.set_xdata(t_new * 1000)  # Time in ms
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Rescale view
    fig.canvas.draw_idle()

# Connect sliders to update function
s_speed.on_changed(update)
s_d_particle.on_changed(update)
s_m0.on_changed(update)
s_sl.on_changed(update)

plt.show()