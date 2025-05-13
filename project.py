import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import messagebox

def run_simulation(equilibrium_override=False):
    try:
        m = float(entry_m.get())
        g = float(entry_g.get())
        c = float(entry_c.get())
        l0 = float(entry_l0.get())
        omega = float(entry_omega.get())
        alpha_deg = float(entry_alpha.get())
        alpha = np.radians(alpha_deg)

        critical_stiffness = m * omega ** 2 * np.sin(alpha) ** 2
        if c > critical_stiffness:
            stability_comment = f"Устойчивое равновесие (c > mω²sin²α)"
            stable = True
        elif c == critical_stiffness:
            stability_comment = f"Нейтральное равновесие (c = mω²sin²α)"
            stable = False
        else:
            stability_comment = f"Неустойчивое равновесие (c < mω²sin²α)"
            stable = False

        denominator = c - m * omega ** 2 * np.sin(alpha) ** 2
        equilibrium = (c * l0 - m * g * np.cos(alpha)) / denominator if denominator != 0 else l0

        if equilibrium_override:
            if stable:
                y0 = [equilibrium, 0]
            else:
                y0 = [equilibrium + 0.001, 0]  # Маленькое смещение
            t_span = (0, 5) if stable else (0, 15)
        else:
            y0 = [float(entry_l_init.get()), float(entry_v_init.get())]
            t_span = (0, 30)

        def dynamics(t, y):
            l, v = y
            f_spring = -c * (l - l0)
            f_gravity = -m * g * np.cos(alpha)
            f_centrifugal = m * omega ** 2 * l * np.sin(alpha) ** 2
            a = (f_spring + f_gravity + f_centrifugal) / m
            return [v, a]

        t_eval = np.linspace(*t_span, 2000)
        sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval)

        def U(l):
            return 0.5 * c * (l - l0) ** 2 + m * g * l * np.cos(alpha) - 0.5 * m * omega ** 2 * l ** 2 * np.sin(alpha) ** 2

        l_vals = np.linspace(0.1, 5, 500)
        U_vals = U(l_vals)

        plt.close('all')
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_title(f'Движение кольца\n{stability_comment}\nПоложение равновесия: {equilibrium:.2f} м')

        rod_length = 5
        rod_up_line, = ax1.plot([], [], [], 'k-', lw=3)
        rod_down_line, = ax1.plot([], [], [], 'k-', lw=3)
        ring, = ax1.plot([], [], [], 'ro', markersize=10)
        spring_line, = ax1.plot([], [], [], 'orange', lw=2)

        ax1.set_xlim3d(-5, 5)
        ax1.set_ylim3d(-5, 5)
        ax1.set_zlim3d(0, 5)
        ax1.set_box_aspect([1, 1, 1])

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("Фазовый портрет: v(l)")
        ax2.set_xlabel("Положение l (м)")
        ax2.set_ylabel("Скорость v (м/с)")
        ax2.set_xlim(0, 5)
        ax2.set_ylim(-10, 10)
        ax2.grid(True)
        phase_line, = ax2.plot([], [], 'r')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("Положение l(t)")
        ax3.set_xlabel("Время (с)")
        ax3.set_ylabel("l (м)")
        ax3.set_xlim(t_span)
        ax3.set_ylim(0, 5)
        ax3.grid(True)
        position_line, = ax3.plot([], [], 'b')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("Потенциальная энергия U(l)")
        ax4.set_xlabel("Положение l (м)")
        ax4.set_ylabel("U (Дж)")
        ax4.plot(l_vals, U_vals, 'm', label='U(l)')
        eq_energy = U(equilibrium)
        ax4.axvline(equilibrium, color='k', linestyle='--', label=f'Равновесие\nU={eq_energy:.2f} Дж')
        ax4.legend()
        ax4.set_xlim(0, 5)
        ax4.grid(True)

        def create_spring(start, end, turns=15, radius=0.05):
            vec = np.array(end) - np.array(start)
            length = np.linalg.norm(vec)
            t = np.linspace(0, 1, turns * 20)
            x = start[0] + vec[0] * t + radius * np.sin(2 * np.pi * turns * t) * (-vec[1])
            y = start[1] + vec[1] * t + radius * np.sin(2 * np.pi * turns * t) * (vec[0])
            z = start[2] + vec[2] * t
            return x, y, z

        def update(frame):
            theta = omega * sol.t[frame]
            l = sol.y[0][frame]

            x = l * np.sin(alpha) * np.cos(theta)
            y_ = l * np.sin(alpha) * np.sin(theta)
            z = l * np.cos(alpha)

            rod_vector = np.array([np.sin(alpha) * np.cos(theta), np.sin(alpha) * np.sin(theta), np.cos(alpha)])

            up_end = rod_vector * rod_length
            down_end = np.array([0, 0, 0])

            rod_up_line.set_data([0, up_end[0]], [0, up_end[1]])
            rod_up_line.set_3d_properties([0, up_end[2]])

            rod_down_line.set_data([0, down_end[0]], [0, down_end[1]])
            rod_down_line.set_3d_properties([0, down_end[2]])

            ring.set_data([x], [y_])
            ring.set_3d_properties([z])

            x_spring, y_spring, z_spring = create_spring([0, 0, 0], [x, y_, z])
            spring_line.set_data(x_spring, y_spring)
            spring_line.set_3d_properties(z_spring)

            phase_line.set_data(sol.y[0][:frame], sol.y[1][:frame])
            position_line.set_data(sol.t[:frame], sol.y[0][:frame])

            return rod_up_line, rod_down_line, ring, spring_line, phase_line, position_line

        ani = FuncAnimation(fig, update, frames=len(t_eval), interval=int(animation_speed.get()), blit=True)
        plt.show()

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка ввода данных:\n{e}")

def go_to_equilibrium():
    run_simulation(equilibrium_override=True)

root = tk.Tk()
root.title("Параметры модели")

params_frame = tk.Frame(root)
params_frame.pack(pady=10)

fields = [
    ("Масса m (кг)", "1"),
    ("g (м/с²)", "9.81"),
    ("Жесткость c (Н/м)", "300"),
    ("Длина пружины l0 (м)", "1"),
    ("Угловая скорость ω (рад/с)", "5"),
    ("Угол α (градусы)", "45"),
    ("Начальное положение l (м)", "1.5"),
    ("Начальная скорость (м/с)", "0"),
]

entries = []
for label, default in fields:
    tk.Label(params_frame, text=label).pack()
    e = tk.Entry(params_frame)
    e.insert(0, default)
    e.pack()
    entries.append(e)

entry_m, entry_g, entry_c, entry_l0, entry_omega, entry_alpha, entry_l_init, entry_v_init = entries

tk.Button(root, text="Запустить модель", command=run_simulation).pack(pady=5)
tk.Button(root, text="Перейти в равновесие", command=go_to_equilibrium).pack(pady=5)

tk.Label(root, text="Скорость анимации (мс)").pack()
animation_speed = tk.Scale(root, from_=1, to=200, orient=tk.HORIZONTAL)
animation_speed.set(15)
animation_speed.pack()

root.mainloop()



