import time
import os
import matplotlib.pyplot as plt

CSV_PATH = "distances.csv"
WINDOW = 20

plt.ion()
fig, ax = plt.subplots()

(line,) = ax.plot([], [], marker="o", linestyle="-", label="max_x")
(avg_line,) = ax.plot([], [], linestyle="-", linewidth=2, label=f"media {WINDOW}")

ax.set_title("Distanta maxima pe viata (live)")
ax.set_xlabel("Viata #")
ax.set_ylabel("max x_pos")
ax.legend()

lives = []
distances = []
last_size = -1

def moving_average(values, window):
    if len(values) < window:
        return []
    out = []
    s = sum(values[:window])
    out.append(s / window)
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out.append(s / window)
    return out

while True:
    if os.path.exists(CSV_PATH):
        size = os.path.getsize(CSV_PATH)

        if size != last_size:
            last_size = size
            lives.clear()
            distances.clear()

            with open(CSV_PATH, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()

            for row in lines[1:]:
                if not row.strip():
                    continue
                a, b = row.split(",")
                lives.append(int(a))
                distances.append(float(b))

            line.set_data(lives, distances)

            avg_vals = moving_average(distances, WINDOW)
            if avg_vals:
                avg_lives = lives[WINDOW - 1 :]
                avg_line.set_data(avg_lives, avg_vals)
            else:
                avg_line.set_data([], [])

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    plt.pause(0.2)
    time.sleep(0.2)