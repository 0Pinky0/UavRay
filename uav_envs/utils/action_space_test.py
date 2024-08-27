v_range = (0.0, 2.0)
w_range = (-2.0, 2.0)
nvec = (3, 5)

for speed_action in range(nvec[0]):
    speed = (v_range[0]
             + (speed_action + 0) / (nvec[0] - 1) * (v_range[1] - v_range[0]))
    print(speed)
print()
for turn_action in range(nvec[1]):
    angular_velocity = (w_range[0]
                        + turn_action / (nvec[1] - 1) * (w_range[1] - w_range[0]))
    print(angular_velocity)
