import numpy as np
from matplotlib import pyplot as plt


def plot(env, dq, i, best=False):
    if env.I_zones.shape == 'box':
        up = env.I_zones.up

        low = env.I_zones.low
        x1, y1 = up[:2]
        x2, y2 = low[:2]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label='initial set', c='b')
    else:
        r = np.sqrt(env.I_zones.r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.I_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.I_zones.center[1] for v in thta]
        plt.plot(x, y, label='initial set', c='b')
    if env.G_zones.shape == 'box':
        up = env.G_zones.up
        low = env.G_zones.low
        x1, y1 = up[:2]
        x2, y2 = low[:2]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label='target set', c='green')
    else:
        r = np.sqrt(env.G_zones.r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.G_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.G_zones.center[1] for v in thta]
        plt.plot(x, y, label='target set', c='yellow')
    co = 0
    for X, Y in dq:
        plt.plot(X, Y, linewidth=0.5,
                 c='brown')  # ('brown' if co % 2 == 0 else 'yellow'), label=('u0' if co % 2 == 0 else 'u')
        co += 1
    # plt.plot(range(len(Y)), Y,label='Y')
    # plt.plot(range(len(Z)),Z,label='Z')
    if env.U_zones.shape == 'box':
        up = env.U_zones.up
        low = env.U_zones.low
        x1, y1 = up[:2]
        x2, y2 = low[:2]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label='unsafe region', c='r')
    else:
        r = np.sqrt(env.U_zones.r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.U_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.U_zones.center[1] for v in thta]
        plt.plot(x, y, label='unsafe region', c='r')
    plt.legend()
    # if i % 2 == 1:
    #     plt.savefig('1.jpg', dpi=1000)

    # if best:
    #     plt.savefig(f'./model/ex{env.id}/best.jpg', dpi=1000)

    plt.show()
