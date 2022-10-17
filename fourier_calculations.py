import numpy as np
from numpy import pi
from scipy.integrate import quad
import matplotlib.pyplot as plt
import io
import base64
import requests

def w(p):
    return (2*pi)/p


def check_if_f(func):
    try:
        f = int(func)
        return False
    except Exception as e:
        return True


def ft(func):
    f = eval(func)
    return f


def t_integrand(f, a, b):
    return f*(b - a)


def calculate_period(a, b):
    return np.absolute(a) + np.absolute(b)


def fourier_a0(f, p, a, b):
    if check_if_f(f):
        integration, e = quad(lambda t: eval(f), a, b)
    else:
        integration = t_integrand(int(f), a, b)

    res = (1 / p) * integration
    return res


def fourier_an(f, p, a, b, n):
    w_nt = w(p) * n

    def cos_t(t):
        return np.cos((w_nt * t))

    if check_if_f(f):
        integration, e = quad(lambda t: eval(f)*cos_t(t), a, b)
    else:
        integration, e = quad(lambda t: cos_t(t), a, b)
        integration = int(f) * integration

    return (2/p) * integration


def fourier_bn(f, p, a, b, n):
    w_nt = w(p) * n

    def sin_t(t):
        return np.sin((w_nt * t))

    # integration, e = quad(lambda t: f(t)*sin_t(t), a, b)

    if check_if_f(f):
        integration, e = quad(lambda t: eval(f)*sin_t(t), a, b)
    else:
        integration, e = quad(lambda t: sin_t(t), a, b)
        integration = int(f) * integration

    return (2/p) * integration


def energy_f(f, a, b):
    if check_if_f(f):
        print("f = ", f)
        integration, e = quad(lambda t: eval(f)**2, a, b)
    else:
        integration = (t_integrand(int(f)**2, a, b))

    return integration


def calc_ice(e_f, a_0, funcs, p, N):
    result = 0
    for x in range(1, N + 1):
        a_n = sum(
            fourier_an(f=funcs[i].f_t, p=p, a=funcs[i].start, b=funcs[i].end, n=x) for i in range(0, len(funcs)))
        b_n = sum(
            fourier_bn(f=funcs[i].f_t, p=p, a=funcs[i].start, b=funcs[i].end, n=x) for i in range(0, len(funcs)))
        result += a_n ** 2 + b_n ** 2

    result = (p / 2) * result
    result = a_0 ** 2 * p + result
    result = e_f - result
    return result


def fourier_trigonometric(a_0, a_n, b_n, p, N):
    def func(t):
        return a_0 + sum((a_n * np.cos(w(p) * n * t)) + b_n * np.sin(w(p) * n * t) for n in range(1, 2 * N + 1))

    return func


def plot_fourier(a_0, a_n, b_n, p, N):
    t = np.linspace(-2, 2, 1000)
    f = fourier_trigonometric(a_0, a_n, b_n, p, N)
    fig = plt.figure(figsize=(12, 4))
    axes = plt.axes()
    axes.plot(t, np.vectorize(f)(t))
    my_string_byte = io.BytesIO()
    plt.savefig(my_string_byte, format='jpg')
    my_string_byte.seek(0)
    plt.close()
    base64_image = base64.b64encode(my_string_byte.read())
    omega = w(p)
    axes = plt.axes()
    my_string_byte = io.BytesIO()
    plt.text(0, 0.5, r'$f(t) = %s - \sum_{n=1}^{%s} (%s) \times cos(n \times %s \times t) '
                     r'+ (%s) \times sin(n \times %s \times t) $' %(trunc(a_0), str(2 * N + 1),
                                                                    trunc(a_n), trunc(omega),
                                                                    trunc(b_n), trunc(omega)), fontsize=10)
    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.box(False)
    plt.savefig(my_string_byte, format='jpg')
    my_string_byte.seek(0)
    plt.close()
    equation = base64.b64encode(my_string_byte.read())

    return base64_image.decode("utf-8").replace("\n", ""), equation.decode("utf-8").replace("\n", "")


def trunc(n):
    return format(n, '.2f')