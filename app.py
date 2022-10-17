from flask import Flask, jsonify, request

import fourier_calculations as fc
from function_model import FunctionModel

app = Flask(__name__)

@app.route('/fourier', methods=['POST'])
def receive_input():  # put application's code here
    content = request.get_json()
    funcs = []
    for i in content['functions']:
        funcs.append(FunctionModel(i['f_t'], i['start'], i['end']))

    p = fc.calculate_period(funcs[0].start, funcs[len(funcs) - 1].end)
    n = 1
    e_f = sum(fc.energy_f(funcs[i].f_t, funcs[i].start, funcs[i].end) for i in range(0, len(funcs)))
    frac_e_f = 0.02 * e_f
    a_0 = sum(fc.fourier_a0(funcs[i].f_t, p, funcs[i].start, funcs[i].end) for i in range(0, len(funcs)))
    a_n = 0
    b_n = 0
    result = 0
    while True:
        a_n = sum(fc.fourier_an(f=funcs[i].f_t, p=p, a=funcs[i].start, b=funcs[i].end, n=n) for i in range(0, len(funcs)))
        b_n = sum(fc.fourier_bn(f=funcs[i].f_t, p=p, a=funcs[i].start, b=funcs[i].end, n=n) for i in range(0, len(funcs)))
        result = fc.calc_ice(e_f=e_f, a_0=a_0, funcs=funcs, p=p, N=n)

        if result <= frac_e_f:
            break
        else:
            n += 1

    plot, equation = fc.plot_fourier(a_0, a_n, b_n, p, n)

    return jsonify({
        "a_n": fc.trunc(a_n),
        "b_n": fc.trunc(b_n),
        "a_0": fc.trunc(a_0),
        "period": str(p),
        "value_n": str(n),
        "plot": plot,
        "equation": equation,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0')
