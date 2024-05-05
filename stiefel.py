import copy

import numpy as np


def stiefel_gradient(euclidian_grad, X):
    stiefel_drv = euclidian_grad @ X.conj().transpose() - X @ euclidian_grad.conj().transpose()
    return stiefel_drv


def get_stiefel_gradient_function(euclidian_grad_fn):
    def stiefel_grad(_X):
        return euclidian_grad_fn(_X) @ _X.conj().transpose() - _X @ euclidian_grad_fn(_X).conj().transpose()

    return stiefel_grad


def cayley_curve_decent(alpha, stiefel_grad, X):
    Id_mat = np.identity(stiefel_grad.shape[0], dtype=complex)
    L = np.linalg.pinv(Id_mat + alpha * stiefel_grad / 2) @ (Id_mat - alpha * stiefel_grad / 2) @ X
    return L


def stiefel_optimization(cost_fun, callback=None, print_cost=False, method='gradient_decent', **kwargs):
    if method == 'bgd':
        _alpha = kwargs.get('alpha')
        _stiefel_grad_fn = kwargs.get('stiefel_grad_fn')
        _X0 = kwargs.get('X0')
        _max_iterations = kwargs.get('max_iterations')
        _tol = kwargs.get('tol', 1e-6)

        return _stochastic_gradient_decent_stiefel_opt(_alpha, cost_fun, _stiefel_grad_fn, _X0,
                                                       _max_iterations, tol=_tol, callback=callback,
                                                       print_cost=print_cost)

    if method == 'gradient_decent':
        _euclidian_grad_fn = kwargs.get('euclidian_grad_fn')
        _stiefel_grad_fn = kwargs.get('stiefel_grad_fn')
        _X0 = kwargs.get('X0')
        _max_iterations = kwargs.get('max_iterations')
        _tol = kwargs.get('tol', 1e-6)
        _alpha_0 = kwargs.get('alpha_0', 0.5)
        _tau = kwargs.get('tau', 0.9)
        _c = kwargs.get('c', 0.4)

        return _gradient_decent_stiefel_opt(cost_fun, _euclidian_grad_fn, _stiefel_grad_fn, _X0,
                                            _max_iterations, alpha_0=_alpha_0, tol=_tol, tau=_tau, c=_c,
                                            callback=callback,
                                            print_cost=print_cost)
    # 如果梯度下降方式是SGD
    if method == 'sgd-m':
        _alpha = kwargs.get('alpha')
        _beta = kwargs.get('beta', 0.9)
        _euclidian_grad_fn = kwargs.get('euclidian_grad_fn')
        _X0 = kwargs.get('X0')
        _max_iterations = kwargs.get('max_iterations')
        _delta = kwargs.get('delta', 1e-6)

        return _sgd_m_stiefel_opt(_alpha, cost_fun, _euclidian_grad_fn, _X0,
                                  _max_iterations, _beta, callback=callback,
                                  print_cost=print_cost)

    # 如果梯度下降方式是ADAM
    if method == 'adam':
        _alpha = kwargs.get('alpha')
        _beta1 = kwargs.get('beta1', 0.9)
        _beta2 = kwargs.get('beta2', 0.999)
        _eps = kwargs.get('eps', 1e-8)
        _euclidian_grad_fn = kwargs.get('euclidian_grad_fn')
        _X0 = kwargs.get('X0')
        _max_iterations = kwargs.get('max_iterations')
        _delta = kwargs.get('delta', 1e-6)
        _dtol = kwargs.get('dtol')

        return _adam_stiefel_opt(_alpha, _eps, cost_fun, _euclidian_grad_fn, _X0,
                                 _max_iterations, _beta1, _beta2, delta=_delta, callback=callback,
                                 print_cost=print_cost, dtol=_dtol)



def euclidian_optimization(cost_fun, callback=None, print_cost=False, method='adam', **kwargs):

    # 如果梯度下降方式是ADAM
    if method == 'adam':
        _alpha = kwargs.get('alpha')
        _beta1 = kwargs.get('beta1', 0.9)
        _beta2 = kwargs.get('beta2', 0.999)
        _eps = kwargs.get('eps', 1e-8)
        _euclidian_grad_fn = kwargs.get('euclidian_grad_fn')
        _X0 = kwargs.get('X0')
        _max_iterations = kwargs.get('max_iterations')
        _delta = kwargs.get('delta', 1e-6)

        return _adam_euclidian_opt(_alpha, _eps, cost_fun, _euclidian_grad_fn, _X0,
                                 _max_iterations, _beta1, _beta2, delta=_delta, callback=callback,
                                 print_cost=print_cost)



def canonical_inner_product(Z1: np.ndarray, Z2: np.ndarray, X: np.ndarray):
    Id = np.identity(max(X.shape), dtype=complex)
    return np.trace(Z1.conj().transpose() @ (Id - 1 / 2 * X @ X.conj().transpose()) @ Z2)


def _stochastic_gradient_decent_stiefel_opt(alpha, cost_fun, stiefel_grad_fn, X0, max_iterations,
                                            tol=1e-6, callback=None, print_cost=False):
    X = X0
    for _i in range(max_iterations):
        D = stiefel_grad_fn(X)
        X = cayley_curve_decent(alpha, D, X)
        if callback is not None:
            callback(X)

        grad_norm = np.linalg.norm(D)
        if print_cost:
            print('cost = ' + str(cost_fun(X)) + ', grad_norm = ' + str(grad_norm))

        if grad_norm < tol:
            break

    return X


def _gradient_decent_stiefel_opt(cost_fun, euclidian_grad_fn, stiefel_grad_fn, X0, max_iterations,
                                 tol=1e-6, alpha_0=0.5, tau=0.9, c=0.4, callback=None, print_cost=False):
    def line_search(_G: np.ndarray, _X: np.ndarray, _D: np.ndarray) -> float:
        # Check inputs
        assert tau < 1
        assert tau > 0
        assert c > 0
        assert c < 1

        inner_prod = np.real(canonical_inner_product(-_G, _D @ _X, _X))

        _alpha = alpha_0
        fx = cost_fun(_X)
        fx_new = cost_fun(cayley_curve_decent(_alpha, _D, _X))
        rhs = _alpha * c * inner_prod

        while fx_new - fx > rhs:
            _alpha *= tau
            fx_new = cost_fun(cayley_curve_decent(_alpha, _D, _X))
            rhs = _alpha * c * inner_prod

        return _alpha

    X = X0
    for _i in range(max_iterations):
        D = stiefel_grad_fn(X)
        G = euclidian_grad_fn(X)

        alpha = line_search(G, X, D)

        X = cayley_curve_decent(alpha, D, X)
        if callback is not None:
            callback(X)

        grad_norm = np.linalg.norm(D)
        if print_cost:
            print('cost = ' + str(cost_fun(X)) + ', grad_norm = ' + str(grad_norm))

        if grad_norm < tol:
            break

    return X


def _gradient_decent_euclidian_opt(cost_fun, euclidian_grad_fn, X0, max_iterations=10000,
                                   tol=1e-6, alpha_0=0.5, tau=0.9, c=0.4, callback=None, print_cost=False):
    def line_search(_G: np.ndarray, _X: np.ndarray) -> float:
        # Check inputs
        assert tau < 1
        assert tau > 0
        assert c > 0
        assert c < 1

        inner_prod = np.real(np.trace(-_G.conj().transpose() @ _G))
        _alpha = alpha_0
        fx = cost_fun(_X)
        fx_new = cost_fun(_X - _alpha * _G)
        rhs = _alpha * c * inner_prod

        while fx_new - fx > rhs:
            _alpha *= tau
            fx_new = cost_fun(_X - _alpha * _G)
            rhs = _alpha * c * inner_prod

        return _alpha

    X = X0
    for _i in range(max_iterations):

        G = euclidian_grad_fn(X)

        alpha = line_search(G, X)

        X = X - alpha * G
        if callback is not None:
            callback(X)

        grad_norm = np.linalg.norm(G)
        if print_cost:
            print('cost = ' + str(cost_fun(X)) + ', grad_norm = ' + str(grad_norm))

        if grad_norm < tol:
            break

    return X


# 具体SGD算法
def _sgd_m_stiefel_opt(alpha, cost_fun, euclidian_grad_fn, X0, max_iterations,
                       beta, delta=1e-6, callback=None, print_cost=False):
    # Init
    X = X0
    momentum = np.zeros(X0.shape, dtype=complex)

    for _i in range(max_iterations):
        # Update momentum
        G = euclidian_grad_fn(X)
        momentum = beta * momentum - G
        D = stiefel_gradient(-momentum, X)

        # Update var
        X = cayley_curve_decent(alpha, D, X)
        if callback is not None:
            callback(X)

        grad_norm = np.linalg.norm(D)
        if print_cost:
            print('SGD: cost = ' + str(cost_fun(X)) + ', grad_norm = ' + str(grad_norm))

        if grad_norm < delta:
            break

    return X


# 具体ADAM算法

def _adam_stiefel_opt(alpha, eps, cost_fun, euclidian_grad_fn, X0, max_iterations,
                      beta1, beta2, delta=1e-6, callback=None, print_cost=False, dtol=None):
    # Init
    X = X0
    momentum = np.zeros(X0.shape, dtype=complex)
    v = 1
    beta1i = 1
    beta2i = 1
    val = cost_fun(X)
    last_val = val + 10000
    for _i in range(max_iterations):
        # Update momentum
        G = euclidian_grad_fn(X)
        momentum = beta1 * momentum + (1 - beta1) * G
        G_norm = np.linalg.norm(G)
        # Update v
        v = beta2 * v + (1 - beta2) * (G_norm ** 2)
        # Update r
        beta2i = beta2i * beta2
        v_hat = v / (1 - beta2i)
        beta1i = beta1i * beta1
        r = (1 - beta1i) * ((v_hat + eps) ** 0.5)

        W = stiefel_gradient(momentum, X) / r

        adam_alpha = min(alpha, 1 / (np.linalg.norm(W) + eps))

        # Update var
        # 应不应该“*r”？??
        X = cayley_curve_decent(adam_alpha, W, X)

        if callback is not None:
            callback(X)

        grad_norm = np.linalg.norm(stiefel_gradient(G, X))
        val = cost_fun(X)


        if print_cost:
            print('ADAM: cost = ' + str(val) + ', grad_norm = ' + str(grad_norm))

        if grad_norm < delta:
            break
        if dtol is not None:
            if last_val - val < dtol:
                break
        last_val = val
    return X




def _adam_euclidian_opt(alpha, eps, cost_fun, euclidian_grad_fn, X0, max_iterations,
                        beta1, beta2, delta=1e-6, callback=None, print_cost=False):
    # Init
    X = X0
    momentum = np.zeros(X0.shape, dtype=complex)
    v = 1
    beta1i = 1
    beta2i = 1

    for _i in range(max_iterations):
        # Update momentum
        G = euclidian_grad_fn(X)
        momentum = beta1 * momentum + (1 - beta1) * G
        G_norm = np.linalg.norm(G)
        # Update v
        v = beta2 * v + (1 - beta2) * (G_norm ** 2)
        # Update r
        beta2i = beta2i * beta2
        v_hat = v / (1 - beta2i)
        beta1i = beta1i * beta1
        r = (1 - beta1i) * ((v_hat + eps) ** 0.5)

        W = momentum / r

        adam_alpha = min(alpha, 1 / (np.linalg.norm(W) + eps))

        # Update var
        # 应不应该“*r”？🤔
        X = X - adam_alpha * W

        if callback is not None:
            callback(X)

        grad_norm = np.linalg.norm(G)
        if print_cost:
            print('ADAM: cost = ' + str(cost_fun(X)) + ', grad_norm = ' + str(grad_norm))

        if grad_norm < delta:
            break

    return X



if __name__ == '__main__':
    import utils.pt_isometry as iso

    in_dim = 2
    out_dim = 4

    np.random.seed(2)
    H = np.random.uniform(-1, 1, (2, 2)) + 1j * np.random.uniform(-1, 1, (2, 2))
    H = H @ H.conj().transpose()

    K = np.random.uniform(-1, 1, (4, 4)) + 1j * np.random.uniform(-1, 1, (4, 4))
    K = K @ K.conj().transpose()


    def cost_fn(X: np.ndarray):
        return np.real(np.trace(X.conj().transpose() @ X @ X.conj().transpose() @ X))


    def conj_derivative(X: np.ndarray):
        drv = 4 * X @ X.conj().transpose() @ X
        return drv


    def derivative(X: np.ndarray):
        drv = K.transpose() @ X.conj() @ H.transpose()
        return drv


    _X = iso.random_isometry(1, 2)
    _t = 0.01
    res = _gradient_decent_euclidian_opt(cost_fn, conj_derivative, _X, 10000, print_cost=True)



    # print(cost_fn(_X))
    # for i in range(10000):
    #     _G = conj_derivative(_X)
    #     _D = stiefel_gradient(_G, _X)
    #     _X = cayley_curve_decent(_t, _D, _X)
    #
    #     print('cost = ' + str(cost_fn(_X)) + ', G_norm = ' + str(np.linalg.norm(_D)))
    #
    # print(_X.conj().transpose() @ _X)
