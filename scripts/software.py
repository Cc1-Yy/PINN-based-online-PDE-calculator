import io
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import abc
from jax import random, jit, vjp, grad, vmap
import jax.flatten_util as flat_utl
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from pyDOE import lhs
from contextlib import redirect_stderr
import plotly.express as px
import plotly.graph_objects as go
import functools
from scipy.io import savemat
from pathlib import Path

# change JAX to double precision
jax.config.update('jax_enable_x64', True)

# Customized function for collocation point arrangement

import scipy
import jax.scipy as jsp


def gaussian1D_smooth(f, sig, wid):
    """
    :param f: equally spaced 1D position matrix  (N, 1)
    :param sig: stan. devi of gaussian filter (1, ) or scalor
    :param wid: wid of the filter matrix (1, ) or scalor integer
    """
    wid = jnp.int32(wid)
    xg = jnp.linspace(-sig, sig, wid)
    window = jsp.stats.norm.pdf(xg)
    win_n = window / jnp.sum(window)
    f_smooth = scipy.signal.convolve(f[:, 0], win_n, mode='same')[:, None]
    return f_smooth


# sample the data based on a given probability distribution
def colloc1D_set(key, x, f, Ns):
    """
    :param x: 1-D position array (N, 1)
    :param f: 1-D distribution array (N, 1)
    :param Ns: number of points to sample
    """
    # remove last element in each direction
    xc = x[0:-1, :]
    fc = f[0:-1, :]
    dx = xc[1] - xc[0]
    seq = jnp.arange(fc.shape[0] + 1)
    # generate key for random variables
    keys = jax.random.split(key, num=2)

    # obtain the cumulative sum of the z value
    b = jnp.hstack([0., jnp.cumsum(fc)])
    # obtain the random variable
    c = jax.random.uniform(keys[0], [Ns]) * b[-1]
    # generate the index position of each collocation point following the distribution
    # (using the interpolate the index of grid where each random variable stands)
    posi_intp = jnp.interp(c, b, seq)
    # round the result to guarantee that the index position is integer
    posi = jnp.int32(jnp.floor(posi_intp))
    # obtain the real position of each collocation point
    px = xc[posi, :]
    # generate a random fraction for each collocation point
    posi_add = jax.random.uniform(keys[1], [c.shape[0], 1])
    # add the random fraction to the position of each collocation points
    x_col = px + posi_add * dx

    return x_col


# smooth the imaging using Gaussian filter
def gaussian2D_smooth(f, sig, wid):
    """
    :param f: equally spaced 2D position matrix [N, N]
    :param sig: stan. devi of gaussian filter (2, )
    :param wid: wid of the filter matrix (2, ) integer
    """
    wid = jnp.int32(wid)
    xg = jnp.linspace(-sig[0], sig[0], wid[0])
    yg = jnp.linspace(-sig[1], sig[1], wid[1])
    window = jsp.stats.norm.pdf(xg) * jsp.stats.norm.pdf(yg)[:, None]
    win_n = window / jnp.sum(window)
    f_smooth = scipy.signal.convolve2d(f, win_n, mode='same')
    return f_smooth


# sample the data based on a given probability distribution
def colloc2D_set(key, X, Y, F, Ns):
    """
    :param key: Key for random
    :param X: 2-D X position array (N, N)
    :param Y: 2-D Y position array (N, N)
    :param F: 2-D distribution array (N, N)
    :param Ns: number of points to sample
    """

    # remove last element in each direction
    Xc = X[0:-1, 0:-1]
    Yc = Y[0:-1, 0:-1]
    Fc = F[0:-1, 0:-1]
    f = Fc.flatten()

    x = X[0, :]
    y = Y[:, 0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    seq = jnp.arange(f.shape[0] + 1)
    # generate key for random variables
    keys = jax.random.split(key, num=2)

    # obtain the cumulative sum of the z value
    b = jnp.hstack([0., jnp.cumsum(f)])
    # obtain the random variable
    c = jax.random.uniform(keys[0], [Ns]) * b[-1]

    # generate the index position of each collocation point following the distribution
    # (using the interpolate the index of grid where each random variable stands)
    posi_intp = jnp.interp(c, b, seq)
    # round the result to guarantee that the index position is integer
    posi_rd = jnp.floor(posi_intp)
    # obtain the 2D position of each collocation point in the position matrix
    idx_out = jnp.int32(jnp.floor(posi_rd / Fc.shape[1]))
    idx_in = jnp.int32(posi_rd % Fc.shape[1])
    # obtain the real position of each collocation point
    px = Xc[idx_out, idx_in]
    py = Yc[idx_out, idx_in]

    # generate a random fraction for each collocation point
    posi_add = jax.random.uniform(keys[1], [2, c.shape[0]])
    # add the random fraction to the position of each collocation points
    Px = px + posi_add[0] * dx
    Py = py + posi_add[1] * dy

    # group the x,y position of the collocation points into one [Nx2] matrix
    X_col = jnp.hstack((Px[:, None], Py[:, None]))

    return X_col


# Weight initialization and network architecture

# initialize the neural network weights and biases
def init_MLP(parent_key, layer_widths):
    params = []
    nl = len(layer_widths) - 1
    keys = random.split(parent_key, num=nl)
    # create the weights and biases for the network
    for in_dim, out_dim, key, l in zip(layer_widths[:-1], layer_widths[1:], keys, range(nl)):
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        weight_key, bias_key = random.split(key)
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * xavier_stddev]
        )
    return params


# define the basic formation of neural network
def neural_net(params, z, limit, scl, act_s):
    """
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param limit: characteristic scale for normalizeation [matrx with shape [2, m]]
    :param sgn:  1 for even function and -1 for odd function
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    """
    lb = limit[0]  # lower bound for each input
    ub = limit[1]  # upper bound for each input

    # choose the activation function
    actv = [jnp.tanh, jnp.sin][act_s]
    # normalize the input
    H_r = 2.0 * (z[:, 0:1] - lb[0]) / (ub[0] - lb[0]) - 1.0
    H_cost = jnp.cos(z[:, 1:2])  # hard constraint
    H_sint = jnp.sin(z[:, 1:2])
    H = jnp.concatenate([H_r, H_cost, H_sint], axis=1)    # separate the first, hidden and last layers
    first, *hidden, last = params
    # calculate the first layers output with right scale
    H = actv(jnp.dot(H, first[0]) * scl + first[1])
    # calculate the middle layers output
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    # no activation function for last layer
    var = jnp.dot(H, last[0]) + last[1]
    return var


# define sech function
def sech(z):
    return 1 / jnp.cosh(z)


# generate weights and biases for all variables of CLM problem
def sol_init_MLP(parent_key, n_hl, n_unit):
    """
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    """
    layers = [3] + n_hl * [n_unit] + [1]
    # generate the random key for each network
    keys = random.split(parent_key, 1)
    # generate weights and biases for each network
    params_u = init_MLP(keys[0], layers)
    return params_u


# wrapper to create solution function with given domain size
def sol_pred_create(limit, scl, epsil, act_s=0):
    """
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    """

    def f_u(params, z):
        # generate the NN
        u = epsil * neural_net(params, z, limit, scl, act_s)
        return u

    return f_u


def mNN_pred_create(f_u, limit, scl, epsil, act_s=0):
    """
    :param f_u: sum of previous stage network
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    """

    def f_comb(params, z):
        # generate the NN
        u_now = neural_net(params, z, limit, scl, act_s)
        u = f_u(z) + epsil * u_now
        return u

    return f_comb


"""Low-level functions developed for PINN training using JAX"""


# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


# generate matrix required for vjp for vector gradient
def vgmat(z, n_out, idx=None):
    """
    :param n_out: number of output variables
    :param idx: indice (list) of the output variable to take the gradient
    """
    if idx is None:
        idx = range(n_out)
    # obtain the number of index
    n_idx = len(idx)
    # obtain the number of input points
    n_pt = z.shape[0]
    # determine the shape of the gradient matrix
    mat_shape = [n_idx, n_pt, n_out]
    # create the zero matrix based on the shape
    mat = jnp.zeros(mat_shape)
    # choose the associated element in the matrix to 1
    for l, ii in zip(range(n_idx), idx):
        mat = mat.at[l, :, ii].set(1.)
    return mat


# vector gradient of the output with input
def vectgrad(func, z):
    # obtain the output and the gradient function
    sol, vjp_fn = vjp(func, z)
    # determine the mat grad
    mat = vgmat(z, sol.shape[1])
    # calculate the gradient of each output with respect to each input
    grad_sol = vmap(vjp_fn, in_axes=0)(mat)[0]
    # calculate the total partial derivative of output with input
    n_pd = z.shape[1] * sol.shape[1]
    # reshape the derivative of output with input
    grad_all = grad_sol.transpose(1, 0, 2).reshape(z.shape[0], n_pd)
    return grad_all, sol


# governing equation
def gov_eqn(f_u, z):
    u_g, u = vectgrad(f_u, z)
    u_r = u_g[:, 0:1]
    u_r = u_g[:, 0:1]

    fu_r = lambda z: vectgrad(f_u, z)[0][:, 0:1]
    fu_t = lambda z: vectgrad(f_u, z)[0][:, 1:2]
    # calculate the output and its derivative with original coordinates
    u_rr = vectgrad(fu_r, z)[0][:, 0:1]
    u_tt = vectgrad(fu_t, z)[0][:, 1:2]
    # split the input variables
    r, t = jnp.split(z, 2, axis=1)
    # calculate the residue of the CCF equation
    f = u_rr + 1 / r * u_r + 1 / (r ** 2) * u_tt  # TODO: equation:输入"f="后的内容
    return f


def gov_deri_eqn(f_u, z):
    # allocate the value to each variable
    fc_res = lambda z: gov_eqn(f_u, z)
    # calculate the residue of higher derivative of CCF equation
    dfunc = lambda z: vectgrad(fc_res, z)[0]
    # calculate the residue of the first and second derivative of CCF equation
    df, f = vectgrad(fc_res, z)
    return df, f


def loss_create(predf_u, lw, loss_ref):
    """
    a function factory to create the loss function based on given info
    :param loss_ref: loss value at the initial of the training
    :return: a loss function (callable)
    """

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        f_u = lambda z: predf_u(params, z)
        # load the data of normalization condition
        z_bd = data['cond_bd'][0]
        u_bd = data['cond_bd'][1]

        num_z = len(z_bd)
        num_u = len(u_bd)
        u_bd_pred = []
        norm_err = []

        # load the position and weight of collocation points
        x_col = data['x_col']

        # calculate the gradient of phi at origin
        for i in range(num_z):
            u_bd_pred.append(f_u(z_bd[i]))

        # u_bd_p1 = f_u(z_bd[0])
        # u_bd_p2 = f_u(z_bd[1])
        # u_bd_p3 = f_u(z_bd[2])
        # u_bd_p4 = f_u(z_bd[3])

        # calculate the mean squared root error of normalization cond.
        for i in range(num_u):
            norm_err.append(ms_error(u_bd_pred[i] - u_bd[i]))
        # norm_err0 = ms_error(u_bd_p1 - u_bd[0])
        # norm_err1 = ms_error(u_bd_p2 - u_bd[1])
        # norm_err2 = ms_error(u_bd_p3 - u_bd[2])
        # norm_err3 = ms_error(u_bd_p4 - u_bd[3])

        # calculate the error of far-field exponent cond.
        data_err = jnp.hstack(norm_err)

        # calculate the residue of first and second derivative
        # df, f = gov_deri_eqn(f_u, x_col)
        f = gov_eqn(f_u, x_col)

        # calculate the mean squared root error of equation
        eqn_err_f = ms_error(f)
        # eqn_err_df = ms_error(df)
        # eqn_err_d2f = ms_error(d2f)
        eqn_err = jnp.hstack([eqn_err_f])  # , eqn_err_df

        lw = loss_fun.lw
        lref = loss_fun.ref
        # set the weight for each condition and equation
        data_weight = jnp.array([1.])
        eqn_weight = jnp.array([1.])  # , lw[1], lw[1]

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)

        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn)
        loss_n = loss / lref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn]),
                                data_err, eqn_err])
        return loss_n, loss_info

    loss_fun.ref = loss_ref
    loss_fun.lw = lw
    return loss_fun


# create the Adam minimizer
@functools.partial(jit, static_argnames=("lossf", "opt"))
def adam_minimizer(lossf, params, data, opt, opt_state):
    """Basic gradient update step based on the opt optimizer."""
    grads, loss_info = grad(lossf, has_aux=True)(params, data)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss_info, opt_state


def adam_optimizer(R_add, T_add, lossf, predf, params, dataf, F, epoch, key_adam, lr=1e-3):
    # select the Adam as the minimizer
    opt_Adam = optax.adam(learning_rate=lr)
    # obtain the initial state of the params
    opt_state = opt_Adam.init(params)
    # pre-allocate the loss variable
    loss_all = []
    # set the first group of data
    key = key_adam
    data = dataf(key, F, R_add, T_add)
    R = dataf.R
    T = dataf.T

    nc = jnp.int32(jnp.round(epoch / 5))
    nc0 = 2000
    # start the training iteration
    for step in range(epoch):
        # minimize the loss function using Adam
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        # print the loss for every 100 iteration
        if step % 100 == 0 and step > 0:
            # print the results
            print(f"Step: {step} | Loss: {loss_info[0]:.4e} |"
                  f" Loss_d: {loss_info[1]:.4e} | Loss_e: {loss_info[2]:.4e} | ", file=sys.stderr)
            # re-sampling the data points
            key = random.split(key, 1)[0]
            data = dataf(key, F, R_add, T_add)

        # saving the loss
        loss_all.append(loss_info[0:])

        if (step + 1) % nc0 == 0:
            F = predictF(predf, params, R, T)

        if (step + 1) % (2 * nc0) == 0:
            lossend = np.array(loss_all[-2 * nc0:])[:, 0]
            lc1 = lossend[0: nc0]
            lc2 = lossend[nc0:]
            mm12 = jnp.abs(jnp.mean(lc1) - jnp.mean(lc2))
            stdl2 = jnp.std(lc2)
            # if the average loss improvement within 'nc' iteration is less than local loss fluctuation (std)
            if mm12 / stdl2 < 0.4:
                # reduce the learning rate by half
                lr = lr / 2
                opt_Adam = optax.adam(learning_rate=lr)
            print(f"learning rate for Adam: {lr:.4e} | mean: {mm12:.3e} | std: {stdl2:.3e}", file=sys.stderr)

    # obtain the total loss in the last iterations
    lossend = jnp.array(loss_all[-nc:])[:, 0]
    # find the minimum loss value
    lmin = jnp.min(lossend)
    # optain the last loss value
    llast = lossend[-1]
    # guarantee the loss value in last iteration is smaller than anyone before
    for lc in range(2 * nc0):
        if llast < lmin:
            break
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        llast = loss_info[0]
        # saving the loss
        loss_all.append(loss_info[0:])

    print(f"Step: {step} | Loss: {loss_info[0]:.4e} |"
          f" Loss_d: {loss_info[1]:.4e} | Loss_e: {loss_info[2]:.4e} | ", file=sys.stderr)
    return params, loss_all


# A factory to create a function required by tfp.optimizer.lbfgs_minimize.
def lbfgs_function(lossf, init_params, data):
    # obtain the 1D parameters and the function that can turn back to the pytree
    _, unflat = flat_utl.ravel_pytree(init_params)

    def update(params_1d):
        # updating the model's parameters from the 1D array
        params = unflat(params_1d)
        return params

    # A function that can be used by tfp.optimizer.lbfgs_minimize.
    @jit
    def f(params_1d):
        # convert the 1d parameters back to pytree format
        params = update(params_1d)
        # calculate gradients and convert to 1D tf.Tensor
        grads, loss_info = grad(lossf, has_aux=True)(params, data)
        # convert the grad to 1d arrays
        grads_1d = flat_utl.ravel_pytree(grads)[0]
        loss_value = loss_info[0]

        # # store loss value so we can retrieve later
        jax.debug.callback(lambda x: f.loss.append(x), loss_info[0:])
        jax.debug.callback(lambda x: print(f"Step: NaN | Loss: {x[0]:.4e} |"
                                           f" Loss_d: {x[1]:.4e} | Loss_e: {x[2]:.4e}"),
                           loss_info)

        return loss_value, grads_1d

    # store these information as members so we can use them outside the scope
    f.update = update
    f.loss = []
    return f


# define the function to apply the L-BFGS optimizer
def lbfgs_optimizer(lossf, params, data, epoch):
    func_lbfgs = lbfgs_function(lossf, params, data)
    # convert initial model parameters to a 1D array
    init_params_1d = flat_utl.ravel_pytree(params)[0]
    # calculate the effective number of iteration
    max_nIter = jnp.int32(epoch / 3)
    # train the model with L-BFGS solver
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func_lbfgs, initial_position=init_params_1d,
        tolerance=1e-10, max_iterations=max_nIter)
    params = func_lbfgs.update(results.position)
    # history = func_lbfgs.loss
    num_iter = results.num_objective_evaluations
    loss_all = func_lbfgs.loss
    print(f" Total iterations: {num_iter}")
    return params, loss_all


# Dynamic data sampling

# Dynamic data sampling

def data_func_create(N_col, N_bd, boundary, domain):
    r = jnp.linspace(domain["x_min"], domain["x_max"], 111)
    t = jnp.linspace(domain["y_min"], domain["y_max"], 111)
    R, T = jnp.meshgrid(r, t)

    # generate the points close to boundary and collocation points
    F = 1 + R * 0
    idx = jnp.where((R > (domain["x_min"] + (domain["x_max"] - domain["x_min"]) / 20)) & (
            R < (domain["x_max"] - (domain["x_max"] - domain["x_min"]) / 20)) & (
                            T > (domain["y_min"] + (domain["y_max"] - domain["y_min"]) / 20)) & (
                            T < (domain["y_max"] - (domain["y_max"] - domain["y_min"]) / 20)))  # & (R < 0.98)
    F_bd = F.at[idx[0], idx[1]].set(0)

    # define the function that can re-sampling for each calling
    def dataf(key, F, R_add, T_add):
        keys = random.split(key, 2)
        # set the initial and boundary conditions

        # rt1 = lhs(2, N_bd) * jnp.array(
        #     [boundary["bd_x1_max"] - boundary["bd_x1_min"], boundary["bd_y1_max"] - boundary["bd_y1_min"]]) + jnp.array(
        #     [boundary["bd_x1_min"], boundary["bd_y1_min"]])
        # u1 = boundary["bd_u1"] * jnp.ones(N_bd)[:, None]
        # rt2 = lhs(2, N_bd) * jnp.array(
        #     [boundary["bd_x2_max"] - boundary["bd_x2_min"], boundary["bd_y2_max"] - boundary["bd_y2_min"]]) + jnp.array(
        #     [boundary["bd_x2_min"], boundary["bd_y2_min"]])
        # u2 = boundary["bd_u2"] * jnp.ones(N_bd)[:, None]

        # group the initial and boundary conditions
        x_bd = []
        u_bd = []
        num = len(boundary) // 5
        for i in range(num):
            rt = lhs(2, N_bd) * jnp.array(
                [boundary[f"bd_x{i + 1}_max"] - boundary[f"bd_x{i + 1}_min"],
                 boundary[f"bd_y{i + 1}_max"] - boundary[f"bd_y{i + 1}_min"]]) + jnp.array(
                [boundary[f"bd_x{i + 1}_min"], boundary[f"bd_y{i + 1}_min"]])
            u = boundary[f"bd_u{i + 1}"] * jnp.ones(N_bd)[:, None]
            x_bd.append(rt)
            u_bd.append(u)

        # prepare the collocation points
        x_col = lhs(2, N_col[0]) * jnp.array([domain["x_max"]-domain["x_min"], domain["y_max"]-domain["y_min"]]) + jnp.array([domain["x_min"], domain["y_min"]])
        xc_bd = colloc2D_set(keys[0], R, T, F_bd, N_col[1])
        xc_add = colloc2D_set(keys[1], R_add, T_add, F, N_col[2])

        # add the collocation at the boundary region
        x_col = jnp.vstack([x_col, xc_bd, jnp.vstack(x_bd), xc_add])

        # group all the conditions and collocation points
        data = dict(x_col=x_col, cond_bd=[x_bd, u_bd])
        return data

    dataf.R = R
    dataf.T = T
    return dataf


# plot the collocation point
def colpoint_plot(U, X_col, limit, fig_str, file_name):
    # limit = [x1min, x1max, x1min, x2max]
    # fig_str = ['title', 'xlabel', 'yabel']

    fig = plt.figure(figsize=[12, 10], dpi=100)

    ax = plt.subplot()

    h = ax.imshow(U, interpolation='nearest', cmap='rainbow',
                  extent=limit, origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_col[:, 0], X_col[:, 1], 'kx', markersize=5, clip_on=False)
    ax.set_title(fig_str[0], fontsize=15)
    ax.set_xlabel(fig_str[1], fontsize=15)
    ax.set_ylabel(fig_str[2], fontsize=15, rotation=0)
    plt.show()
    np.savez(
        f"../data/{file_name}",
        U=U,
        X_col=X_col,
        limit=np.array(limit),  # [x1min, x1max, x2min, x2max]
    )


def predictF(predf, params, z1, z2):
    # create the output function
    fsol = lambda z: predf(params, z)
    # generate the input for network
    z_star = jnp.hstack((z1.flatten()[:, None], z2.flatten()[:, None]))
    # calculate the equation residue
    f0 = gov_eqn(fsol, z_star)
    # calculate the maximum of the square of residue for all equations
    f_sq = f0 ** 2
    # normalize the distribution function and add a basic level
    f_nm = f_sq / jnp.mean(f_sq) + 0.5
    # create the 2D distribution function F
    F = jnp.reshape(f_nm, z1.shape)
    # smooth the weight function by Gaussian filter
    Fs = gaussian2D_smooth(F, [1, 1], [5, 5])
    return Fs


def run_pinn_training(
        equation: str,
        boundary: dict,
        domain: dict,
        scl: float,
        epsil: float,
        sample_points: dict,
        network_size: dict,
        testing_size: dict,
        epochs: dict,
        equation_weight: dict,
        *,
        log_path: str = "../data/training.log"
):
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # with log_file.open("w") as f, redirect_stderr(f):
    # --- 1. 构造测试网格 ---
    # Problem Setup
    # Boundary/Initial Condition
    # m_bd_x1_min = boundary["bd_x1_min"]
    # m_bd_x1_max = boundary["bd_x1_max"]
    # m_bd_y1_min = boundary["bd_y1_min"]
    # m_bd_y1_max = boundary["bd_y1_max"]
    # m_bd_u1 = boundary["bd_u1"]
    # m_bd_x2_min = boundary["bd_x2_min"]
    # m_bd_x2_max = boundary["bd_x2_max"]
    # m_bd_y2_min = boundary["bd_y2_min"]
    # m_bd_y2_max = boundary["bd_y2_max"]
    # m_bd_u2 = boundary["bd_u2"]
    # Domain Boundary
    m_x_min, m_x_max = domain["x_min"], domain["x_max"]
    m_y_min, m_y_max = domain["y_min"], domain["y_max"]
    # SCL (Frequency)
    m_scl = scl
    # Epsilon (Range)
    m_epsilon = epsil

    # Training Settings
    # Sample Points:
    m_n_col = sample_points["n_col"]
    m_n_bd = sample_points["n_bd"]
    m_n_add = sample_points["n_add"]
    # Network Size
    m_depth = network_size["depth"]
    m_width = network_size["width"]
    # Testing Size
    m_nx = testing_size["x"]
    m_ny = testing_size["y"]
    # Training Epoch
    m_adam = epochs["adam"]
    m_lbfgs = epochs["lbfgs"]
    # Equation Weight
    m_f = equation_weight["f"]
    m_df = equation_weight["df"]

    # Problem setup
    """Set the conditions of the problem"""
    # select the random seed
    seed = 1234
    key = random.PRNGKey(seed)
    np.random.seed(seed)

    # create the subkeys
    keys = random.split(key, 10)

    # number of sampling points (x_col, x_bd, x_add)
    N_col = jnp.array([m_n_col, m_n_bd, m_n_add])
    N_bd = 100

    """loading the dataset and define the domain"""

    r = jnp.linspace(m_x_min, m_x_max, m_nx)
    t = jnp.linspace(m_y_min, m_y_max, m_ny)

    R, T = jnp.meshgrid(r, t)
    X_star = jnp.hstack((R.flatten()[:, None], T.flatten()[:, None]))

    # Domain bounds
    lb = jnp.array([m_x_min, m_y_min])
    ub = jnp.array([m_x_max, m_y_max])
    limit = [lb, ub]

    # 1st stage neural network

    # initialize the weights and biases of the network
    trained_params1 = sol_init_MLP(keys[0], m_width, m_depth)

    # set the training iteration (data/eqn, eqn 1st/2nd)
    lw1 = jnp.array([m_f, m_df])

    # create the solution function (act_s: tan-0, sin-1)
    pred_u1 = sol_pred_create(limit, m_scl, m_epsilon, act_s=0)

    # create the data function
    dataf1 = data_func_create(N_col, N_bd, boundary, domain)
    Fs = R * 0 + 1
    key_adam = keys[1]
    key_lbfgs = random.split(keys[2], 1)
    data1 = dataf1(key_adam, Fs, R, T)

    z_c1 = data1['x_col']

    """Figure 1.1 - Collation Point"""
    # plot the collocation point
    plot = 1
    if plot == 1:
        colpoint_plot(Fs, z_c1, [domain["x_min"], domain["x_max"], domain["y_min"], domain["y_max"]], ['collo. point', '$t$', '$\th$'],
                      "collocation_point_1.npz")

    # calculate the loss function
    NN_loss = loss_create(pred_u1, lw1, loss_ref=1)
    NN_loss.ref = NN_loss(trained_params1, data1)[1][0]

    """Training using Adam"""

    # set the learning rate for Adam
    lr = 1e-3
    epoch1 = m_adam
    trained_params1, loss1 = adam_optimizer(R, T, NN_loss, pred_u1, trained_params1, dataf1, Fs, epoch1, key_adam,
                                            lr=lr)
    Fs = predictF(pred_u1, trained_params1, R, T)
    data1 = dataf1(key_lbfgs[0], Fs, R, T)

    """Training using L-BFGS"""

    loss2 = []
    epoch2 = m_lbfgs
    for l in range(1):
        trained_params1, loss = lbfgs_optimizer(NN_loss, trained_params1, data1, epoch2)
        Fs = predictF(pred_u1, trained_params1, R, T)
        data1 = dataf1(key_lbfgs[l], Fs, R, T)
        loss2 += loss

    # Result of 1st stage

    """Solution of stage 1"""

    # calculate the equation residue
    f_u1 = lambda z: pred_u1(trained_params1, z)

    # calculate the solution
    u_p1 = f_u1(X_star)
    f_p1 = gov_eqn(f_u1, X_star)

    U1 = jnp.reshape(u_p1, R.shape)
    F1 = jnp.reshape(f_p1, R.shape)

    # generate the last loss
    loss_all1 = np.array(loss1 + loss2)
    # loss_all1 = np.array(loss1)

    U = jnp.reshape(u_p1, R.shape)
    F = jnp.reshape(f_p1, R.shape)

    """Figure 2.1 - Result + Loss-e"""
    fig1 = plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(121)
    contour1 = ax1.contourf(R, T, U, 100, cmap='jet')
    fig1.colorbar(contour1, ax=ax1)
    ax1.set_title('u')
    ax1.set_xlabel('r')
    ax1.set_ylabel('theta')

    ax2 = plt.subplot(122)
    contour2 = ax2.contourf(R, T, F, 100, cmap='jet')
    fig1.colorbar(contour2, ax=ax2)
    ax2.set_title('f')
    ax2.set_xlabel('r')
    ax2.set_ylabel('theta')

    plt.tight_layout()
    plt.show()

    r_vec = np.linspace(m_x_min, m_x_max, m_nx)
    t_vec = np.linspace(m_y_min, m_y_max, m_ny)
    U_np = np.array(U)
    F_np = np.array(F)
    np.savez('../data/solution_residual_1.npz',
             r=r_vec,
             t_vec=t_vec,
             U=U_np,
             F=F_np
             )

    """Figure 3.1 - error"""
    fig1 = plt.figure(figsize=(6, 5))
    U_real = jnp.log(R) / jnp.log(0.1)
    Error = U - U_real
    ax3 = plt.subplot(111)
    contour3 = ax3.contourf(R, T, Error, 100, cmap='jet')
    fig1.colorbar(contour3, ax=ax3)
    ax3.set_title('Error')
    ax3.set_xlabel('r')
    ax3.set_ylabel('error')
    plt.tight_layout()
    plt.show()

    r = np.array(R[0, :])  # θ 网格横坐标
    t = np.array(T[:, 0])  # r 网格纵坐标
    Error_np = np.array(U - U_real)
    # 保存到 data/error_contour.npz
    np.savez(
        "../data/error_1.npz",
        r=r,  # shape (nx,)
        t=t,  # shape (ny,)
        Error=Error_np  # shape (ny, nx)
    )

    """Figure 4.1 - Loss_d + Loss_e"""
    fig2 = plt.figure(figsize=(16, 5))

    # subplot - loss
    ax1 = plt.subplot(131)
    ax1.plot(range(len(loss_all1)), loss_all1[:, 0])
    ax1.set_yscale('log')
    ax1.set_title('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('loss')

    # subplot - loss_data
    ax2 = plt.subplot(132)
    ax2.plot(range(len(loss_all1)), loss_all1[:, 1])
    ax2.set_yscale('log')
    ax2.set_title('Data Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('loss_data')

    # subplot - loss_eqn
    ax3 = plt.subplot(133)
    ax3.plot(range(len(loss_all1)), loss_all1[:, 2])
    ax3.set_yscale('log')
    ax3.set_title('Eqn Loss')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('loss_eqn')

    plt.tight_layout()
    plt.show()

    np.savez('../data/loss_1.npz', loss=loss_all1)

    """Figure 5.1 - Loss_xy_l & Loss_e"""
    fig3 = plt.figure(figsize=(16, 5))

    # subplot - boundary loss - xy_l
    ax3 = plt.subplot(131)
    ax3.plot(range(len(loss_all1)), loss_all1[:, 3])
    ax3.set_yscale('log')
    ax3.set_title('Boundary Loss - xy_l')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss_xy_l')

    # subplot - boundary loss - xy_r
    ax3 = plt.subplot(132)
    ax3.plot(range(len(loss_all1)), loss_all1[:, 4])
    ax3.set_yscale('log')
    ax3.set_title('Boundary Loss - xy_r')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss_xy_r')

    plt.tight_layout()
    plt.show()

    loss_xy_l = loss_all1[:, 3]
    loss_xy_r = loss_all1[:, 4]

    np.savez(
        "../data/boundary_loss_1.npz",
        loss_xy_l=loss_xy_l,
        loss_xy_r=loss_xy_r
    )

    r1_rms = ms_error(F1) ** 0.5
    r1_rms = ms_error(r1_rms) ** 0.5
    e1_rms = ms_error(Error) ** 0.5
    e1_rms = ms_error(e1_rms) ** 0.5
    diff = r1_rms / e1_rms

    """Figure 6.1 - Frequency detection"""
    from scipy.fftpack import fft2, fftshift, fftfreq

    F1 = np.array(jnp.reshape(f_p1, R.shape))

    # Perform 2D FFT
    u_fft = fft2(F1)
    u_fft_shifted = fftshift(u_fft)  # Shift zero frequency to the center
    magnitude = np.abs(u_fft_shifted)

    # Get frequency components
    freq_x = fftshift(fftfreq(111, d=(r[1] - r[0])))
    freq_t = fftshift(fftfreq(111, d=(t[1] - t[0])))

    # Plot the frequency spectrum
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    contour = ax.contourf(freq_x, freq_t, np.log1p(magnitude), 100, cmap="jet")
    fig.colorbar(contour, ax=ax)
    ax.set_xlim(0, 5)  # Adjust x-axis range (frequency in t)
    ax.set_ylim(0, 5)  # Adjust y-axis range (frequency in x)
    ax.set_title("2D Frequency Spectrum Centered at Origin (0,0)")
    ax.set_xlabel("r")
    ax.set_ylabel("t")
    plt.show()

    log_mag = np.log1p(magnitude)
    np.savez(
        "../data/frequency_spectrum.npz",
        freq_x=freq_x,
        freq_t=freq_t,
        log_mag=log_mag
    )

    # 2nd stage neural network

    # select the size of neural network
    n_hl2 = 6
    n_unit2 = 50
    if e1_rms > 50:
        scl2 = 30
    else:
        scl2 = diff

    # set the training iteration
    lw2 = jnp.array([m_f / diff, m_df / (diff ** 2)])
    epsil2 = jnp.array([e1_rms])

    # initialize the weights and biases of the network
    trained_params2 = sol_init_MLP(keys[3], n_hl2, n_unit2)

    # create the solution function of 2nd stage
    pred_u2 = mNN_pred_create(f_u1, limit, scl2, epsil2, act_s=1)

    # create the data function
    dataf2 = data_func_create(N_col * 2, N_bd * 2, boundary, domain)
    Fs = R * 0 + 1
    key_adam = keys[4]
    key_lbfgs = random.split(keys[5], 1)
    data2 = dataf2(key_adam, Fs, R, T)

    z_c2 = data2['x_col']

    """Figure 1.2 - Collation Point"""
    # plot the collocation point
    plot = 1
    if plot == 1:
        colpoint_plot(Fs, z_c2, [domain["x_min"], domain["x_max"], domain["y_min"], domain["y_max"]], ['collo. point', '$t$', '$\th$'],
                      "collocation_point_2.npz")

    # calculate the loss function
    NN_loss = loss_create(pred_u2, lw2, loss_ref=1)
    NN_loss.ref = NN_loss(trained_params2, data2)[1][0]

    """Training using Adam"""

    # set the learning rate for Adam
    lr = 1e-3
    epoch1 = m_adam * 3
    trained_params2, loss1 = adam_optimizer(R, T, NN_loss, pred_u2, trained_params2, dataf2, Fs, epoch1, key_adam,
                                            lr=lr)
    Fs = predictF(pred_u2, trained_params2, R, T)
    data2 = dataf2(key_lbfgs[0], Fs, R, T)

    """Training using L-BFGS"""

    loss2 = []
    epoch2 = m_lbfgs * 3
    for l in range(1):
        trained_params2, loss = lbfgs_optimizer(NN_loss, trained_params2, data2, epoch2)
        Fs = predictF(pred_u2, trained_params2, R, T)
        data2 = dataf2(key_lbfgs[l], Fs, R, T)
        loss2 += loss

    # Result

    '''Calculating the output'''

    f_u2 = lambda z: pred_u2(trained_params2, z)

    # calculate the solution
    u_p2 = f_u2(X_star)
    f_p2 = gov_eqn(f_u2, X_star)

    # generate the last loss
    loss_all2 = np.array(loss1 + loss2)
    # loss_all2 = np.array(loss1)
    loss_all = np.vstack([loss_all1, loss_all2])

    U = jnp.reshape(u_p2, R.shape)
    F = jnp.reshape(f_p2, R.shape)

    """Figure 2.2 - Result + Loss-e"""
    fig1 = plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(121)
    contour1 = ax1.contourf(R, T, U, 100, cmap='jet')
    fig1.colorbar(contour1, ax=ax1)
    ax1.set_title('u')
    ax1.set_xlabel('r')
    ax1.set_ylabel('theta')

    ax2 = plt.subplot(122)
    contour2 = ax2.contourf(R, T, F, 100, cmap='jet')
    fig1.colorbar(contour2, ax=ax2)
    ax2.set_title('f')
    ax2.set_xlabel('r')
    ax2.set_ylabel('theta')

    plt.tight_layout()
    plt.show()

    r = np.array(R[:, 0])  # R.shape == (ny, nx)
    t = np.array(T[0, :])  # T.shape == (ny, nx)
    U_np = np.array(U)
    F_np = np.array(F)
    np.savez('../data/solution_residual_2.npz',
             r=r,
             t=t,
             U=U_np,
             F=F_np
             )

    """Figure 3.2 - error"""
    fig1 = plt.figure(figsize=(6, 5))
    Error = U - U_real
    ax3 = plt.subplot(111)
    contour3 = ax3.contourf(R, T, Error, 100, cmap='jet')
    fig1.colorbar(contour3, ax=ax3)
    ax3.set_title('Error')
    ax3.set_xlabel('r')
    ax3.set_ylabel('error')
    plt.tight_layout()
    plt.show()

    r = np.array(R[0, :])  # θ 网格横坐标
    t = np.array(T[:, 0])  # r 网格纵坐标
    Error_np = np.array(U - U_real)
    # 保存到 data/error_contour.npz
    np.savez(
        "../data/error_2.npz",
        r=r,  # shape (nx,)
        t=t,  # shape (ny,)
        Error=Error_np  # shape (ny, nx)
    )

    """Figure 4.2 - Loss_d + Loss_e"""
    fig2 = plt.figure(figsize=(16, 5))

    # subplot - loss
    ax1 = plt.subplot(131)
    ax1.plot(range(len(loss_all)), loss_all[:, 0])
    ax1.set_yscale('log')
    ax1.set_title('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('loss')

    # subplot - loss_data
    ax2 = plt.subplot(132)
    ax2.plot(range(len(loss_all)), loss_all[:, 1])
    ax2.set_yscale('log')
    ax2.set_title('Data Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('loss_data')

    # subplot - loss_eqn
    ax3 = plt.subplot(133)
    ax3.plot(range(len(loss_all)), loss_all[:, 2])
    ax3.set_yscale('log')
    ax3.set_title('Eqn Loss')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('loss_eqn')

    plt.tight_layout()
    plt.show()

    np.savez('../data/loss_2.npz', loss=loss_all)

    """Figure 5.2 - Loss_xy_l & Loss_xy_r"""
    fig3 = plt.figure(figsize=(16, 10))

    # subplot - boundary loss - xy_l
    ax3 = plt.subplot(231)
    ax3.plot(range(len(loss_all)), loss_all[:, 3])
    ax3.set_yscale('log')
    ax3.set_title('Boundary Loss - xy_l')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss_xy_l')

    # subplot - boundary loss - xy_r
    ax3 = plt.subplot(232)
    ax3.plot(range(len(loss_all)), loss_all[:, 4])
    ax3.set_yscale('log')
    ax3.set_title('Boundary Loss - xy_r')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss_xy_r')

    # subplot - initial loss _ xy_b_t
    ax3 = plt.subplot(233)
    ax3.plot(range(len(loss_all)), loss_all[:, 5])
    ax3.set_yscale('log')
    ax3.set_title('Initial Loss - xy_b_t')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss_xy_b_t')

    plt.tight_layout()
    plt.show()

    loss_xy_l = loss_all[:, 3]
    loss_xy_r = loss_all[:, 4]

    np.savez(
        "../data/boundary_loss_2.npz",
        loss_xy_l=loss_xy_l,
        loss_xy_r=loss_xy_r
    )


if __name__ == "__main__":
    params1 = {
        "equation": "test equation",
        "boundary": {
            "bd_x1_min": 0.1,
            "bd_x1_max": 0.1,
            "bd_y1_min": 0,
            "bd_y1_max": 1,
            "bd_u1": 1,
            "bd_x2_min": 1,
            "bd_x2_max": 1,
            "bd_y2_min": 0,
            "bd_y2_max": 1,
            "bd_u2": 0,
        },
        "domain": {
            "x_min": 0.1,
            "x_max": 1,
            "y_min": 0,
            "y_max": 1
        },
        "scl": 1,
        "epsil": 1,
    }
    params2 = {
        "sample_points": {
            "n_col": 3000,
            "n_bd": 1000,
            "n_add": 1000
        },
        "network_size": {
            "depth": 60,
            "width": 6
        },
        "testing_size": {
            "x": 111,
            "y": 111
        },
        "training_epoch": {
            "adam": 1,
            "lbfgs": 1
        },
        "equation_weight": {
            "f": 0.05,
            "df": 0
        },
    }
    run_pinn_training(
        equation=params1["equation"],
        boundary=params1["boundary"],
        domain=params1["domain"],
        scl=params1["scl"],
        epsil=params1["epsil"],
        sample_points=params2["sample_points"],
        network_size=params2["network_size"],
        testing_size=params2["testing_size"],
        epochs=params2["training_epoch"],
        equation_weight=params2["equation_weight"],
    )
