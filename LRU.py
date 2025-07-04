import jax
import numpy as np
from flax import nnx  # The Flax NNX API.
import jax.numpy as jnp  # JAX NumPy

parallel_scan = jax.lax.associative_scan

# From Orvieto et al., 2023, (https://arxiv.org/abs/2303.06349)


def forward(lru_parameters, input_sequence):
    """Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."""

    # All LRU parameters
    nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = lru_parameters

    # Materializing the diagonal of Lambda and projections
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    C = C_re + 1j * C_im

    # Running the LRU + output projection
    # For details on parallel scan, check discussion in Smith et al (2022).
    Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = parallel_scan(binary_operator_diag, elements)  # all x_k
    y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)

    return y


def init_lru_parameters(N, H, r_min=0, r_max=1, max_phase=0.314):
    """Initialize parameters of the LRU layer."""

    # N: state dimension, H: model dimension
    # Initialization of Lambda is complex valued distributed uniformly on ring
    # between r_min and r_max, with phase in [0, max_phase].
    u1 = np.random.uniform(size=(N,))
    u2 = np.random.uniform(size=(N,))
    nu_log = np.log(-0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    theta_log = np.log(max_phase * u2)

    # Glorot initialized Input/Output projection matrices
    B_re = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
    B_im = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
    C_re = np.random.normal(size=(H, N)) / np.sqrt(N)
    C_im = np.random.normal(size=(H, N)) / np.sqrt(N)
    D = np.random.normal(size=(H,))

    # Normalization factor
    diag_lambda = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
    gamma_log = np.log(np.sqrt(1 - np.abs(diag_lambda) ** 2))

    return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log


def binary_operator_diag(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


Array = jax.Array


class LRU(nnx.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,  # not inferred from carry for now
        *,
        r_min=0,
        r_max=1,
        max_phase=6.28,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = init_lru_parameters(
            hidden_features, in_features, r_min=r_min, r_max=r_max, max_phase=max_phase
        )

        self.nu_log = nnx.Param(nu_log)
        self.theta_log = nnx.Param(theta_log)
        self.B_re = nnx.Param(B_re)
        self.B_im = nnx.Param(B_im)
        self.C_re = nnx.Param(C_re)
        self.C_im = nnx.Param(C_im)
        self.D = nnx.Param(D)
        self.gamma_log = nnx.Param(gamma_log)

    def __call__(self, inputs: Array):  # type: ignore[override]
        # jax.debug.print("test:{}", jnp.sin(self.nu_log + self.theta_log))
        Lambda = jnp.exp(
            -jnp.exp(self.nu_log.value) + 1j * jnp.exp(self.theta_log.value)
        )
        B_norm = (self.B_re.value + 1j * self.B_im.value) * jnp.expand_dims(
            jnp.exp(self.gamma_log.value), axis=-1
        )
        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(Lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        elements = (Lambda_elements, Bu_elements)
        C = self.C_re + 1j * self.C_im
        _, h = parallel_scan(binary_operator_diag, elements)
        y = jax.vmap(lambda x, u: (C @ x).real + self.D * u)(h, inputs)
        return y
