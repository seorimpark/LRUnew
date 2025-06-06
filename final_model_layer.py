from flax import nnx  # The Flax NNX API.
import jax.numpy as jnp  # JAX NumPy
import LRU
import layer_parameterization
import jax


class LRUMLP_layer(nnx.Module):
    # DON'T FORGET TO CHANGE THE MODEL NAME BEFORE RUNNING
    # According to the scheme of the paper (Figure 1), input_size=M, encoded_size=H,layer_dim=number of neurons in MLP, out_dim=number of classes
    def __init__(
        self,
        token_size,
        token_len,
        encoded_dim,
        hidden_dim,
        layer_dim,
        out_dim,
        r_min,
        r_max,
        max_phase,
        dropout,
        pool,
        lin_encoder_sigma,
        MLP1_sigma,
        MLP2_sigma,
        out_sigma,
        rngs: nnx.Rngs,
    ):

        self.lin_encoder = layer_parameterization.Linear_encoder(
            in_features=token_size,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=lin_encoder_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )

        self.rnn1 = LRU.LRU(
            in_features=encoded_dim,
            hidden_features=hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.rnn2 = LRU.LRU(
            in_features=encoded_dim,
            hidden_features=hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.rnn3 = LRU.LRU(
            in_features=encoded_dim,
            hidden_features=hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.rnn4 = LRU.LRU(
            in_features=encoded_dim,
            hidden_features=hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.rnn5 = LRU.LRU(
            in_features=encoded_dim,
            hidden_features=hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.rnn6 = LRU.LRU(
            in_features=encoded_dim,
            hidden_features=hidden_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
        )
        self.linear1_1 = layer_parameterization.Linear_MLP1(
            in_features=encoded_dim,
            out_features=layer_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP1_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear1_2 = layer_parameterization.Linear_MLP2(
            in_features=layer_dim // 2,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP2_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear2_1 = layer_parameterization.Linear_MLP1(
            in_features=encoded_dim,
            out_features=layer_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP1_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear2_2 = layer_parameterization.Linear_MLP2(
            in_features=layer_dim // 2,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP2_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear3_1 = layer_parameterization.Linear_MLP1(
            in_features=encoded_dim,
            out_features=layer_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP1_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear3_2 = layer_parameterization.Linear_MLP2(
            in_features=layer_dim // 2,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP2_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear4_1 = layer_parameterization.Linear_MLP1(
            in_features=encoded_dim,
            out_features=layer_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP1_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear4_2 = layer_parameterization.Linear_MLP2(
            in_features=layer_dim // 2,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP2_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear5_1 = layer_parameterization.Linear_MLP1(
            in_features=encoded_dim,
            out_features=layer_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP1_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear5_2 = layer_parameterization.Linear_MLP2(
            in_features=layer_dim // 2,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP2_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear6_1 = layer_parameterization.Linear_MLP1(
            in_features=encoded_dim,
            out_features=layer_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP1_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.linear6_2 = layer_parameterization.Linear_MLP2(
            in_features=layer_dim // 2,
            out_features=encoded_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=MLP2_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.batchnorm1 = nnx.BatchNorm(num_features=encoded_dim, rngs=rngs)
        self.batchnorm2 = nnx.BatchNorm(num_features=encoded_dim, rngs=rngs)
        self.batchnorm3 = nnx.BatchNorm(num_features=encoded_dim, rngs=rngs)
        self.batchnorm4 = nnx.BatchNorm(num_features=encoded_dim, rngs=rngs)
        self.batchnorm5 = nnx.BatchNorm(num_features=encoded_dim, rngs=rngs)
        self.batchnorm6 = nnx.BatchNorm(num_features=encoded_dim, rngs=rngs)

        # Linear layers
        if pool:  # If pooling layer takes the average over the token sequence length
            self.linear3 = lambda x: jnp.mean(x, axis=1)
        else:  # learn the parameters of the linear transformation
            self.linear3 = nnx.Linear(in_features=token_len, out_features=1, rngs=rngs)
        self.linear4 = layer_parameterization.Linear_out(
            in_features=encoded_dim,
            out_features=out_dim,
            rngs=rngs,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=out_sigma, mode="fan_in", distribution="truncated_normal"
            ),
        )
        self.out_dim = out_dim
        self.token_len = token_len
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    @nnx.vmap(in_axes=(None, 0))
    def block_after_batchnorm1(self, x):
        x = self.rnn1(x)
        x = self.linear1_1(x)
        x = nnx.glu(x, axis=-1)
        x = self.dropout(x)
        x = self.linear1_2(x)
        return x

    @nnx.vmap(in_axes=(None, 0))
    def block_after_batchnorm2(self, x):
        x = self.rnn2(x)
        x = self.linear2_1(x)
        x = nnx.glu(x, axis=-1)
        x = self.dropout(x)
        x = self.linear2_2(x)
        return x

    @nnx.vmap(in_axes=(None, 0))
    def block_after_batchnorm3(self, x):
        x = self.rnn3(x)
        x = self.linear3_1(x)
        x = nnx.glu(x, axis=-1)
        x = self.dropout(x)
        x = self.linear3_2(x)
        return x

    @nnx.vmap(in_axes=(None, 0))
    def block_after_batchnorm4(self, x):
        x = self.rnn4(x)
        x = self.linear4_1(x)
        x = nnx.glu(x, axis=-1)
        x = self.dropout(x)
        x = self.linear4_2(x)
        return x

    @nnx.vmap(in_axes=(None, 0))
    def block_after_batchnorm5(self, x):
        x = self.rnn5(x)
        x = self.linear5_1(x)
        x = nnx.glu(x, axis=-1)
        x = self.dropout(x)
        x = self.linear5_2(x)
        return x

    @nnx.vmap(in_axes=(None, 0))
    def block_after_batchnorm6(self, x):
        x = self.rnn6(x)
        x = self.linear6_1(x)
        x = nnx.glu(x, axis=-1)
        x = self.dropout(x)
        x = self.linear6_2(x)
        return x

    @nnx.vmap(in_axes=(None, 0))
    def final_linear_projections(self, x):
        x = self.linear3(x.T)
        x = self.linear4(x.T)
        return x.reshape(self.out_dim)

    def __call__(self, x):
        x = self.lin_encoder(x)
        y = x.copy()

        # LRU+MLP block*6
        x = self.batchnorm1(x)
        x = self.block_after_batchnorm1(x)
        x += y

        x = self.batchnorm2(x)
        x = self.block_after_batchnorm2(x)
        x += y

        x = self.batchnorm3(x)
        x = self.block_after_batchnorm3(x)
        x += y

        x = self.batchnorm4(x)
        x = self.block_after_batchnorm4(x)
        x += y

        x = self.batchnorm5(x)
        x = self.block_after_batchnorm5(x)
        x += y

        x = self.batchnorm6(x)
        x = self.block_after_batchnorm6(x)
        x += y

        return self.final_linear_projections(x)
