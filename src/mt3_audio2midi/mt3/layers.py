# Copyright 2024 The mt3_audio2midi.mt3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np


# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


def sinusoidal(min_scale: float = 1.0,
               max_scale: float = 10000.0,
               dtype: DType = jnp.float32) -> Initializer:
  """Creates 1D Sinusoidal Position Embedding Initializer.

  Args:
    min_scale: Minimum frequency-scale in sine grating.
    max_scale: Maximum frequency-scale in sine grating.
    dtype: The DType of the returned values.

  Returns:
    The sinusoidal initialization function.
  """

  def init(key: PRNGKey, shape: Shape, dtype: DType = dtype) -> Array:
    """Sinusoidal init."""
    del key
    if dtype != np.float32:
      raise ValueError('The sinusoidal initializer only supports float32.')
    if len(list(shape)) != 2:
      raise ValueError(
          f'Expected a 2D shape (max_len, features), but got {shape}.')
    max_len, features = shape
    pe = np.zeros((max_len, features), dtype=dtype)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, features // 2) * scale_factor)
    pe[:, :features // 2] = np.sin(position * div_term)
    pe[:, features // 2:2 * (features // 2)] = np.cos(position * div_term)
    return jnp.array(pe)

  return init


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'joined_kv'),
        dtype=self.dtype)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        batch, num_heads, head_dim, length = (cached_key.value.shape)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))

        # Create a OHE of the current index. NOTE: the index is increased below.
        cur_index = cache_index.value
        one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
        # In order to update the key, value caches with the current key and
        # value, we move the length axis to the back, similar to what we did for
        # the cached ones above.
        # Note these are currently the key and value of a single position, since
        # we feed one position at a time.
        one_token_key = jnp.moveaxis(key, -3, -1)
        one_token_value = jnp.moveaxis(value, -3, -1)
        # Update key, value caches with our new 1d spatial slices.
        # We implement an efficient scatter into the cache via one-hot
        # broadcast and addition.
        key = cached_key.value + one_token_key * one_hot_indices
        value = cached_value.value + one_token_value * one_hot_indices
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # Move the keys and values back to their original shapes.
        key = jnp.moveaxis(key, -1, -3)
        value = jnp.moveaxis(value, -1, -3)

        # Causal mask for cached decoder self-attention: our single query
        # position should only attend to those key positions that have already
        # been generated and cached, not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(length) <= cur_index,
                # (1, 1, length) represent (head dim, query length, key length)
                # query length is 1 because during decoding we deal with one
                # index.
                # The same mask is applied to all batch elements and heads.
                (batch, 1, 1, length)))

        # Grab the correct relative attention bias during decoding. This is
        # only required during single step decoding.
        if bias is not None:
          # The bias is a full attention matrix, but during decoding we only
          # have to take a slice of it.
          # This is equivalent to bias[..., cur_index:cur_index+1, :].
          bias = dynamic_vector_slice_in_dim(
              jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=('joined_kv', 'embed'),
        dtype=self.dtype,
        name='out')(
            x)
    return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseGeneral(
          self.intermediate_dim,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          kernel_axes=('embed', 'mlp'),
          name=dense_name)(
              inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo')(
            x)
    return output


class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        self.embedding_init, (self.num_embeddings, self.features),
        jnp.float32,
        axes=('vocab', 'embed'))

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class FixedEmbed(nn.Module):
  """Fixed (not learnable) embeddings specified by the initializer function.

  Attributes:
    init_fn: The initializer function that defines the embeddings.
    max_length: The maximum supported length.
    dtype: The DType to use for the embeddings.
  """
  features: int
  max_length: int = 2048
  embedding_init: Initializer = sinusoidal()
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    # The key is set to None because sinusoid init is deterministic.
    shape = (self.max_length, self.features)
    self.embedding = self.embedding_init(None, shape, self.dtype)  # pylint: disable=too-many-function-args  # pytype: disable=wrong-arg-types  # jax-ndarray

  @nn.compact
  def __call__(self,
               inputs,
               *,
               decode: bool = False):
    """Returns the fixed position embeddings specified by the initializer.

    Args:
      inputs: <int>[batch_size, seq_len] input position indices.
      decode: True if running in single-position autoregressive decode mode.

    Returns:
      The fixed position embeddings <float32>[batch_size, seq_len, features].
    """
    # We use a cache position index for tracking decoding position.
    if decode:
      position_embedder_index = self.variable(
          'cache', 'position_embedder_index',
          lambda: jnp.array(-1).astype(jnp.uint32))
      i = position_embedder_index.value
      position_embedder_index.value = i + 1
      return jax.lax.dynamic_slice(self.embedding, jnp.array((i, 0)),
                                   np.array((1, self.features)))

    return jnp.take(self.embedding, inputs, axis=0)


#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------
class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes(
        'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: DType = jnp.float32) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
      # [batch, len_q] -> [batch, len_q, 1]
      jnp.expand_dims(query_input, axis=-1),
      # [batch, len_q] -> [batch, 1, len_kv]
      jnp.expand_dims(key_input, axis=-2))

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, length, length]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return combine_masks(*masks, dtype=dtype)  # pytype: disable=bad-return-type  # jax-ndarray
