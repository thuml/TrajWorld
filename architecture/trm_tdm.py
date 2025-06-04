from typing import Any, Dict, NamedTuple, Tuple, Sequence, List
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

class Attention(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float
    causal: float

    def setup(self) -> None:
        self.Dense_0 = nn.Dense(self.h_dim)
        self.Dense_1 = nn.Dense(self.h_dim)
        self.Dense_2 = nn.Dense(self.h_dim)
        self.Dense_3 = nn.Dense(self.h_dim)

        self.attn_drop = nn.Dropout(self.drop_p)
        self.resid_drop = nn.Dropout(self.drop_p)

    def __call__(self, x: jnp.ndarray, padding_mask: jnp.ndarray, training=True) -> jnp.ndarray:
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        # rearrange q, k, v as (B, N, T, D)
        q = self.Dense_0(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = self.Dense_1(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = self.Dense_2(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        # weights (B, N, T, T) jax
        weights = jnp.einsum("bntd,bnfd->bntf", q, k) / jnp.sqrt(D)
        if self.causal:
            # causal mask applied to weights
            ones = jnp.ones((self.max_T, self.max_T))
            mask = jnp.tril(ones).reshape(1, 1, self.max_T, self.max_T)
            weights = jnp.where(mask[..., :T, :T] == 0, -jnp.inf, weights[..., :T, :T])
        # apply padding mask (B, T)
        weights = jnp.where(padding_mask[:, None, None, :T] == 0, -1e4, weights)  # TODO: fix magic number
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)
        # attention (B, N, T, D)
        attention = self.attn_drop(
            jnp.einsum("bntf,bnfd->bntd", normalized_weights, v),
            deterministic=not training,
        )
        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N * D)
        out = self.resid_drop(
            self.Dense_3(attention),
            deterministic=not training
        )
        return out

    def call_kv_cache(self,
                      x: jnp.ndarray, padding_mask: jnp.ndarray,
                      k_cache: jnp.ndarray, v_cache: jnp.ndarray, padding_mask_cache: jnp.ndarray,
                      training=False) -> jnp.ndarray:
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        t = k_cache.shape[2]

        # rearrange q, k, v as (B, N, T, D)
        q = self.Dense_0(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = self.Dense_1(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = self.Dense_2(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        # concat cache
        k = jnp.concatenate([k_cache, k], axis=2)  # (B, N, t+T, D)
        v = jnp.concatenate([v_cache, v], axis=2)  # (B, N, t+T, D)
        padding_mask = jnp.concatenate([padding_mask_cache, padding_mask], axis=1)  # (B, t+T)
        # weights (B, N, T, t+T) jax
        weights = jnp.einsum("bntd,bnfd->bntf", q, k) / jnp.sqrt(D)
        if self.causal:
            # causal mask applied to weights
            ones = jnp.ones((self.max_T, self.max_T))
            mask = jnp.tril(ones).reshape(1, 1, self.max_T, self.max_T)
            weights = jnp.where(mask[..., t:t + T, :t + T] == 0, -jnp.inf, weights[..., :T, :t + T])
        # apply padding mask (B, T)
        weights = jnp.where(padding_mask[:, None, None, :t + T] == 0, -1e4, weights)  # TODO: fix magic number
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)
        # attention (B, N, T, D)
        attention = self.attn_drop(
            jnp.einsum("bntf,bnfd->bntd", normalized_weights, v),
            deterministic=not training,
        )
        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N * D)
        out = self.resid_drop(
            self.Dense_3(attention),
            deterministic=not training
        )
        return out, k, v, padding_mask


class Block(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float

    def setup(self) -> None:
        self.Attention_1 = Attention(
            self.h_dim, self.max_T, self.n_heads, self.drop_p, causal=True
        )
        self.LayerNorm_1 = nn.LayerNorm()

        self.Dense_2 = nn.Dense(4 * self.h_dim)
        self.Dense_3 = nn.Dense(self.h_dim)
        self.out_drop_1 = nn.Dropout(self.drop_p)
        self.LayerNorm_3 = nn.LayerNorm()

    def __call__(self, x: jnp.ndarray, padding_mask: jnp.ndarray, training=True) -> jnp.ndarray:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        """
        B, T, M, d = x.shape

        # Attention ->  LayerNorm -> FFN -> LayerNorm
        # x = jnp.einsum("btmd->bmtd", x).reshape(B * M, T, d)  # btmd->(bm)td
        # [M0, M1, M2, M3]
        x = x.reshape(B, T * M, d)  # (B, T, M, d) -> (B, T*M, d)
        padding_mask = jnp.repeat(padding_mask, M, axis=1)
        x = x + self.Attention_1(x, padding_mask=padding_mask, training=training)  # residual
        x = self.LayerNorm_1(x)

        out = self.Dense_2(x)
        out = nn.gelu(out)
        out = self.Dense_3(out)
        out = self.out_drop_1(out, deterministic=not training)
        x = x + out
        x = self.LayerNorm_3(x)
        x = x.reshape(B, T, M, d)
        return x

    def call_kv_cache(self, x: jnp.ndarray, padding_mask: jnp.ndarray,
                      k_cache: jnp.ndarray, v_cache: jnp.ndarray, padding_mask_cache: jnp.ndarray,
                      training=False) -> jnp.ndarray:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        k_cache: (B*M, N, t, D)
        v_cache: (B*M, N, t, D)
        padding_mask_cache: (B, T)
        """
        B, T, M, d = x.shape

        # Attention ->  LayerNorm -> FFN -> LayerNorm
        # [M0, M1, M2, M3]
        x = x.reshape(B, T * M, d)  # (B, T, M, d) -> (B, T*M, d)
        padding_mask = jnp.repeat(padding_mask, M, axis=1)
        attn_out, k_cache, v_cache, padding_mask_cache = self.Attention_1.call_kv_cache(
            x, padding_mask=padding_mask,
            k_cache=k_cache, v_cache=v_cache, padding_mask_cache=padding_mask_cache,
            training=training)  # kv cache
        x = x + attn_out  # residual
        x = self.LayerNorm_1(x)

        out = self.Dense_2(x)
        out = nn.gelu(out)
        out = self.Dense_3(out)
        out = self.out_drop_1(out, deterministic=not training)
        x = x + out
        x = self.LayerNorm_3(x)

        x = x.reshape(B, T, M, d)
        return x, (k_cache, v_cache, padding_mask_cache)

    def call_variate_mask(self, x: jnp.ndarray, padding_mask: jnp.ndarray, variate_mask: jnp.ndarray,
                          training=True) -> jnp.ndarray:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        variate_mask: (B, M)
        """
        B, T, M, d = x.shape
        # Temporal Attention -> (FFN ->) LayerNorm -> Spatial Attention -> LayerNorm -> FFN -> LayerNorm
        # [M0, M1, M2, M3]
        x = x.reshape(B, T * M, d)  # (B, T, M, d) -> (B, T*M, d)
        padding_mask = jnp.repeat(padding_mask, M, axis=1)
        x = x + self.Attention_1(x, padding_mask=padding_mask, training=training)  # residual
        x = self.LayerNorm_1(x)

        out = self.Dense_2(x)
        out = nn.gelu(out)
        out = self.Dense_3(out)
        out = self.out_drop_1(out, deterministic=not training)
        x = x + out
        x = self.LayerNorm_3(x)

        x = x.reshape(B, T, M, d)
        return x


class TDMTransformer(nn.Module):
    vocab_size: int
    n_blocks: int
    h_dim: int
    n_heads: int
    drop_p: float
    max_timestep: int = 4096
    use_variate_embed: bool = True
    shuffle_variate: bool = False
    mask_ratio: float = 0.0
    prompt: bool = False  # whether to enable prompt
    freeze_params = None

    def setup(self) -> None:
        self.embed = nn.Embed(self.vocab_size, self.h_dim)
        self.embed_proj = nn.Dense(self.h_dim)
        self.embed_obs_act = nn.Embed(2, self.h_dim)  # 0 for obs, 1 for act
        self.embed_timestep = nn.Embed(self.max_timestep, self.h_dim)

        self.blocks = [
            Block(self.h_dim, self.max_timestep, self.n_heads, self.drop_p)
            for _ in range(self.n_blocks)
        ]
        self.head = nn.Dense(self.vocab_size)

    def set_freeze_params(self, freeze_params):
        self.freeze_params = freeze_params

    def __call__(
            self,
            inputs: jnp.ndarray,
            obs_act_indicator: jnp.ndarray,
            padding_mask: jnp.ndarray,
            training=True,
            variate_key: jax.random.PRNGKey = None,
            prompt=Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        inputs: (B, T, M, d)
        obs_act_indicator: (B, T, M)
        padding_mask: (B, T)
        """
        # embedded = self.embed(inputs)
        embedded = self.embed_proj(inputs)
        if self.mask_ratio > 0.0 and training:
            variate_key, mask_key = jax.random.split(variate_key)
            mask = jax.random.bernoulli(mask_key, self.mask_ratio, inputs.shape[:-1])
            embedded = jnp.where(mask[..., None], 0.0, embedded)
        embedded += self.embed_obs_act(obs_act_indicator)
        timesteps = jnp.arange(inputs.shape[1])
        embedded += self.embed_timestep(timesteps)[:, None, :]

        h = embedded  # (B, T, M, d)

        for block in self.blocks:
            h = block(h, padding_mask=padding_mask, training=training)
        pred = self.head(h)
        return pred

    def call_variate_mask(
            self,
            inputs: jnp.ndarray,
            obs_act_indicator: jnp.ndarray,
            padding_mask: jnp.ndarray,
            variate_mask: jnp.ndarray,
            training=True,
            variate_key: jax.random.PRNGKey = None,
            prompt=Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        inputs: (B, T, M, d)
        obs_act_indicator: (B, T, M)
        padding_mask: (B, T)
        """
        # embedded = self.embed(inputs)
        embedded = self.embed_proj(inputs)
        if self.mask_ratio > 0.0 and training:
            variate_key, mask_key = jax.random.split(variate_key)
            mask = jax.random.bernoulli(mask_key, self.mask_ratio, inputs.shape[:-1])
            embedded = jnp.where(mask[..., None], 0.0, embedded)
        embedded += self.embed_obs_act(obs_act_indicator)
        timesteps = jnp.arange(inputs.shape[1])
        embedded += self.embed_timestep(timesteps)[:, None, :]

        h = embedded  # (B, T, M, d)
        for block in self.blocks:
            h = block.call_variate_mask(h, padding_mask=padding_mask, variate_mask=variate_mask, training=training)
        pred = self.head(h)
        return pred

    def call_kv_cache(
            self,
            inputs: jnp.ndarray, obs_act_indicator: jnp.ndarray, padding_mask: jnp.ndarray,
            caches: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
            training=False,
            variate_key: jax.random.PRNGKey = None,
            prompt=Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        cache: Tuple of
            k_caches: List[jnp.ndarray] of shape (B*M, N, t, D)
            v_caches: List[jnp.ndarray] of shape (B*M, N, t, D)
            padding_mask_caches: List[jnp.ndarray] of shape (B, t)
        """
        # embedded = self.embed(inputs)
        embedded = self.embed_proj(inputs)
        if self.mask_ratio > 0.0 and training:
            variate_key, mask_key = jax.random.split(variate_key)
            mask = jax.random.bernoulli(mask_key, self.mask_ratio, inputs.shape[:-1])
            embedded = jnp.where(mask[..., None], 0.0, embedded)
        embedded += self.embed_obs_act(obs_act_indicator)
        N0 = inputs.shape[2]
        timesteps = jnp.arange(caches[0][0].shape[2] // N0, (caches[0][0].shape[2] + inputs.shape[1] * inputs.shape[2]) // N0)
        embedded += self.embed_timestep(timesteps)[:, None, :]

        h = embedded  # (B, T, M, d)

        updated_caches = []
        for i, block in enumerate(self.blocks):
            h, updated_cache = block.call_kv_cache(h, padding_mask=padding_mask,
                                                   k_cache=caches[i][0], v_cache=caches[i][1],
                                                   padding_mask_cache=caches[i][2],
                                                   training=training)
            updated_caches.append(updated_cache)
        pred = self.head(h)
        return pred, updated_caches

    def get_empty_cache(self, batch_size: int) -> List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        caches = []
        for _ in range(self.n_blocks):
            caches.append((jnp.zeros((batch_size, self.n_heads, 0, self.h_dim // self.n_heads)),
                           jnp.zeros((batch_size, self.n_heads, 0, self.h_dim // self.n_heads)),
                           jnp.zeros((batch_size, 0))))
        return caches