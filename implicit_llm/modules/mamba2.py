"""
This file is adapted from the states-spaces/mamba repository.
Copyright (c) 2024, Tri Dao, Albert Gu.

The original license is as follows:
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2023 Tri Dao, Albert Gu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)
print(f"Fast path available: {is_fast_path_available}")

from ..modules.dropout import VariationalDropout1d
from ..utils import apply_gaussian_noise


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class Mamba2Cache:
    """
    Inference cache for our Mamba2 implementation.

    Arguments:
        layers:      list of Mamba2 layers (e.g. model.layers)
        batch_size:  int, generation batch size
        dtype:       torch.dtype, cache tensor dtype (default: torch.float16)
        device:      torch.device or string
    Attributes:
        conv_states:  Tensor of shape
                      [num_layers, batch_size, conv_dim, kernel_size]
        ssm_states:   Tensor of shape
                      [num_layers, batch_size, nheads, headdim, d_state]
    """

    def __init__(self, layers, batch_size, dtype=torch.float16, device=None):
        # pick parameters from the first layer
        layer0 = layers[0].mixer
        self.dtype = dtype
        self.num_layers = len(layers)
        # convolution buffer dimensions
        self.conv_kernel_size = layer0.d_conv
        self.conv_dim = layer0.d_ssm + 2 * layer0.ngroups * layer0.d_state
        # SSM buffer dimensions
        self.nheads = layer0.nheads
        self.headdim = layer0.headdim
        self.d_state = layer0.d_state

        # allocate storage
        # conv_states: for each layer, keep last `kernel_size` inputs of shape conv_dim
        self.conv_states = torch.zeros(
            self.num_layers,
            batch_size,
            self.conv_dim,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        # ssm_states: for each layer, keep the current SSM state
        self.ssm_states = torch.zeros(
            self.num_layers,
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=dtype,
        )

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ):
        """
        Update the convolution circular buffer for layer `layer_idx`.
        - new_conv_state should be shape [batch, conv_dim, kernel_size] if cache_init=True,
          or [batch, conv_dim] (the newest time-step) if cache_init=False.
        """
        if cache_init:
            # initialize full buffer
            # expect new_conv_state.shape == (B, conv_dim, W)
            self.conv_states[layer_idx].copy_(
                new_conv_state.to(
                    self.conv_states.dtype, device=self.conv_states.device
                )
            )
        else:
            # shift left and insert new last column
            # new_conv_state: (B, conv_dim)
            buf = self.conv_states[layer_idx]
            buf[:, :, :-1] = buf[:, :, 1:]
            buf[:, :, -1] = new_conv_state.to(buf.dtype, device=buf.device)

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        """
        Replace the SSM state for layer `layer_idx`.
        - new_ssm_state should be shape [batch, nheads, headdim, d_state].
        """
        self.ssm_states[layer_idx].copy_(
            new_ssm_state.to(self.ssm_states.dtype, device=self.ssm_states.device)
        )


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        dropout=0.0,
        A_init_range=(1, 16),
        # rescale A to 2A-1 (see mononoid-word problem response paper)
        allow_negative_A_EV=False,
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        state_noise_db=20.0,
        latent_noise_db=20.0,
        noise_mode="multiplicative",  # 'additive' or 'multiplicative'
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        use_materializing_scan=False,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # inject noise
        self.state_noise_db = state_noise_db
        self.latent_noise_db = latent_noise_db
        self.noise_mode = noise_mode

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.use_pscan = use_materializing_scan
        assert not (self.use_mem_eff_path and self.use_pscan)
        self.layer_idx = layer_idx
        #
        self.allow_negative_A_EV = allow_negative_A_EV
        if allow_negative_A_EV:
            assert self.use_pscan, "Only support allow_negative_A_EV with pscan"
        # Order: [z, x, B, C, dt]
        self.d_in_proj = (
            2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        )
        if self.process_group is None:
            self.in_proj = nn.Linear(
                self.d_model, self.d_in_proj, bias=bias, **factory_kwargs
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                self.d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D._no_weight_decay = True

        # Dropout
        self.drop = VariationalDropout1d(dropout=dropout)

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        if self.process_group is None:
            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

    def forward(
        self,
        u: Tensor,
        injected_inputs: Optional[Tensor] = None,
        seqlen: Optional[int] = None,
        seq_idx: Optional[int] = None,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        skip_kv_update: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Streamlined forward supporting both full-sequence and step-wise inference via cache_params.
        Args:
            u: (batch, seqlen, d_model) or flattened (batch*seqlen, d_model)
            injected_inputs: optional same-shape additional projection
            seqlen: original sequence length when u is flattened
            seq_idx: index for sequence-parallel prefix
            cache_params: Mamba2Cache instance holding states
            cache_position: LongTensor with current generation position
            attention_mask: padding mask
        Returns:
            Tensor of same shape as u
        """
        # ---------------------------------------------
        # 1. Handle step-wise inference via cache
        # ---------------------------------------------
        if (
            cache_params is not None
            and cache_position is not None
            and cache_position[0] > 0
        ):
            # apply noise to states and inputs
            apply_gaussian_noise(u, snr_db=self.latent_noise_db, mode=self.noise_mode)

            # extract per-layer states
            conv_state = cache_params.conv_states[self.layer_idx]
            ssm_state = cache_params.ssm_states[self.layer_idx]
            out, (new_conv, new_ssm) = self.step(
                u, injected_inputs, conv_state, ssm_state
            )
            # update cache
            if (
                not skip_kv_update
            ):  # implicit models only update kv cache on last iteration
                cache_params.conv_states[self.layer_idx].copy_(new_conv)
                cache_params.ssm_states[self.layer_idx].copy_(new_ssm)

                apply_gaussian_noise(
                    cache_params.conv_states[self.layer_idx],
                    self.state_noise_db,
                    mode=self.noise_mode,
                )
                apply_gaussian_noise(
                    cache_params.ssm_states[self.layer_idx],
                    self.state_noise_db,
                    mode=self.noise_mode,
                )
            return out

        # ---------------------------------------------
        # 2. Full-sequence inference or training
        # ---------------------------------------------
        # reshape flattened input
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, _ = u.shape  # u: (B, L, D)
        else:
            batch_seqlen, _ = u.shape  # u: (B*L, D)
            batch = batch_seqlen // seqlen

        # project inputs
        zxbcdt = self.in_proj(u)  # (..., d_in_proj)
        if injected_inputs is not None:
            zxbcdt = zxbcdt + injected_inputs
        if seqlen_og is not None:
            # (B*L, D') -> (B, L, D')
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # prepare A and dt limits
        A = -torch.exp(self.A_log)
        dt_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        # fused mem-efficient path
        if self.use_mem_eff_path and cache_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=(
                    rearrange(self.D, "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.D
                ),
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            # split projected states: z0, x0, z, xBC, dt
            d_mlp = (
                zxbcdt.shape[-1]
                - 2 * self.d_ssm
                - 2 * self.ngroups * self.d_state
                - self.nheads
            ) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.d_ssm,
                    self.d_ssm + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )
            # conv transformation
            if cache_params is not None:
                # init cache at seq start
                xBC_t = rearrange(xBC, "b l d -> b d l")
                cache_params.conv_states[self.layer_idx].copy_(
                    F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                )
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1,
            )
            # SSM transformation
            out_ssm, last_state = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=(
                    rearrange(self.D, "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.D
                ),
                z=(
                    rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
                    if not self.rmsnorm
                    else None
                ),
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                return_final_states=(cache_params is not None),
                **dt_kwargs,
            )
            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(last_state)
            y = rearrange(out_ssm, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        # dropout
        return self.drop(out)

    def step(
        self,
        hidden_states: Tensor,
        injected_inputs: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Sequentially step through a sequence and carry over internal states for convolutions and state-space models.
        Args:
            hidden_states: input to the Mamba2 layer (B, D_model)
            injected_inputs: injected input to the Mamba2 layer (B, D_in_proj)
            conv_state: carry for convolution (B, D_conv, W)
            ssm_state: carry for state-space model (B, nheads, headdim, D_state)

        Returns:
            out: output for this step (B, D_model)
            new_conv_state: updated convolution state (B, D_conv, W)
            new_ssm_state: updated state-space model state (B, nheads, headdim, D_state)1

        """
        dtype = hidden_states.dtype

        if hidden_states.dim() > 2:
            assert (
                hidden_states.shape[1] == 1
            ), "Only support decoding with 1 token at a time for now"
            hidden_states = hidden_states.squeeze(1)
        zxbcdt = self.in_proj(hidden_states)  # (B 2D)

        # inject inputs
        if injected_inputs is not None:
            if injected_inputs.dim() > 2:
                assert (
                    injected_inputs.shape[1] == 1
                ), "Only support decoding with 1 token at a time for now"
                injected_inputs = injected_inputs.squeeze(1)
            zxbcdt += injected_inputs

        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # Conv step
        new_conv_state = torch.roll(
            conv_state, shifts=-1, dims=-1
        )  # Update state (B D W)
        new_conv_state[:, :, -1] = xBC
        xBC = torch.sum(
            new_conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )  # (B D)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC).to(dtype=dtype)

        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
        # Discretize A and B
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        new_ssm_state = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx
        y = torch.einsum("bhpn,bn->bhp", new_ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        if not self.rmsnorm:
            y = y * self.act(z)  # (B D)
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)

        return out, (new_conv_state, new_ssm_state)

    def allocate_inference_cache(self, batch_size, device, dtype=None, **kwargs):
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.conv1d.weight.shape[0],
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
