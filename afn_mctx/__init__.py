# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""AFN MCTX: Monte Carlo tree search for AFlowNets in JAX."""

from afn_mctx._src.action_selection import gumbel_muzero_interior_action_selection
from afn_mctx._src.action_selection import gumbel_muzero_root_action_selection
from afn_mctx._src.action_selection import GumbelMuZeroExtraData
from afn_mctx._src.action_selection import muzero_action_selection
from afn_mctx._src.base import ChanceRecurrentFnOutput
from afn_mctx._src.base import DecisionRecurrentFnOutput
from afn_mctx._src.base import InteriorActionSelectionFn
from afn_mctx._src.base import LoopFn
from afn_mctx._src.base import PolicyOutput
from afn_mctx._src.base import RecurrentFn
from afn_mctx._src.base import RecurrentFnOutput
from afn_mctx._src.base import RecurrentState
from afn_mctx._src.base import RootActionSelectionFn
from afn_mctx._src.base import RootFnOutput
from afn_mctx._src.policies import gumbel_aflownet_policy
from afn_mctx._src.qtransforms import qtransform_by_completion
from afn_mctx._src.qtransforms import qtransform_by_min_max
from afn_mctx._src.qtransforms import qtransform_by_parent_and_siblings
from afn_mctx._src.qtransforms import qtransform_completed_by_mix_value
from afn_mctx._src.search import search
from afn_mctx._src.tree import Tree

__version__ = "0.0.3"

__all__ = (
    "ChanceRecurrentFnOutput",
    "DecisionRecurrentFnOutput",
    "GumbelMuZeroExtraData",
    "InteriorActionSelectionFn",
    "LoopFn",
    "PolicyOutput",
    "RecurrentFn",
    "RecurrentFnOutput",
    "RecurrentState",
    "RootActionSelectionFn",
    "RootFnOutput",
    "Tree",
    "gumbel_muzero_interior_action_selection",
    "gumbel_muzero_policy",
    "gumbel_aflownet_policy",
    "gumbel_muzero_root_action_selection",
    "muzero_action_selection",
    "muzero_policy",
    "qtransform_by_completion",
    "qtransform_by_min_max",
    "qtransform_by_parent_and_siblings",
    "qtransform_completed_by_mix_value",
    "search",
    "stochastic_muzero_policy",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Mctx public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
