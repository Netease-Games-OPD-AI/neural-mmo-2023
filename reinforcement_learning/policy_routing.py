import argparse
import math

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn.parameter import Parameter
from torch.nn import init

from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models

import nmmo
from nmmo.entity.entity import EntityState
from reinforcement_learning.policy import (
    TileEncoder,
    PlayerEncoder,
    ItemEncoder,
    InventoryEncoder,
    MarketEncoder,
    TaskEncoder,
    ActionDecoder,
)

EntityId = EntityState.State.attr_name_to_col["id"]


class PolicyRoutingModel(pufferlib.models.Policy):
    """Multi-Task Reinforcement Learning with Soft Modularization"""

    L = 2
    n = 2

    def __init__(self, env, input_size=256, hidden_size=256, task_size=4096):
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.item_encoder = ItemEncoder(input_size, hidden_size)
        self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
        self.proj_fc = torch.nn.Linear(4 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

        self.input_size = input_size

        self.routing_layers = ModuleList(
            [RoutingLayer(input_size, self.n, l) for l in range(self.L)]
        )

        self.base_policy_layers = ModuleList(
            [BasePolicyLayer(input_size, self.n) for l in range(self.L)]
        )

        self.g_fc = torch.nn.Linear(input_size, self.n * input_size)
        self.hidden_fc = torch.nn.Linear(self.n, 1)

    def encode_observations(self, flat_observations):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            flat_observations,
            self.flat_observation_space,
            self.flat_observation_structure,
        )
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )

        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        inventory = self.inventory_encoder(item_embeddings)

        market_embeddings = self.item_encoder(env_outputs["Market"])
        market = self.market_encoder(market_embeddings)

        # inputs of base policy network & routing network
        # shape (BS, input_size)
        state_enc = torch.cat([tile, my_agent, inventory, market], dim=-1)
        state_enc = self.proj_fc(state_enc)
        task_enc = self.task_encoder(env_outputs["Task"])

        # shape (BS, input_size)
        state_mul_task = state_enc * task_enc

        batch_size, _ = state_mul_task.shape

        p_l = None
        gs = self.g_fc(state_enc)
        gs = gs.view(batch_size, self.n, self.input_size)
        for l in range(self.L):
            # shape (BS, n^2)
            p_l = self.routing_layers[l](state_mul_task, p_l)

            # shape (BS, n, n)
            p_l_mat = p_l.view(batch_size, self.n, self.n)
            p_l_softmax = torch.softmax(p_l_mat, dim=-1)

            # shape (BS, n, input_size)
            gs = self.base_policy_layers[l](gs, p_l_softmax)

        # shape (BS, input_size)
        hidden = self.hidden_fc(gs.transpose(-1, -2)).squeeze(-1)

        return hidden, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value


class RoutingLayer(torch.nn.Module):
    def __init__(self, input_size, n, l):
        """
        input_size: D
        """
        super().__init__()
        self.l = l

        n2 = n**2

        if self.l > 0:
            self.w_u_fc = torch.nn.Linear(n2, input_size, bias=False)

        self.w_d_fc = torch.nn.Linear(input_size, n2, bias=False)

    def forward(self, state_mul_task, p_l):
        """
        state_mul_task: shape (BS, input_size)
        p_l: shape (BS, n^2)
        """
        x = state_mul_task
        if self.l > 0:
            assert p_l is not None
            # shape (BS, input_size)
            x = self.w_u_fc(p_l) * x

        x = F.relu(x)
        x = self.w_d_fc(x)

        return x


class BasePolicyLayer(torch.nn.Module):
    def __init__(self, input_size, n):
        """
        input_size: D
        """
        super().__init__()

        self.weight = Parameter(torch.empty((1, n, input_size, input_size)))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, p_l_softmax):
        """
        x: shape (BS, n, input_size)
        p_l_softmax: shape (BS, n, n)
        """
        # shape (BS, n, input_size)
        x = torch.matmul(
            x.unsqueeze(-2),  # shape (BS, n, 1, input_size)
            self.weight,  # shape (1, n, input_size, input_size)
        ).squeeze(-2)

        x = F.relu(x)

        # shape (BS, n, input_size)
        x = torch.matmul(p_l_softmax, x)

        return x


class PolicyRoutingModelDeep(PolicyRoutingModel):
    L = 4
    n = 4
