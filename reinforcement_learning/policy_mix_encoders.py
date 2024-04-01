import argparse
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
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


class MixtureEncodersModel(pufferlib.models.Policy):
    """Multi-Task Reinforcement Learning with Context-based Representations"""
    def __init__(self, env, input_size=256, hidden_size=256, task_size=4096):
        super().__init__(env)

        self.k = 4

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.tile_encoders = ModuleList(
            [TileEncoderV2(input_size) for _ in range(self.k)]
        )
        self.player_encoders = ModuleList(
            [PlayerEncoder(input_size, hidden_size) for _ in range(self.k)]
        )
        self.item_encoders = ModuleList(
            [ItemEncoder(input_size, hidden_size) for _ in range(self.k)]
        )
        self.inventory_encoders = ModuleList(
            [InventoryEncoder(input_size, hidden_size) for _ in range(self.k)]
        )
        self.market_encoders = ModuleList(
            [MarketEncoder(input_size, hidden_size) for _ in range(self.k)]
        )
        self.proj_enc_fcs = ModuleList(
            [torch.nn.Linear(4 * input_size, input_size) for _ in range(self.k)]
        )

        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
        self.proj_z_fc = torch.nn.Linear(input_size, input_size)
        self.proj_fc = torch.nn.Linear(2 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_observations):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            flat_observations,
            self.flat_observation_space,
            self.flat_observation_structure,
        )

        encs = []
        player_embeddings_list = []
        item_embeddings_list = []
        market_embeddings_list = []

        tile = env_outputs["Tile"]
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7

        for i in range(self.k):
            tile_encoder = self.tile_encoders[i]
            player_encoder = self.player_encoders[i]
            item_encoder = self.item_encoders[i]
            inventory_encoder = self.inventory_encoders[i]
            market_encoder = self.market_encoders[i]
            proj_enc_fc = self.proj_enc_fcs[i]

            tile = tile_encoder(env_outputs["Tile"])
            # env_outputs["Entity"] shape (BS, agents, n_entity_states)
            # player_embeddings shape (BS, agents, input_size)
            player_embeddings, my_agent = player_encoder(
                env_outputs["Entity"], env_outputs["AgentId"][:, 0]
            )

            item_embeddings = item_encoder(env_outputs["Inventory"])
            inventory = inventory_encoder(item_embeddings)

            market_embeddings = item_encoder(env_outputs["Market"])
            market = market_encoder(market_embeddings)

            enc = torch.cat([tile, my_agent, inventory, market], dim=-1)
            # shape (BS, input_size)
            enc = proj_enc_fc(enc)

            encs.append(enc.unsqueeze(-2))

            player_embeddings_list.append(player_embeddings.unsqueeze(-1))
            item_embeddings_list.append(item_embeddings.unsqueeze(-1))
            market_embeddings_list.append(market_embeddings.unsqueeze(-1))

        # shape (BS, k, input_size)
        encs = torch.cat(encs, dim=-2)
        # shape (BS, agents, input_size, k)
        player_embeddings_list = torch.cat(player_embeddings_list, dim=-1)
        item_embeddings_list = torch.cat(item_embeddings_list, dim=-1)
        market_embeddings_list = torch.cat(market_embeddings_list, dim=-1)

        task = self.task_encoder(env_outputs["Task"])

        with torch.no_grad():
            # shape (BS, k)
            alpha = torch.matmul(encs, task.unsqueeze(-1)).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1)

            # shape (BS, input_size)
            z_context = torch.matmul(
                encs.transpose(-1, -2), alpha.unsqueeze(-1)
            ).squeeze(-1)

            # shape (BS, agents, input_size)
            player_embeddings = torch.matmul(
                player_embeddings_list, alpha.unsqueeze(1).unsqueeze(-1)
            ).squeeze(-1)
            item_embeddings = torch.matmul(
                item_embeddings_list, alpha.unsqueeze(1).unsqueeze(-1)
            ).squeeze(-1)
            market_embeddings = torch.matmul(
                market_embeddings_list, alpha.unsqueeze(1).unsqueeze(-1)
            ).squeeze(-1)

        # shape (BS, input_size)
        z_enc = self.proj_z_fc(z_context)

        state_enc = torch.cat([task, z_enc], dim=-1)
        obs = self.proj_fc(state_enc)

        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value


class TileEncoderV2(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tile_offset = torch.tensor([i * 256 for i in range(3)])
        self.embedding = torch.nn.Embedding(3 * 256, 32)

        self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

    def forward(self, tile):
        tile = self.embedding(
            tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
        )

        agents, tiles, features, embed = tile.shape
        tile = (
            tile.view(agents, tiles, features * embed)
            .transpose(1, 2)
            .view(agents, features * embed, 15, 15)
        )

        tile = F.relu(self.tile_conv_1(tile))
        tile = F.relu(self.tile_conv_2(tile))
        tile = tile.contiguous().view(agents, -1)
        tile = F.relu(self.tile_fc(tile))

        return tile
