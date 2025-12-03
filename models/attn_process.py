from diffusers.models.attention_processor import Attention


from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ConditionAttnProcessor2_0:
    def __init__(self, height=480, width=832, num_frames=7):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.height = height
        self.width = width
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        B, N, C = hidden_states.shape
        # for cross-attn hidden_states.shape -> torch.Size([B, 17550, 1536])  encoder_hidden_states.shape -> torch.Size([2, 226, 1536])
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query) # RMSNorm 中，除了最后一个维度，其他所有维度都被当作 batch 维处理。
        if attn.norm_k is not None:
            key = attn.norm_k(key) # RMSNorm 中，除了最后一个维度，其他所有维度都被当作 batch 维处理。

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            # breakpoint()
            # 这里要分别用两次位置编码
            if rotary_emb2 is not None:
                query[:, :, :query.shape[2] // 2] = apply_rotary_emb(query[:, :, :query.shape[2] // 2], rotary_emb)
                key[:, :, :key.shape[2] // 2] = apply_rotary_emb(key[:, :, :key.shape[2] // 2], rotary_emb)
                # ---------------------------------- Add ----------------------------------------------
                query[:, :, query.shape[2] // 2: ] = apply_rotary_emb(query[:, :, query.shape[2] // 2: ], rotary_emb2)
                key[:, :, key.shape[2] // 2: ] = apply_rotary_emb(key[:, :, key.shape[2] // 2: ], rotary_emb2)
            else:
                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)
            # ---------------------------------------------------------------------------------------------------

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states



class CausalWanAttnProcessor2_0:
    def __init__(self, height=480, width=832):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.height = height
        self.width = width


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_history_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        B, N, C = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query) # RMSNorm 中，除了最后一个维度，其他所有维度都被当作 batch 维处理。
        if attn.norm_k is not None:
            key = attn.norm_k(key) # RMSNorm 中，除了最后一个维度，其他所有维度都被当作 batch 维处理。

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
            # additional
            if rotary_history_emb is not None:
                key = apply_rotary_emb(key, rotary_history_emb)
            else:
                key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states




class FlowCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(
        self,
        rank=16,
        scale=1.0,
        block_index=0,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.block_index = block_index
        self.scale = scale
        # Plan A
        # self.context_k_down = nn.Linear(3072, rank)
        # self.context_k_up = nn.Linear(rank, 3072)
        # #
        # self.context_v_down = nn.Linear(3072, rank)
        # self.context_v_up = nn.Linear(rank, 3072)
        # init
        # nn.init.kaiming_uniform_(self.context_k_down.weight)
        # nn.init.kaiming_uniform_(self.context_v_down.weight)
        # nn.init.zeros_(self.context_k_up.weight)
        # nn.init.zeros_(self.context_v_up.weight)
        
        # Plan B
        self.context_k = nn.Linear(512, 3072)
        self.context_v = nn.Linear(512, 3072)
        # init
        nn.init.zeros_(self.context_k.weight)
        nn.init.zeros_(self.context_k.bias)
        nn.init.zeros_(self.context_v.weight)
        nn.init.zeros_(self.context_v.bias)


    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        flow_emb=None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # encoder_hidden_states.shape ---> torch.Size([2, 226, 3072]) 
        # hidden_states.shape ---> torch.Size([2, 17550, 3072])
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )


        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        # breakpoint()
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads


        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            # video 部分做旋转编码，文本部分不变
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # breakpoint()
        
        # 加在这里
        if flow_emb is not None:
            _, num_frame, _ = flow_emb.shape
            # query reshape
            query = query[:, :, text_seq_length:]
            query = rearrange(query, 'b t (g n) d -> (b g) t n d', g=num_frame)

            # context_key = self.context_k_up(F.silu(self.context_k_down(flow_emb), inplace=True))
            context_key = self.context_k(flow_emb)
            context_key = context_key.view(-1, context_key.shape[-1]).unsqueeze(1)

            # breakpoint()

            # context_value = self.context_v_up(F.silu(self.context_v_down(flow_emb), inplace=True))
            context_value = self.context_v(flow_emb)
            context_value = context_value.view(-1, context_value.shape[-1]).unsqueeze(1)
            # 
            
            context_key = context_key.view(batch_size * num_frame, -1, attn.heads, head_dim).transpose(1, 2)
            context_value = context_value.view(batch_size * num_frame, -1, attn.heads, head_dim).transpose(1, 2)

            context_hidden_states = F.scaled_dot_product_attention(
                query, context_key, context_value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            # reshape
            # breakpoint()
            context_hidden_states = rearrange(context_hidden_states, '(b g) t n d -> b t (g n) d', g=num_frame)
        else:
            context_hidden_states = 0
        
        hidden_states[:, :, text_seq_length:] = hidden_states[:, :, text_seq_length:] + self.scale * context_hidden_states
        

        # breakpoint()        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        return hidden_states, encoder_hidden_states