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