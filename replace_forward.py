import paddlenlp
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPastAndCrossAttentions


def llamamodel_forward(
    self,
    input_ids=None,
    position_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    use_cache=None,
    past_key_values=None,
    output_attentions=False,
    output_hidden_states=None,
    return_dict=False,
    attn_mask_startend_row_indices=None,
    **kwargs,
):
    if self.sequence_parallel and use_cache:
        raise ValueError("We currently only support sequence parallel without cache.")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.layers))
    # NOTE: to make cache can be clear in-time
    past_key_values = list(past_key_values)

    seq_length_with_past = seq_length
    cache_length = 0
    if past_key_values[0] is not None:
        cache_length = past_key_values[0][0].shape[1]
        seq_length_with_past += cache_length
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self.sequence_parallel:
        # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
        bs, seq_len, hidden_size = inputs_embeds.shape
        inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
        # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
        inputs_embeds = ScatterOp.apply(inputs_embeds)

    if self.config.context_parallel_degree > 1 and (attention_mask is not None or self.config.alibi):
        raise NotImplementedError("Ring FlashAttention dosen't support attention_mask or alibi")

    # embed positions
    if self.config.use_flash_attention_for_generation:
        attention_mask = None
    elif attn_mask_startend_row_indices is None and attention_mask is None:
        # [bs, seq_len]
        attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
    if attn_mask_startend_row_indices is None and self.config.alibi:
        if self.config.use_long_sequence_strategies:
            alibi_layer = LongSequenceStrategies.build_long_sequence_strategy(
                self.config.long_sequence_strategy_type,
                self.config.long_sequence_strategy_name,
                **self.config.long_sequence_init_args,
            )
            alibi = alibi_layer(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
        else:
            alibi = build_alibi_tensor(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
        if self.config.tensor_parallel_degree > 1:
            block_size = self.config.num_attention_heads // self.config.tensor_parallel_degree
            alibi = alibi[
                :,
                self.config.tensor_parallel_rank
                * block_size : (self.config.tensor_parallel_rank + 1)
                * block_size,
            ]
            alibi = alibi.reshape([batch_size * block_size, 1, seq_length_with_past])
        else:
            alibi = alibi.reshape([batch_size * self.config.num_attention_heads, 1, seq_length_with_past])
    else:
        alibi = None

    if position_ids is None:
        position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

    use_casual_mask = get_use_casual_mask() and not self.config.alibi

    if self.config.use_flash_attention_for_generation or use_casual_mask:
        attention_mask = None
    elif attn_mask_startend_row_indices is None:
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
        )  # [bs, 1, seq_len, seq_len]

    is_casual = False

    if (
        attn_mask_startend_row_indices is None
        and self.config.use_flash_attention
        and get_env_device() not in ["gcu", "intel_hpu"]
    ):
        if self.config.use_flash_attention_for_generation or use_casual_mask:
            is_casual = True
        else:
            is_casual = is_casual_mask(attention_mask)
        if get_env_device() not in ["npu", "mlu"]:
            if is_casual and alibi is None:
                attention_mask = None
        else:
            attention_mask = None if attention_mask is None else attention_mask.astype("bool")
    hidden_states = inputs_embeds
    orig_seq_len = hidden_states.shape[1]
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    
    remove_token_states = None
    remove_token_indices_list = []
    keep_token_indices_list = list(range(orig_seq_len))
    for idx, (decoder_layer) in enumerate(self.layers):
        if idx % GROUP_SIZE == 0:
            group_attention_weights = all_self_attns
            all_self_attns = () if output_attentions else None # 用完就重置为空，不存储所有的attn_weights，减少显存占用
            token_reduction_mask = get_important_token_indices(group_attention_weights)

            # keep, remove = paddle.where(token_reduction_mask)[1].tolist(), paddle.where(~token_reduction_mask)[1].tolist()

            # remove_token_indices_list.extend(np.array(keep_token_indices_list)[remove].tolist())
            # keep_token_indices_list = np.delete(np.array(keep_token_indices_list), remove).tolist()

            # all_indices = paddle.concat([paddle.to_tensor(keep), paddle.to_tensor(remove)], axis=0)

            keep_indices = paddle.where(token_reduction_mask)[1].squeeze().tolist()
            remove_indices = paddle.where(~token_reduction_mask)[1].squeeze().tolist()

            remove_set = set(remove_indices)

            remove_element = [keep_token_indices_list[i] for i in remove_indices]
            remove_token_indices_list.extend(remove_element)

            keep_token_indices_list = [
                elem for idx, elem in enumerate(keep_token_indices_list) if idx not in remove_set
            ]

            all_indices = paddle.concat([
                paddle.to_tensor(keep_indices, dtype='int64'),
                paddle.to_tensor(remove_indices, dtype='int64')
            ], axis=0)

            combined_hidden_states = hidden_states.index_select(all_indices, axis=1)
            keep_part = combined_hidden_states[:,:len(keep_indices)]
            remove_part = combined_hidden_states[:, len(keep_indices):]

            hidden_states = keep_part

        if remove_token_states is None:
            remove_token_states = remove_part
        else:
            remove_token_states = paddle.concat((remove_token_states, remove_part), axis=1)


        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and idx not in self.no_recompute_layers
            and has_gradient
            and self.recompute_granularity == "full"
        ):
            layer_outputs = self.recompute_training_full(
                decoder_layer,
                hidden_states,
                position_ids,
                attention_mask,
                output_attentions,
                past_key_value,
                use_cache,
                alibi=alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids,
                attention_mask,
                output_attentions,
                past_key_value,
                use_cache,
                alibi=alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                npu_is_casual=is_casual,
            )

        # NOTE: clear outdate cache after it has been used for memory saving
        past_key_value = past_key_values[idx] = None
        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
    if len(remove_token_indices_list) != remove_token_states.shape[1]:
        raise ValueError("The length of removed_token_indices_list and remove_token_states must be the same.")

    if hidden_states.shape[1] + remove_token_states.shape[1] != orig_seq_len:
        raise ValueError("The sum of hidden_states and remove_token_states lengths must equal orig_seq_len.")

    if len(remove_token_indices_list) + len(keep_token_indices_list) != orig_seq_len:
        raise ValueError("The sum of removed_token_indices_list and keep_token_indices_list lengths must equal orig_seq_len.")

    hidden_states = hidden_states.squeeze(0)

    recover_hidden_states = paddle.zeros([orig_seq_len, hidden_states.shape[-1]]).to(hidden_states.place)
    recover_hidden_states.scatter_(paddle.to_tensor(remove_token_indices_list), remove_token_states.squeeze(0))
    recover_hidden_states.scatter_(paddle.to_tensor(keep_token_indices_list), hidden_states)
    
    hidden_states = recover_hidden_states
    if self.config.use_last_token_for_generation:
        hidden_states = paddle.unsqueeze(hidden_states[:, -1, :], 1)

    hidden_states = self.norm(hidden_states.unsqueeze(0))

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=None,
    )



def replace_llamamodel_forward():
    paddlenlp.transformers.llama.modeling.LlamaModel.forward = llamamodel_forward