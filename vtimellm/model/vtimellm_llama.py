import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from .modeling_llama_VideoExpert import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vtimellm_arch import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM

import einops

class VTimeLLMConfig(LlamaConfig):
    model_type = "VTimeLLM"

class VTimeLLMLlamaModel(LlamaModel, VTimeLLMMetaModel):
    config_class = VTimeLLMConfig

    def __init__(self, config: LlamaConfig):
        super(VTimeLLMLlamaModel, self).__init__(config)

class VTimeLLMLlamaForCausalLM(LlamaForCausalLM, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMConfig

    def __init__(self, config, **kwargs):

        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.det_loss_weight = kwargs.pop("det_loss_weight", None)
        self.loc_token_idx = kwargs.pop("loc_token_idx")

        super(LlamaForCausalLM, self).__init__(config)
        self.model = VTimeLLMLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                timestamp: Optional[torch.Tensor] = None,
                span_label_nn: Optional[torch.Tensor] = None,
                timestamp_window: Optional[torch.Tensor] = None,):
        if labels is None:
            return self.forward_inference(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          position_ids=position_ids, 
                                          past_key_values=past_key_values,
                                          inputs_embeds=inputs_embeds, 
                                          labels=labels, 
                                          use_cache=use_cache, 
                                          output_attentions=output_attentions,
                                          output_hidden_states=output_hidden_states,
                                          images=images, 
                                          return_dict=return_dict,)
        
        return self.model_forward(input_ids=input_ids, 
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  images=images,
                                  return_dict=return_dict,
                                  timestamp=timestamp,
                                  span_label_nn=span_label_nn,
                                  timestamp_window=timestamp_window, )
    
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        timestamp: Optional[torch.Tensor] = None,
        span_label_nn: Optional[torch.Tensor] = None,
        timestamp_window: Optional[torch.Tensor] = None,
    ):
        if images.ndim == 5 and images.shape[2] == 3:  
            assert images.ndim == 5, "multiple videos per sample not supported yet" 
            
            b = len(images)
            tokens = einops.rearrange(images, 'b t c h w -> (b t) c h w')
            tokens = self.model.vision_tower(tokens)
            images = einops.rearrange(tokens, '(b t) s d -> b t s d', b=b)

        images, num_PaddingToken = self.model.Tree_based_patch_filter(images)

        loc_token_mask = input_ids[:, 1:] == self.loc_token_idx
        loc_token_mask = torch.cat(
			[
				loc_token_mask,
				torch.zeros((loc_token_mask.shape[0], 1)).bool().cuda(),
			],
			dim=1,
		)
        loc_token_mask = torch.cat(
			[torch.zeros((loc_token_mask.shape[0], num_PaddingToken-1)).bool().cuda(), loc_token_mask],
			dim=1,
		)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                frame_token_mask
            ) = self.prepare_inputs_labels_for_multimodal(  
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        
        output = super().forward(
            input_ids=input_ids,
            im_mask=frame_token_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        output_hidden_states = output.hidden_states
        output['loss'] = output['loss'] * self.ce_loss_weight

        hidden_states_det = []
        assert len(self.model.text_hidden_fcs_det) == 1
        hidden_states_det.append(self.model.text_hidden_fcs_det[0](output_hidden_states[-1]))
        last_hidden_state_det = torch.stack(hidden_states_det, dim=-1).sum(dim=-1)
        loc_token_counts = loc_token_mask.int().sum(-1)
        pred_loc_embed_det = last_hidden_state_det[loc_token_mask]
        frame_embed_det = last_hidden_state_det[frame_token_mask]

        loc_bs = len(loc_token_counts)
        frame_embed_det = frame_embed_det.view(loc_bs, 100, frame_embed_det.size(1) )
        cur_frame_embed = []

        for i, count in enumerate(loc_token_counts):
            sample = frame_embed_det[i]  
            replicated_sample = sample.repeat(count, 1, 1)
            cur_frame_embed.append(replicated_sample)
        frame_embed_det = torch.cat(cur_frame_embed, dim=0)

        if self.model.loc_interaction_type == 'simp_add':
            frame_embed_det = frame_embed_det + pred_loc_embed_det.unsqueeze(1)
        elif self.model.loc_interaction_type == 'intact_add':
            pred_loc_embed_det = pred_loc_embed_det.unsqueeze(1)
            frame_embed_det_hidden = torch.cat([frame_embed_det,pred_loc_embed_det.expand(-1, 100, -1)], dim=2)
            frame_embed_det = self.model.loc_interact[0](frame_embed_det_hidden)
        elif self.model.loc_interaction_type == 'simp-atten':
            pass
        elif self.model.loc_interaction_type == 'self-atten':
            pred_loc_embed_det = pred_loc_embed_det.unsqueeze(1)
            frame_det_mask = torch.ones(frame_embed_det.shape[0], frame_embed_det.shape[1], device=frame_embed_det.device, dtype=torch.bfloat16)
            pred_loc_mask = torch.ones(pred_loc_embed_det.shape[0], pred_loc_embed_det.shape[1], device=pred_loc_embed_det.device, dtype=torch.bfloat16)
            frame_embed_det = frame_embed_det + self.model.token_type_embeddings(torch.full_like(frame_det_mask.long(), 1))
            pred_loc_embed_det = pred_loc_embed_det + self.model.token_type_embeddings(torch.zeros_like(pred_loc_mask.long()))
            src = torch.cat([frame_embed_det, pred_loc_embed_det], dim=1)
            mask = torch.cat([frame_det_mask, pred_loc_mask], dim=1).bool()
            pos_vid = self.model.vid_position_embedding(frame_embed_det, frame_det_mask).to(torch.bfloat16)  
            pos_txt = torch.zeros_like(pred_loc_embed_det, device=pred_loc_embed_det.device)
            pos = torch.cat([pos_vid, pos_txt], dim=1)
            memory = self.model.loc_interact(src, ~mask, pos)
            frame_embed_det = memory[:, :frame_embed_det.shape[1], :]
        elif self.model.loc_interaction_type == 'gating':
            pass

        outputs_class = self.model.class_embed_head(frame_embed_det).sigmoid()
        outputs_coord = self.model.span_embed_head(frame_embed_det)

        if self.model.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).to(outputs_coord.device)
            idx_mask = idx_mask.repeat(outputs_coord.shape[0], outputs_coord.shape[1], 1)
            outputs_coord = outputs_coord * idx_mask
        else:
            raise NotImplementedError
            
        if timestamp is not None:
            assert timestamp.shape[0] == outputs_coord.shape[0]
            detection_result_batch = {'pred_logits': outputs_class, 'pred_spans': outputs_coord,}
            target_det = {'timestamp': timestamp, 'timestamp_mask': torch.ones(timestamp.shape[:2], device=timestamp.device), 'span_label_nn': span_label_nn, 'timestamp_window': timestamp_window }   
            loss_dict = self.model.det_criterion(detection_result_batch, target_det)
            weight_dict = self.model.det_weight_dict
            det_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            detection_loss = det_losses*self.det_loss_weight
            output['loss'] = output['loss'] + detection_loss
        
        return output

        
    def forward_inference(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                frame_token_mask
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("VTimeLLM", VTimeLLMConfig)
AutoModelForCausalLM.register(VTimeLLMConfig, VTimeLLMLlamaForCausalLM)
