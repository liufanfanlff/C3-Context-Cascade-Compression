
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os

import dataclasses
from enum import auto, Enum
from typing import List, Tuple
from transformers import StoppingCriteria
from transformers import TextStreamer


DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

class C3Config(Qwen2Config):
    model_type = "C3"


class C3QwenModel(Qwen2Model):
    config_class = C3Config

    def __init__(self, config: Qwen2Config):
        super(C3QwenModel, self).__init__(config)

        self.Q = nn.Embedding(config.latent_token_len , config.contexts_compression_llm_hidden_size)
        self.mm_projector = nn.Linear(config.contexts_compression_llm_hidden_size, config.hidden_size)
        self.llm1 = None
        self.config.use_im_start_end = True
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        context_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if orig_embeds_params is not None:
            with torch.no_grad():
                self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        context_embeds = self.llm1.model.embed_tokens(context_ids)

        #######encoder#######

        if input_ids.shape[1] != 1 or self.training:
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            context_features = []

            for i in range(context_embeds.shape[0]):
                context_features.append([self.Q.weight])

          
            use_im_start_end = True
            new_context_embeds = []
            image_start_tokens_list = []
            for cur_context_ids, cur_context_embeds, cur_context_features in zip(context_ids, context_embeds, context_features):

                if use_im_start_end:
                    image_start_tokens = torch.where(cur_context_ids == im_start_token)[0]
                    image_start_tokens_list.append(image_start_tokens)

                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_context_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_context_embeds.device)
                        num_patches = per_cur_image_features.shape[0]
                        if cur_context_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        cur_context_embeds = torch.cat(
                            (
                                cur_context_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_context_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )
                    new_context_embeds.append(cur_context_embeds)
                else:
                    raise NotImplementedError

            image_start_tokens_list = torch.tensor(image_start_tokens_list)

            context_embeds = torch.stack(new_context_embeds, dim=0)
            llm1_hidden_states = self.llm1.forward(
                input_ids=None, attention_mask=context_attention_mask, past_key_values=None,
                inputs_embeds=context_embeds, use_cache=None, position_ids = None,
                output_attentions=output_attentions, output_hidden_states=True,
                return_dict=return_dict
            )['hidden_states'][-1]
            latent_contexts = []
            for i, llm1_hidden_state in enumerate(llm1_hidden_states): 
                image_start_token_pos = image_start_tokens_list[i]
                llm1_hidden_state = llm1_hidden_state[image_start_token_pos+1:image_start_token_pos + num_patches+1]
                latent_contexts.append(llm1_hidden_state)
           
            ########decoder########
            latent_features = []

            for latent_context in latent_contexts:
                latent_context = self.mm_projector(latent_context)
                latent_features.append([latent_context])


            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_latent_features in zip(input_ids, inputs_embeds, latent_features):

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_latent_features in zip(image_start_tokens, cur_latent_features):
                        per_cur_latent_features = per_cur_latent_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_latent_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_latent_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )
                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            
        return super(C3QwenModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )



class C3QwenForCausalLM(Qwen2ForCausalLM):
    config_class = C3Config
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = C3QwenModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        context_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs  = self.model(
            input_ids=input_ids,
            context_ids=context_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            context_attention_mask=context_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict  
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                #max_cache_length = past_key_values.get_max_length()
                max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                #"images": kwargs.get("images", None),
                "context_ids": kwargs.get("context_ids", None),
            }
        )
        return model_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
       
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        if os.path.exists(pretrained_model_name_or_path):
            llm1_path = os.path.join(pretrained_model_name_or_path, "llm1") 
            print(f"Loading llm1 from path: {llm1_path}")
            
            dtype = kwargs.get("torch_dtype", torch.float16) 
            device = kwargs.get("device_map", "auto") 
            
            llm1 = Qwen2ForCausalLM.from_pretrained(
                llm1_path,
                use_safetensors=kwargs.get("use_safetensors", True),
                torch_dtype=dtype,
                device_map=device, 
            )

        else:
            print(f"Loading llm1 from HF")
            dtype = kwargs.get("torch_dtype", torch.float16) 
            device = kwargs.get("device_map", "auto") 
            
            llm1 = Qwen2ForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="llm1",
                use_safetensors=kwargs.get("use_safetensors", True),
                torch_dtype=dtype,
                device_map=device, 
            )
        
        model.model.llm1 = llm1
        print("Successfully loaded and attached llm1.")
    
        
        return model

    def initialize_special_tokenizer(
        self, 
        tokenizer, 
        device="cuda"
    ):
        config = self.get_model().config
        self.resize_token_embeddings(len(tokenizer))
        config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        config.use_im_start_end = True

        if config.use_im_start_end:
            self.resize_token_embeddings(len(tokenizer))
            config.im_start_token, config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    def chat(self, tokenizer, context, prompt):

        self.initialize_special_tokenizer(tokenizer)

        qs = prompt
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*self.get_model().config.latent_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
        

        context = context + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*self.get_model().config.latent_token_len + DEFAULT_IM_END_TOKEN

        conv_mode = "mpt"
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        inputs_context = tokenizer([context])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        inputs_context_ids = torch.as_tensor(inputs_context.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.generate(
                input_ids,
                context_ids=inputs_context_ids,
                do_sample=False,
                num_beams = 1,
                no_repeat_ngram_size = 20,
                streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        return outputs



AutoConfig.register("C3", C3Config)
AutoModelForCausalLM.register(C3Config, C3QwenForCausalLM)


