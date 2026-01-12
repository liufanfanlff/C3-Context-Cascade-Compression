
import sys
import logging
import pathlib
import torch
# torch.set_num_threads(1)
import transformers

# from C3.train.trainer import C3Trainer
# from C3.train.trainer_vit_llrd import C3Trainer
from C3.train.trainer_C3 import C3Trainer
from C3.model import *
from C3.data import make_supervised_data_module
from C3.utils.arguments import *
from C3.utils.constants import *
from C3.utils.utils import smart_tokenizer_and_embedding_resize
# from C3.model.vision_encoder.vary_b import build_vary_vit_b
import os
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM


os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['OSS_ENDPOINT'] = "http://oss.i.shaipower.com"

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length,)
    model = C3QwenForCausalLM.from_pretrained(model_args.model_name_or_path, use_safetensors=True)

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    model.to(dtype=dtype, device=training_args.device)
    data_args.image_token_len = model.get_model().config.latent_token_len
    data_args.use_im_start_end = model_args.use_im_start_end

    # model.requires_grad_(False)
    # for p in model.get_model().Q.parameters():
    #     p.requires_grad = True
    # for p in model.get_model().mm_projector.parameters():
    #     p.requires_grad = True
    # for p in model.get_model().llm1.parameters():
    #     p.requires_grad = True
    

    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    data_module = make_supervised_data_module(
        interleave=training_args.interleave, 
        tokenizer=tokenizer, 
        data_args=data_args
    )

    trainer = C3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
