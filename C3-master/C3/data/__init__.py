
import torch
import transformers
from dataclasses import dataclass, field

from C3.utils.constants import *


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        input_ids, labels, context_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "context_ids"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        context_ids = torch.nn.utils.rnn.pad_sequence(
            context_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            context_ids=context_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            context_attention_mask=context_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch
    

def make_supervised_data_module(interleave, tokenizer, data_args):

    if data_args.conversation_version == 'mpt':
        from C3.data.conversation_dataset_qwen import ConversationDataset
        dataset_cls = ConversationDataset
        
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        multimodal_cfg=dict(
            image_token_len=data_args.image_token_len,
            use_im_start_end=data_args.use_im_start_end,
        )
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)