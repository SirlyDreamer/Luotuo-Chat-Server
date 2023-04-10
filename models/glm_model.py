import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel, AutoTokenizer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_size = "left"

    if not multi_gpu:
        model = AutoModel.from_pretrained(
            base,
            trust_remote_code=True,
            device_map="auto"
        )

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        )

        model.load_state_dict(
            torch.load(finetuned),
            strict=False
        )
        return model, tokenizer
    else:
        model = AutoModel.from_pretrained(
            base,
            trust_remote_code=True,
            device_map={'':0}
        )

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        )

        model.load_state_dict(
            torch.load(finetuned),
            strict=False
        )
        model.half()
        return model, tokenizer
        