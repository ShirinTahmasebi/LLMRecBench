def import_hf_model_and_tokenizer(model_name: str, access_token: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login
    import torch
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    login(token=access_token)
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=True,
        device_map="auto",
        token=access_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return language_model, tokenizer


def import_genrec_model_and_tokenizer(model_name: str, access_token: str, **kwargs):
    import torch
    from peft import PeftModel
    from huggingface_hub import login
    from transformers import LlamaForCausalLM, LlamaTokenizer

    login(token=access_token)
    
    language_model =  LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={'':0},
        token=access_token
    )

    language_model = PeftModel.from_pretrained(
        language_model,
        kwargs["lora_weights"],
        torch_dtype=torch.float16,
        device_map={'':0},
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return language_model, tokenizer
