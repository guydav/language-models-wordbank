import datasets
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

#https://huggingface.co/nyu-mll/roberta-base-100M-1
#checkpoint_name = "nyu-mll/roberta-base-1B-1"
#checkpoint_name = "nyu-mll/roberta-base-100M-1"
checkpoint_name = "nyu-mll/roberta-base-10M-1"
#checkpoint_name = "nyu-mll/roberta-med-small-1M-1"
# checkpoint_name = 'bert-base-uncased'

tokenizer = RobertaTokenizer.from_pretrained(checkpoint_name)
model = RobertaForMaskedLM.from_pretrained(checkpoint_name)
#Need to move to GPU?
print("\nModel initialized: " + checkpoint_name + "\n")
sm = torch.nn.Softmax()

def check_Roberta():
    sequence = f"Distilled models are smaller than the models they mimic. Using them instead"\
        " of the large versions would help {tokenizer.mask_token} our carbon footprint."
    childes_sent_1 = f"and you can sit some people down here"
    childes_sent_1_masked = f"and you can sit some people {tokenizer.mask_token} here"
    #childes_sent_1 = f"do you do that a lot"
    #childes_sent_1_masked = f"do you do that {tokenizer.mask_token} {tokenizer.mask_token}"
    childes_sent_2 = f"want to put somebody to bed"
    childes_sent_2_masked = f"want to put somebody {tokenizer.mask_token} bed"
    #childes_sent_2_masked = f"want to put somebody to {tokenizer.mask_token}"
    
    options = 20
    input = tokenizer.encode(childes_sent_1_masked, return_tensors="pt")
    #print(input)
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    output = model(input, return_dict=True)
    #print(output)
    token_logits = output.logits
    #token_logits = model(input).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    #https://discuss.pytorch.org/t/how-to-extract-probabilities/2720/14
    mask_token_probs = sm(mask_token_logits)
    #print(torch.topk(mask_token_logits, options, dim=1))
    #top_5_token_probs, top_5_tokens = torch.topk(mask_token_logits, 5, dim=1)
    top_k_tokens = torch.topk(mask_token_logits, options, dim=1).indices[0].tolist()
    top_k_token_probs = torch.topk(mask_token_probs, options, dim=1).values[0].tolist()
    
    options_2 = 10
    input_2 = tokenizer.encode(childes_sent_2_masked, return_tensors="pt")
    mask_token_index_2 = torch.where(input_2 == tokenizer.mask_token_id)[1]
    output_2 = model(input_2, return_dict=True)
    token_logits_2 = output_2.logits
    mask_token_logits_2 = token_logits_2[0, mask_token_index_2, :]
    top_k_tokens_2 = torch.topk(mask_token_logits_2, options_2, dim=1).indices[0].tolist()
    mask_token_probs_2 = sm(mask_token_logits_2)
    top_k_token_probs_2 = torch.topk(mask_token_probs_2, options_2, dim=1).values[0].tolist()
    
    i = 0
    for token in top_k_tokens:
        print(childes_sent_1_masked.replace(tokenizer.mask_token, tokenizer.decode([token])), top_k_token_probs[i])
        i = i + 1
    
    i = 0
    for token in top_k_tokens_2:
        print(childes_sent_2_masked.replace(tokenizer.mask_token, tokenizer.decode([token])), top_k_token_probs_2[i])
        i = i + 1

check_Roberta()
