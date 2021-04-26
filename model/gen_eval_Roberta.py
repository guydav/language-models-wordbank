import datasets
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import sys, random


DEBUG = 1
RUN_ID = "01"

def check_Roberta():
    #https://huggingface.co/nyu-mll/roberta-base-100M-1
    #checkpoint_name = "nyu-mll/roberta-base-1B-1"
    #checkpoint_name = "nyu-mll/roberta-base-100M-1"
    checkpoint_name = "nyu-mll/roberta-base-10M-1"
    #checkpoint_name = "nyu-mll/roberta-med-small-1M-1"
    # checkpoint_name = 'bert-base-uncased'
    
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_name)
    model = RobertaForMaskedLM.from_pretrained(checkpoint_name)
    #Need to move to GPU?
    print("\nModel initialized for checking: " + checkpoint_name + "\n")
    sm = torch.nn.Softmax()

    sequence = f"Distilled models are smaller than the models they mimic. Using them instead"\
        " of the large versions would help {tokenizer.mask_token} our carbon footprint."
    childes_sent_1 = f"and you can sit some people down here"
    childes_sent_1_masked = f"and you can sit some people {tokenizer.mask_token} here"
    childes_sent_2 = f"want to put somebody to bed"
    childes_sent_2_masked = f"want to put somebody {tokenizer.mask_token} bed"
    #childes_sent_2_masked = f"want to put somebody to {tokenizer.mask_token}"
    
    options = 5
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
        
    print("\nCheck successfully generated words.")

check_Roberta()

#Read words from wordbank
def read_wordbank(wordbank):
    type_of_words = {}
    with open("../data/worbank_with_category.tsv", 'r') as wordbank_file:
        for line in wordbank_file:
            word, type = line.split("\t")
            wordbank.append(word)
            type_of_words[word] = type
    wordbank_file.close()


wordbank = []
read_wordbank(wordbank)
print("\nWordbank: ", wordbank)
#Some sample words:
#['value', 'a lot', 'all gone', 'alligator', 'applesauce', 'baa baa', 'babysitter', "babysitter's name",'belly button',
#'choo choo', 'church', 'go', 'go potty', 'gonna get you!', 'going to', 'grrr', 'gum', 'have to',
#'high chair', 'him', 'lawn mower', 'leg', 'let me', 'so big!', 'soap',
#'this', 'this little piggy', 'yourself', 'yucky', 'yum yum', 'zebra', 'zipper', 'zoo']

#Read childes sentences along with the matches from wordbank
def read_childes(childes_sentences, word_occurences_in_childes):
    i = 0
    with open("../data/childes_wordbank_cleaned_data.tsv", 'r') as childes_file:
        #Store in hashmaps? Or any other better methods? The size is small
        #and maynot require a database.
        #https://huggingface.co/docs/datasets/loading_datasets.html
        for line in childes_file:
            if(i > 0):
                gloss, matches, start, end, num_tokens, target_child_age, type = line.split("\t", 7)
                matches_as_list = matches.split(", ")
                starts_as_list = start.split(", ")
                ends_as_list = end.split(", ")
                
                childes_sentences.append(gloss)
                #Tokens, age, type data not being stored as of now.
                j = 0
                for match in matches_as_list:
                    start = starts_as_list[j]
                    end = ends_as_list[j]
                    word_occurence_list = word_occurences_in_childes.get(match, [])
                    word_occurence_list.append((i-1, start, end))#ERROR??
                    word_occurences_in_childes[match] = word_occurence_list
                    j = j + 1
                #print(match, start, end)
                if(i == 11):
                    print("\nPrinting a few sentences and word occurences from childes.")
                    print(childes_sentences)
                    print(word_occurences_in_childes)
                    #break
            i = i + 1
    childes_file.close()
    return i


childes_sentences = []
word_occurences_in_childes = {}
no_lines_read = read_childes(childes_sentences, word_occurences_in_childes)
print("\nFinished reading Childes data from " + str(no_lines_read) + " lines.")



#Generate predictions for wordbank words at their positions using an MLM
def generate_predictions(checkpoint_name, max_words, scoring="top_k", min_prob = 0.1, cutoff = 0.5):    
    #Need to convert to Auto classes. Also check with others if they're sticking to
    #Huggingface models only.
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_name)
    model = RobertaForMaskedLM.from_pretrained(checkpoint_name)
    #Need to move to GPU?
    print("\nModel initialized: " + checkpoint_name + "\n")
    sm = torch.nn.Softmax()
    
    i = 0
    max_sentences_to_sample = 10#hyper parameter of sorts
    no_of_options = 20
    #If checkpoint_name has a /, we need to create a folder if it doesn't exist.
    predictions_file_path = "../output/"+checkpoint_name+"_predictions.tsv"
    predictions_file = open(predictions_file_path, "w")
    print("\nOpened "+ predictions_file_path + " for writing.")
    score_file_path = "../output/"+checkpoint_name+"_scores.tsv"
    score_file = open(score_file_path, "w")
    print("\nOpened "+ score_file_path + " for writing.")
    ignore_word_list = ['babysitter', 'child\'s own name', 'pet\'s name']
    for word in wordbank:
        print("\n" + word)
        if(' ' in word):
            print("\n Skipped the word '"+ word + "' for being multi-worded. Not scored.")
            score_file.write(word+"\t \n")
            continue#skip multi-word wordbank words
        if(word in ignore_word_list):
            print("\n Skipped the word '"+ word + "' for being in ignore list. Not scored.")
            score_file.write(word+"\t \n")
            continue#skip words like babysitter (too specific)
        word_occurence_list = word_occurences_in_childes.get(word, [])
        if(len(word_occurence_list) > max_sentences_to_sample):
            sampled = random.sample(word_occurence_list, max_sentences_to_sample)
        else:
            sampled = word_occurence_list
        print(sampled)
        len_sampled = len(sampled)
        if len(sampled) == 0:
            score_file.write(word+"\t \n")
            print("\n Skipped the word '"+ word + "' for lack of sentences. Not scored.")
        else:
            passed_among_sampled = 0
            for sample in sampled:
                print("\n", sample)
                sentence_id, start, end = sample
                sentence = childes_sentences[sentence_id]
                if int(start) == 0:
                    sentence_start = ""
                else:
                    sentence_start = sentence[:int(start)-1]
                if int(end) == len(sentence):
                    sentence_end = ""
                else:
                    sentence_end = sentence[int(end):]
                sentence_masked = sentence_start + tokenizer.mask_token + sentence_end
                print(sentence_masked)
                #sentence.replace(, tokenizer.mask_token)
                input = tokenizer.encode(sentence_masked, return_tensors="pt")
                mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
                #Need to convert to batch mode for faster inference
                output = model(input, return_dict=True)
                token_logits = output.logits
                #Should we play with temperature?
                mask_token_logits = token_logits[0, mask_token_index, :]
                mask_token_probs = sm(mask_token_logits)
                #Have to deal with multiple instances of the same word. Especially with a space extra or a cap.
                #what <mask> did we see at the restaurant yesterday remember
                #[' else', ' exactly', ' ', ',', ' happened', ' people', ' what', 'What', ' time', ' we', ' little', ' you', '?', ' more', ' anyone', ' What', ' much', ' man', ' pictures', ' day']
                top_k_tokens = torch.topk(mask_token_logits, no_of_options, dim=1).indices[0].tolist()
                top_k_words = [tokenizer.decode(token) for token in top_k_tokens]
                top_k_token_probs = torch.topk(mask_token_probs, no_of_options, dim=1).values[0].tolist()
                print(top_k_words)
                print(top_k_token_probs)
                #https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.decode
                #HAVE TO DEAL WITH extra spaces, CAPS and SYNONYMS etc
                if(scoring == "top_k"):
                    if word in top_k_words or ' '+word in top_k_words:
                        passed_among_sampled = passed_among_sampled + 1
                else:
                    #DEal with spaces
                    word_token_index = torch.where(top_k_words == word or top_k_words == ' '+word)
                    if(scoring == "min_prob"):
                        if top_k_token_probs[word_token_index] >= min_prob:
                            passed_among_sampled = passed_among_sampled + 1
                    elif(scoring == "min_rel_prob"):
                        if top_k_token_probs[word_token_index] >= min_prob * max(top_k_token_probs):
                            passed_among_sampled = passed_among_sampled + 1
                        
                predictions_file.write(word+"\t"+str(sentence_id)+"\t"+childes_sentences[sentence_id]+
                                       "\t"+start+"\t"+end+"\t"+
                                       ', '.join(top_k_words)+"\t"+
                                       #', '.join([str(elem) for elem in top_k_tokens])+"\t"+
                                       ', '.join([str(elem) for elem in top_k_token_probs])+"\n")
                
            print("\n", word, ": ", passed_among_sampled, " / ", len_sampled)
            if(passed_among_sampled/len_sampled >= cutoff):
                score_file.write(word+"\t1\n")
            else:
                score_file.write(word+"\t0\n")
        i = i + 1
        if i > max_words:
            break
    predictions_file.close()
    print("\nClosed "+ predictions_file_path + " for writing.")
    score_file.close()
    print("\nClosed "+ score_file_path + " for writing.")
    return i


#https://huggingface.co/nyu-mll/roberta-base-100M-1
checkpoint_name = "nyu-mll/roberta-base-1B-1"
#checkpoint_name = "nyu-mll/roberta-base-100M-1"
#checkpoint_name = "nyu-mll/roberta-base-10M-1"
#checkpoint_name = "nyu-mll/roberta-med-small-1M-1"
# checkpoint_name = 'bert-base-uncased'

no_of_words_evaluated = generate_predictions(checkpoint_name, 10000, scoring="top_k", min_prob = 0.1, cutoff = 0.5)
print("\nGenerated predictions (or attempted) for "+str(no_of_words_evaluated)+ 
      " words from wordbank using the model " + checkpoint_name)

#When and if we're skipping multi-word wordbank words, IRT parameters
#won't have underestimated difficulty parameters. The normal
#distribution assumption is only on ability parameter.