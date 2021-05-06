import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer 
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import sys, random, os
from dill._dill import check


DEBUG = 1
#Do not repeat this RUN_ID. Every time you run, use a new one. Also, use different ranges so that we don't overlap.
RUN_ID = "22_GPU_44_models_all_childes_words"

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

#When and if we're skipping multi-word wordbank words, IRT parameters
#won't have underestimated difficulty parameters. The normal
#distribution assumption is only on ability parameter.

def create_folder_if_not(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder: ", path)
    else:
        print(path, " folder already exists.")

#Generate predictions for wordbank words at their positions using multiple
def generate_predictions_multiple_models(checkpoint_name_list, max_words, scoring="top_k", min_prob = 0.1, cutoff = 0.5):
    if(not torch.cuda.is_available()):
        print("CUDA not available. Exiting...")
        sys.exit()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #ignore_word_list = ['babysitter', 'child\'s own name', 'pet\'s name']
    ignore_word_list = ['child\'s own name', 'pet\'s name']
    score_dictionaries = {}
    used_wordbank = []
    used_wordbank_indicator = 0
    for checkpoint_name in checkpoint_name_list:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_name)
        #This GPU code can be written better
        #https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
        model.to(device)
        #Need to move to GPU?
        print("\nModel initialized: " + checkpoint_name + "\n")
        sm = torch.nn.Softmax()
        
        score_dictionaries[checkpoint_name] = {}
        
        i = 0
        max_sentences_to_sample = 10#hyper parameter of sorts
        no_of_options = 20
        #If checkpoint_name has a /, we need to create a folder if it doesn't exist.
        if ('/' in checkpoint_name):
            folder_name, checkpoint_folderless_name = checkpoint_name.split("/")
            create_folder_if_not("../output/" + folder_name)
        predictions_file_path = "../output/"+checkpoint_name+"_RUN_" + RUN_ID+"_predictions.tsv"
        predictions_file = open(predictions_file_path, "w")
        print("\nOpened "+ predictions_file_path + " for writing.")
        for word in wordbank:
            if(DEBUG == 1):
                print("\n" + word)
            if(' ' in word):
                if(DEBUG == 1):
                    print("\n Skipped the word '"+ word + "' for being multi-worded. Not scored.")
                #score_file.write(word+"\t \n")
                continue#skip multi-word wordbank words
            if(word in ignore_word_list):
                if(DEBUG == 1):
                    print("\n Skipped the word '"+ word + "' for being in ignore list. Not scored.")
                #score_file.write(word+"\t \n")
                continue#skip words like babysitter (too specific)
            word_occurence_list = word_occurences_in_childes.get(word, [])
            if(len(word_occurence_list) > max_sentences_to_sample):
                sampled = random.sample(word_occurence_list, max_sentences_to_sample)
            else:
                sampled = word_occurence_list
            if(DEBUG == 1):
                print(sampled)
            len_sampled = len(sampled)
            if len(sampled) == 0:
                #score_file.write(word+"\t \n")
                if(DEBUG == 1):
                    print("\n Skipped the word '"+ word + "' for lack of sentences. Not scored.")
            else:
                if used_wordbank_indicator == 0:
                    used_wordbank.append(word)
                passed_among_sampled = 0
                for sample in sampled:
                    if(DEBUG == 1):
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
                    if(DEBUG == 1):
                        print(sentence_masked)
                    #sentence.replace(, tokenizer.mask_token)
                    input = tokenizer.encode(sentence_masked, return_tensors="pt")
                    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
                    input = input.to(device)
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
                    #This line can be an issue. Had to write it as some models were outputting words like 'i n s e c t'
                    top_k_words = [s.replace(" ", "") for s in top_k_words]
                    top_k_token_probs = torch.topk(mask_token_probs, no_of_options, dim=1).values[0].tolist()
                    if(DEBUG == 1):
                        print(top_k_words)
                        print(top_k_token_probs)
                    #https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.decode
                    #HAVE TO DEAL WITH extra spaces, CAPS and SYNONYMS etc
                    if(scoring == "top_k"):
                        if word in top_k_words or ' '+word in top_k_words:
                            passed_among_sampled = passed_among_sampled + 1
                    else:
                        #Deal with alternatives/synonyms
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
                    
                if(DEBUG == 1):
                    print("\n", word, ": ", passed_among_sampled, " / ", len_sampled)
                if(passed_among_sampled/len_sampled >= cutoff):
                    score_dictionaries[checkpoint_name][word] = "1"
                    #writing to file will be done after all the models have been used for generation
                    #score_file.write(word+"\t1\n")
                else:
                    score_dictionaries[checkpoint_name][word] = "0"
                    #score_file.write(word+"\t0\n")
            i = i + 1
            if i > max_words:
                break
        predictions_file.close()
        print("\nClosed "+ predictions_file_path + " for writing.")
        #append words into used_wordbank only once
        used_wordbank_indicator = 1
        #for printing out "\n" at the end of row
        last_checkpoint_name = checkpoint_name
        last_word = word
    
    model_file_path = "../output/RUN_" + RUN_ID + "_models.txt"
    model_file = open(model_file_path, "w")
    print("\nOpened "+ model_file_path + " for writing.")
    for checkpoint_name in checkpoint_name_list:
        model_file.write(str(checkpoint_name) + "\n")
    model_file.close()
    print("\nClosed "+ model_file_path + " for writing.")
    
    score_file_path = "../output/RUN_" + RUN_ID + "_scores.tsv"
    score_file = open(score_file_path, "w")
    print("\nOpened "+ score_file_path + " for writing.")
    #print the words from wordbank used for generation.
    #score_file.write(str(used_wordbank) + "\n")
    for word in used_wordbank:
        if(last_word == word):
            score_file.write(str(word) + "\n")
        else:
            score_file.write(str(word) + "\t")
    #First three runs had the for loops in the other order. Now, each row stands for a model.
    for checkpoint_name in checkpoint_name_list:
        for word in used_wordbank:
            score_file.write(score_dictionaries[checkpoint_name][word])
            #if(last_checkpoint_name == checkpoint_name):
            if(last_word == word):
                score_file.write("\n")
            else:
                score_file.write("\t")

    
    #To avoid the error in EM algo in mirt module:
    #Error: The following items have only one response category and cannot be estimated: airplane alligator ankle ant 
    checkpoint_name = "all-1"
    score_dictionaries[checkpoint_name] = {}
    for word in used_wordbank:
        score_dictionaries[checkpoint_name][word] = "1"
        score_file.write(score_dictionaries[checkpoint_name][word])
        #if(last_checkpoint_name == checkpoint_name):
        if(last_word == word):
            score_file.write("\n")
        else:
            score_file.write("\t")
    checkpoint_name = "all-0"
    score_dictionaries[checkpoint_name] = {}
    for word in used_wordbank:
        score_dictionaries[checkpoint_name][word] = "0"
        score_file.write(score_dictionaries[checkpoint_name][word])
        #if(last_checkpoint_name == checkpoint_name):
        if(last_word == word):
            score_file.write("\n")
        else:
            score_file.write("\t")
            
    score_file.close()
    print("\nClosed "+ score_file_path + " for writing.")
    
#More probably available at https://huggingface.co/transformers/pretrained_models.html
#MLMs availabel at https://huggingface.co/models?filter=masked-lm
checkpoint_name_1 = "nyu-mll/roberta-base-1B-1"#478MB
checkpoint_name_1_2 = "nyu-mll/roberta-base-1B-2"
checkpoint_name_1_3 = "nyu-mll/roberta-base-1B-3"

checkpoint_name_2 = "nyu-mll/roberta-base-100M-1"#478MB
checkpoint_name_2_2 = "nyu-mll/roberta-base-100M-2"
checkpoint_name_2_3 = "nyu-mll/roberta-base-100M-3"

checkpoint_name_3 = "nyu-mll/roberta-base-10M-1"#478MB
checkpoint_name_3_2 = "nyu-mll/roberta-base-10M-2"
checkpoint_name_3_3 = "nyu-mll/roberta-base-10M-3"

checkpoint_name_4 = "nyu-mll/roberta-med-small-1M-1"#174MB
checkpoint_name_4_2 = "nyu-mll/roberta-med-small-1M-2"
checkpoint_name_4_3 = "nyu-mll/roberta-med-small-1M-3"

#https://huggingface.co/bert-base-uncased/tree/main
checkpoint_name_5 = 'bert-base-uncased'#420MB
checkpoint_name_6 = 'distilbert-base-uncased'

checkpoint_name_7 = 'bert-base-multilingual-cased'
checkpoint_name_8 = 'albert-base-v2'#45MB
checkpoint_name_9 = 'bert-base-multilingual-uncased'#641MB
checkpoint_name_10 = 'distilbert-base-multilingual-cased'#517MB

#vinai models seem like character models?? Output: ['a n t', 'a n t s', 'a c t u a l' ....]
#These are all very small models of size ~1MB
checkpoint_name_11 = 'vinai/phobert-base'
checkpoint_name_11_2 = 'vinai/phobert-large'
checkpoint_name_12 = 'vinai/bertweet-base'
checkpoint_name_12_2 ='vinai/bertweet-covid19-base-cased'
checkpoint_name_12_3 ='vinai/bertweet-covid19-base-uncased'

#https://huggingface.co/transformers/converting_tensorflow_models.html#xlm
#checkpoint_name_13 = 'jplu/tf-xlm-roberta-base'
#checkpoint_name_13_2 = 'jplu/tf-xlm-roberta-large'
#https://huggingface.co/jplu/tf-xlm-r-ner-40-lang not a masked LM?

#Some weights of the model checkpoint at beomi/kcbert-base-dev were not used when initializing BertForMaskedLM
#BAD PERFORMANCE
#checkpoint_name_14 = 'beomi/kcbert-base'
#checkpoint_name_14_2 = 'beomi/kcbert-large'
#checkpoint_name_14_3 = 'beomi/kcbert-base-dev'
#checkpoint_name_14_4 = 'beomi/kcbert-large-dev'
#https://huggingface.co/beomi/KcELECTRA-base not a masked LM?

#Multi-lingual models listed at https://huggingface.co/transformers/multilingual.html
#These require language embeddings.
checkpoint_name_15 = 'xlm-mlm-ende-1024'#796MB
checkpoint_name_15_2 = 'xlm-mlm-enfr-1024'#792MB
checkpoint_name_15_3 = 'xlm-mlm-enro-1024'#795MB
#checkpoint_name_15_4 = 'xlm-mlm-xnli15-1024'#Gives weird predictions
#Gives weird predictions ['?~Aت', 'ums', 'pan', 'und', 'seien', '...', 'sei', '·', 'einen', 'pet', 
#'sum', 'ا?~Dصد', 'and', 'was', 'werde', '?~O?~B', 'cv', 'maan', '"', 'sowie']
#checkpoint_name_15_5 = 'xlm-mlm-tlm-xnli15-1024' #MLM + Translation?
#These don't require language embeddings
checkpoint_name_15_6 = 'xlm-mlm-17-1280'#1.1GB
checkpoint_name_15_7 = 'xlm-mlm-100-1280'#1.1GB
#MLM or CLM?
#xlm-clm-ende-1024
#2.5GB
#checkpoint_name_15_9 = 'xlm-mlm-en-2048'

#bert-base-multilingual-uncased
#bert-base-multilingual-cased
#xlm-roberta-base
#xlm-roberta-large

checkpoint_name_16 = 'bert-base-cased'#416MB
checkpoint_name_17 = 'bert-large-uncased'#1.3GB
checkpoint_name_17_2 = 'bert-large-cased'#1.2GB
checkpoint_name_18 = 'distilroberta-base'#316MB
#Has two other models. Check. Raised the error about config.json
#checkpoint_name_19 = 'TOD-BERT-JNT-V1'#420MB
checkpoint_name_20 = 'bert-large-uncased-whole-word-masking'#1.3GB
checkpoint_name_20_2 = 'bert-large-cased-whole-word-masking'#1.2GB
#800MB+
checkpoint_name_21 = 'albert-xxlarge-v2'#851MB
checkpoint_name_22 = 'sshleifer/tiny-distilroberta-base'#603KB
#https://github.com/google-research/albert
checkpoint_name_23 = 'albert-large-v1'#68MB
checkpoint_name_23_2 = 'albert-large-v2'#68MB
checkpoint_name_23_3 = 'albert-base-v1'#45MB
checkpoint_name_23_4 = 'albert-base-v2'#45MB
checkpoint_name_23_5 = 'albert-xlarge-v2'#225MB

checkpoint_name_24 = 'johngiorgi/declutr-sci-base'#421MB
checkpoint_name_25 = 'bert-base-cased-finetuned-mrpc'#413MB
#microsoft
#OSError: Can't load config for 'microsoft/mpnet-base'
#checkpoint_name_26 = 'microsoft/mpnet-base'
#checkpoint_name_26_??? = 'microsoft/graphcodebert-base'

#google
checkpoint_name_27 = 'google/electra-small-generator'#52MB

#OSError: Can't load weights for 'cardiffnlp/twitter-roberta-base'.
#checkpoint_name_28 = 'cardiffnlp/twitter-roberta-base'#478MB


#checkpoint_names = [checkpoint_name_1, checkpoint_name_2, checkpoint_name_3, checkpoint_name_4, checkpoint_name_5]
#checkpoint_names = [checkpoint_name_4, checkpoint_name_4_2, checkpoint_name_4_3, \
#                    checkpoint_name_3, checkpoint_name_3_2, checkpoint_name_3_3, \
#                    checkpoint_name_2, checkpoint_name_2_2, checkpoint_name_2_3, \
#                    checkpoint_name_1, checkpoint_name_1_2, checkpoint_name_1_3, \
#                    checkpoint_name_5, \
#                    checkpoint_name_6, \
#                    checkpoint_name_7, \
#                    checkpoint_name_8, \
#                    checkpoint_name_9, \
#                    checkpoint_name_10, \
#                    #checkpoint_name_11, checkpoint_name_11_2, checkpoint_name_12, checkpoint_name_12_2, checkpoint_name_12_3, \
#                    #checkpoint_name_13, checkpoint_name_13_2, \
#                    checkpoint_name_14, checkpoint_name_14_2, checkpoint_name_14_3, checkpoint_name_14_4, \
#                    checkpoint_name_15, checkpoint_name_15_2, checkpoint_name_15_3, checkpoint_name_15_4, checkpoint_name_15_5, \
#                    checkpoint_name_15_6, checkpoint_name_15_7 \
#                    ]

checkpoint_names = [checkpoint_name_4, checkpoint_name_4_2, checkpoint_name_4_3, \
                    checkpoint_name_3, checkpoint_name_3_2, checkpoint_name_3_3, \
                    checkpoint_name_2, checkpoint_name_2_2, checkpoint_name_2_3, \
                    checkpoint_name_1, checkpoint_name_1_2, checkpoint_name_1_3, \
                    checkpoint_name_5, \
                    checkpoint_name_6, \
                    checkpoint_name_7, \
                    checkpoint_name_8, \
                    checkpoint_name_9, \
                    checkpoint_name_10, \
                    checkpoint_name_11, checkpoint_name_11_2, checkpoint_name_12, checkpoint_name_12_2, checkpoint_name_12_3, \
                    checkpoint_name_15, checkpoint_name_15_2, checkpoint_name_15_3, \
                    checkpoint_name_15_6, checkpoint_name_15_7, \
                    checkpoint_name_16, \
                    checkpoint_name_17, checkpoint_name_17_2, \
                    checkpoint_name_18, \
                    checkpoint_name_20, checkpoint_name_20_2, \
                    checkpoint_name_21, \
                    checkpoint_name_22, \
                    checkpoint_name_23, checkpoint_name_23_2, checkpoint_name_23_3, checkpoint_name_23_4, checkpoint_name_23_5, \
                    checkpoint_name_24, \
                    checkpoint_name_25, \
                    checkpoint_name_27
                    ]

generate_predictions_multiple_models(checkpoint_names, 11000, scoring="top_k", min_prob = 0.1, cutoff = 0.5)
print("\nGenerated predictions (or attempted) for "+ "???" + 
      " words from wordbank using the models " + str(checkpoint_names) )