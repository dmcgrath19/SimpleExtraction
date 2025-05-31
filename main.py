import argparse
import numpy as np
import sys
import torch
import zlib
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from model_utils import calculate_perplexity, print_best, parse_pilecorpus, parse_splitted, parse_wmt_splitted, parse_local, device


def main(args):
    print(f"Using device: {device}")
    print("Loading dataset...")

    ds = None

    if args.is_local:
        ds = parse_local(path=args.corpus_path)
    elif args.is_splitted:
        ds= parse_splitted(path=args.corpus_path, subset=args.corpus_subset)
    elif args.is_wmt:
        ds= parse_wmt_splitted(path=args.corpus_path, split_set=args.split)
    else:
        ds= parse_pilecorpus(path = args.corpus_path, start_seed=args.random_seed)

    print("Length:", len(ds))
   
    seq_len = 256
    top_k = 40

    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model1)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model1 = AutoModelForCausalLM.from_pretrained(
        args.model1, 
        trust_remote_code=True, 
        return_dict=True
    ).to(device)

    model2 = AutoModelForCausalLM.from_pretrained(
        args.model2, 
        trust_remote_code=True, 
        return_dict=True
    ).to(device)


    model1.config.pad_token_id = model1.config.eos_token_id
    model2.eval()

    samples = []
    prompts_list = []
    prompt_suffix=[]


    scores = {"mem":[], "XL": [], "S": [], "Lower": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    
    with tqdm(total=args.N) as pbar:
        for _ in range(num_batches):
            #input_len 25 works pile
            input_len = args.input_len
            input_ids = []
            attention_mask = []
            
            while len(input_ids) < args.batch_size:
                # Sample random text from the Pile corpus
                r = np.random.randint(0, len(ds))
                
                
                chunk = " ".join(ds[r:r+10000].split(" ")[1:-1])
                tokenized_chunk = tokenizer(chunk, return_tensors="pt")
                token_ids= tokenized_chunk['input_ids'][0]

                prompt_ids= token_ids[:input_len]
                if prompt_ids.shape[0] < input_len:
                    continue  # Skip short prompts
                prompt= tokenizer.decode(prompt_ids, skip_special_tokens=True)
                # print("the lenght of tokenized prompt is:", len(inputs))
                # print(inputs)
                suffix_ids= token_ids[input_len:input_len+ 50 ]
                suffix= tokenizer.decode(suffix_ids, skip_special_tokens=True)


                input_ids.append(prompt_ids)
                attention_mask.append(torch.ones_like(prompt_ids))
                prompts_list.append(prompt)
                prompt_suffix.append(suffix)
            


            inputs = {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_mask)}
            
            
            print("Attention Mask shape:", inputs['attention_mask'].shape)
           
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                #top_k=top_k, 
                top_p=1.0
            )

            texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
            for text in texts:
                p1 = calculate_perplexity(text, model1, tokenizer)
                p2 = calculate_perplexity(text, model2, tokenizer)
                p_lower = calculate_perplexity(text.lower(), model1, tokenizer)
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
                
                samples.append(text)
                print(text)
                
                
                scores["XL"].append(p1.cpu())
                scores["S"].append(p2.cpu())
                scores["Lower"].append(p_lower.cpu())
                scores["zlib"].append(zlib_entropy)
                
            pbar.update(args.batch_size)
    # print("*"*100)
    # print("Prompt List has the following prompts:",len(prompts_list[0]))
    scores["XL"] = np.asarray(scores["XL"])
    scores["S"] = np.asarray(scores["S"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])

    model1_name = args.model1.replace("/", "_")
    model2_name = args.model2.replace("/", "_")

    
   
    # sample_test_full = [s[:200] for s in samples]
    sample_test = [s[input_len:input_len+50] for s in samples]
    
    # print("*"*100)
    comparison_result = [1 if sample == prompt else 0 for sample, prompt in zip(sample_test, prompt_suffix)]
    # print("The comparison list length is:", len(comparison_result))
    ones_count = sum(comparison_result)
    total_count = len(comparison_result)
    memorization = (ones_count / total_count) * 100
    
    
    print("Memorization is: "  , memorization)
    # prompts_list = [item for sublist in prompts_list for item in sublist]
    print("*"*100)
    print("Number of prompts are:", len(prompts_list))
    print("Prompts_list is: ", prompts_list)
    # # print("*"*100)
    # print("Number of Prompt Suffix are:", len(prompt_suffix))

    output_csv = f'output_scores_{model1_name}_{model2_name}_{args.name_tag}.csv'
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['sample', 'prompt', 'suffix', 'memorized', 'PPL_XL', 'PPL_S', 'PPL_Lower', 'Zlib']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\') # Must prepare for the worst
        writer.writeheader()
        for sample,prompt,suff, mem, xl, s, lower, zlib_ in zip(samples, prompts_list, prompt_suffix, comparison_result, scores["XL"], scores["S"], scores["Lower"], scores["zlib"]):
            writer.writerow({'sample': sample, 'prompt': prompt, 'suffix': suff, 'memorized': mem, 'PPL_XL': xl, 'PPL_S': s, 'PPL_Lower': lower, 'Zlib': zlib_})

    print("Results saved to ", output_csv)
    
    output_txt = f'output_results_{model1_name}_{model2_name}_{args.name_tag}.txt'
    with open(output_txt, 'w') as f:
        metric = -np.log(scores["XL"])
        f.write(f"======== top sample by XL perplexity: ========\n")
        f.write(print_best(metric, samples, "PPL", scores["XL"]))
        f.write("\n")

        metric = np.log(scores["S"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of S and XL perplexities: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"]))
        f.write("\n")

        metric = np.log(scores["Lower"]) / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of lower-case and normal-case perplexities: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"]))
        f.write("\n")

        metric = scores["zlib"] / np.log(scores["XL"])
        f.write(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========\n")
        f.write(print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"]))

        f.write(f"======== Percentage of memorization is: ========\n")
        f.write(f"========{memorization}")

    print("Top results written to ", output_txt)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--model1', type=str, required=True, help="Hugging Face model name for the first model")
    parser.add_argument('--model2', type=str, required=True, help="Hugging Face model name for the second model")
    parser.add_argument('--corpus-path', type=str, required=True, help="Path to the corpus dataset")
    parser.add_argument('--corpus-subset', type=str, required=False, help="data subset if using splitted data")
    parser.add_argument('--name-tag', type=str, required=False, help="Path to the corpus dataset")
    parser.add_argument('--random-seed', type=int, required=False, help="Random seed for dataset shuffling")
    parser.add_argument('--split', type=str, required=False, help="Split for dataset")
    parser.add_argument('--is-splitted', action='store_true', help="Determine type of dataset parsing")
    parser.add_argument('--is-wmt',  action='store_true', help="Determine type of dataset parsing")
    parser.add_argument('--is-local',action='store_true', help="local text file")
    parser.add_argument('--input-len', type=int, default=150, help="Default length for input prompt")

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)