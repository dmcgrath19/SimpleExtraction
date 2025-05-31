import torch
import numpy as np
import logging
from datasets import load_dataset, get_dataset_config_names
logging.basicConfig(level='ERROR')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_pilecorpus(path,start_seed=42):
    """
    This is a way for parsing the Pile corpus.
    """
    print("Streaming the main pile corpus")
    
    all_texts = ""
    dataset = load_dataset(path, split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head= shuffled_dataset.skip(0)
    dataset_head = shuffled_dataset.take(1000000)

    for text in dataset_head:
        all_texts+= text['text']

    return all_texts

def parse_local(path):
    file_content=""
    chunk_size = 10 * 1024 * 1024  # 10 MB

    try:
        with open(path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break 
                
                file_content += chunk
        print("File read successfully.")
    except FileNotFoundError:
        print(f"The file at {path} was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file at {path}: {e}")
    
    return file_content


def parse_splitted(path, subset='default', max_examples=100000, start_seed=42):
    """
    This is for parsing the PileSplitted dataset.
    """
    print("Streaming the splitted pile")
    
    all_texts = ""
    examples_processed = 0

    print(f"Subset: {subset}")
    print(f"Path: {path}")

    # Load the dataset subset with streaming enabled
    dataset = load_dataset(path, subset, split="train", streaming=True)

    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head = shuffled_dataset.skip(0).take(max_examples)

    for text in dataset_head:
        # Replace newline characters with spaces
        all_texts += text['text'].replace('\n', ' ')

    print("Completed parsing")

    return all_texts


def parse_wmt_splitted(path, split_set='train', start_seed=33):
    """
    This is for getting data from KaiNylund/WMT-year-splits
    unseen data for the model serving as a base for perplexity
    """

    print("Streaming the WMT splitted dataset")

    all_texts = ""

    # Load the dataset split with streaming enabled
    dataset = load_dataset(path, split=split_set, streaming=True)
    
    shuffled_dataset = dataset.shuffle(seed=start_seed)
    dataset_head = shuffled_dataset.skip(0).take(100000)

    for text in dataset_head:
        # Replace newline characters with spaces
        all_texts += text['text'].replace('\n', ' ')

    print("Completed parsing")
    
    return all_texts


def calculate_perplexity(sentence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    Print the `n` best samples according to the given `metric`.
    Returns a string containing the information for each sample.
    """
    idxs = np.argsort(metric)[::-1][:n]
    output_string = ""

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}"
        else:
            sample_info = f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}"

        sample_text = samples[idx]
        output_string += sample_info + "\n" + sample_text + "\n\n"

    return output_string