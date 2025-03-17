//rustup doc --book

use std::{collections::{HashMap, HashSet}, vec};

fn string_to_u32(input: Vec<&str>) -> Vec<Vec<u32>> {
    input.iter()
            .map(|word| word.chars()
                .map(|letter| letter as u32)
                .collect())
            .collect()
}

fn initialize_vocabulary(input: &Vec<Vec<u32>>) -> Vec<u32> {
    let mut unique_chars = HashSet::new();
    let mut result = Vec::new();

    for letters in input {
        for letter in letters.iter(){
            if unique_chars.insert(letter) { 
                result.push(letter.clone());
            }
        }
    }
    result
}

fn byte_pair_frequency(phrases: &Vec<Vec<u32>>) -> HashMap<Vec<u32>, i32> {
    let mut result: HashMap<Vec<u32>, i32> = HashMap::new();

    for word in phrases {
        for n in 0..(word.len().saturating_sub(1)) {
            let pair = vec![word[n], word[n + 1]]; 
            *result.entry(pair).or_insert(0) += 1; 
        }
    }

    result
}

fn byte_pair_biggest(vector: &HashMap<Vec<u32>, i32>) -> (u32, u32) {
    let mut result: (Vec<u32>, i32) = (Vec::new(), i32::MIN);

    for (key, &value) in vector.iter() {
        if value > result.1 {
            result = (key.clone(), value);
        }
    }

    (result.0[0], result.0[1])
} 

fn merge(ids: &mut Vec<Vec<u32>>, pair: &(u32, u32), idx: u32){
    for id_array in ids.iter_mut(){
        let mut i = 0;
        while i < (id_array.len() - 1) {
            if id_array[i] == pair.0 && id_array[i + 1] == pair.1{
                id_array.remove(i);
                id_array[i] = idx;
            }
            i+=1;
        }
    }
}

fn encoder(ids: &mut Vec<Vec<u32>>, vocab_size: u32) -> HashMap<(u32, u32), u32>{
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
    let num_merges = vocab_size - 256; 
    for i in 1..num_merges{
        let frequency = byte_pair_frequency(&ids);
        let pair = byte_pair_biggest(&frequency);
        let new_id = 256 + i;
        merge(ids, &pair, new_id);
        merges.insert(pair, new_id);
    }
    merges
}

fn recursive_decompose(merges: &HashMap<(u32, u32), u32>, list_result: &mut Vec<u32>, id: u32) {
    if id < 256 {
        list_result.push(id);
        return;
    }

    if let Some((&(a, b), _)) = merges.iter().find(|&(_, &v)| v == id) {
        recursive_decompose(merges, list_result, a);
        recursive_decompose(merges, list_result, b);
    }
}

fn decoder(ids: Vec<Vec<u32>>, merges: HashMap<(u32, u32), u32>) -> Vec<Vec<String>> {
    let mut result: Vec<Vec<String>> = Vec::new();

    for id_line in ids {
        let mut decoded_chars: String = String::new();

        for &id in id_line.iter() {
            let mut decoded_byte: Vec<u32> = Vec::new();
            recursive_decompose(&merges, &mut decoded_byte, id);

            for &byte in decoded_byte.iter() {
                if let Some(character) = char::from_u32(byte) {
                    decoded_chars.push(character);
                }
            }
        }

        result.push(vec![decoded_chars]);
    }
    result
}

fn main(){
    let training_data: Vec<&str> = vec![
        "Hello world!",
        "This is a test sentence.",
        "Byte Pair Encoding is useful for tokenization.",
        "Let's split words into subwords.",
        "Deep learning improves natural language processing.",
        "Rust is a systems programming language.",
        "Tokenization helps machines understand text.",
        "Pre-trained models improve accuracy.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Artificial intelligence is the future.",
        "Neural networks process data efficiently.",
        "Hyperparameters affect model performance.",
        "Text processing is an important NLP task.",
        "Data-driven approaches improve predictions.",
        "Linguistics plays a role in computational models.",
        "Large datasets require efficient algorithms.",
        "Training corpora must be diverse and representative.",
        "BPE helps handle out-of-vocabulary words.",
        "Sentence segmentation is the first step in NLP.",
        "Compression algorithms reduce data size.",
        "Whitespace and punctuation affect tokenization.",
        "Vocabulary size impacts model performance.",
        "Lowercasing and stemming are preprocessing techniques.",
        "Feature extraction is key to machine learning.",
        "Statistical methods improve text analysis.",
        "Parsing text requires syntactic understanding.",
        "Word embeddings map words to vector space.",
        "Sequence models capture language dependencies.",
        "Language models predict the next word.",
    ];


    let mut reference_phrases = string_to_u32(training_data);
    

    let test = encoder(&mut reference_phrases, 280);
    println!("");
    println!("{:?}", reference_phrases);
    let result = decoder(reference_phrases, test);
    println!("{:?}", result);
}
