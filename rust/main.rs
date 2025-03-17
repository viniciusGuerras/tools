//rustup doc --book

use std::{collections::{HashMap, HashSet}, vec};
use lazy_static::lazy_static;
use fancy_regex::Regex;


fn string_to_u32(input: &str) -> Vec<u32> {
    let mut output = Vec::with_capacity(input.len());
    output.extend(input.chars().map(|letter| letter as u32));
    output
}


fn tokenize(input: &str) -> Vec<String> {
    lazy_static! {
        static ref RE: Regex = Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d|(?:\s{2,}(?=\S))| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+|\$+"
        ).unwrap();
    }

    let mut tokens = Vec::new();
    for mat in RE.find_iter(input) {
        let m = mat.unwrap();
        tokens.push(m.as_str().to_string());
    }
    tokens
}

fn adjust_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut adjusted = Vec::new();
    let mut iter = tokens.into_iter().peekable();

    while let Some(token) = iter.next() {
        if token.trim().is_empty() {
            if let Some(next_token) = iter.peek() {
                if !next_token.starts_with(' ') {
                    if token.len() > 1 {
                        let remaining_whitespace = token[..token.len()-1].to_string();
                        adjusted.push(remaining_whitespace);
                    }
                    let next = iter.next().unwrap();
                    adjusted.push(format!(" {}", next));
                    continue;
                }
            }
        }
        adjusted.push(token);
    }
    adjusted
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

fn byte_pair_frequency(phrase: &Vec<u32>) -> HashMap<(u32, u32), i32> {
    let mut result: HashMap<(u32, u32), i32> = HashMap::new();
    let len = phrase.len();
    if len < 2{
    }
    for n in 0..len-1{
        *result.entry((phrase[n], phrase[n+1])).or_insert(0) += 1; 
    }
    result
}

fn byte_pair_biggest(vector: &HashMap<(u32,u32), i32>) -> (u32, u32) {
    let mut result: ((u32, u32), i32) = ((u32::MIN, u32::MIN), i32::MIN);

    for (key, &value) in vector.iter() {
        if value > result.1 {
            result = (*key, value);
        }
    }

    result.0
} 

/*
fn merge(ids: &mut Vec<Vec<u32>>, pair: &(u32, u32), idx: u32){
    for id_array in ids.iter_mut(){
        let mut i = 0;
        while i < (id_array.len() - 1) {
            if id_array[i] == pair.0 && id_array[i + 1] == pair.1 {
                id_array.splice(i..=i + 1, [idx]);
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
*/

fn main(){
    let training_data: Vec<&str> = vec![
        "Hello        world!",
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


    for phrase in training_data{
        let tokenized_phrase = tokenize(phrase);
        let the_tokens = adjust_tokens(tokenized_phrase);
        //println!("{:?}", tokenized_phrase);
        let text = string_to_u32(tok);
        //println!("{:?}", text);
        let next = byte_pair_frequency(&text);
        //println!("{:?}", next);
        let biggest = byte_pair_biggest(&next);
        println!("{:?}", biggest);
    }
    /*     println!(" ");
    //re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L\p{N}]+|\s+(?!\S) |\$+""");
    let mut reference_phrases = string_to_u32(training_data);
    

    println!("{:?}", reference_phrases);
    println!("");

    let test = encoder(&mut reference_phrases, 280);

    println!("");
    println!("{:?}", reference_phrases);
    let result = decoder(reference_phrases, test);

    println!("");
    println!("{:?}", result);
*/

}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_to_u32() {
        let input = vec!["Hello", "world"];
        let result = string_to_u32(input);
        assert_eq!(result, vec![vec![72, 101, 108, 108, 111], vec![119, 111, 114, 108, 100]]);
    }
    
    #[test]
    fn test_initialize_vocabulary() {
        let input = vec![vec![72, 101, 108, 108, 111], vec![119, 111, 114, 108, 100]];
        let result = initialize_vocabulary(&input);
        assert!(result.contains(&72) && result.contains(&100));
    }
    
    #[test]
    fn test_byte_pair_frequency() {
        let input = vec![vec![72, 101, 108, 108, 111], vec![119, 111, 114, 108, 100]];
        let result = byte_pair_frequency(&input);
        assert_eq!(result.get(&(108, 108)), Some(&1));
    }
    
    #[test]
    fn test_byte_pair_biggest() {
        let mut frequencies = HashMap::new();
        frequencies.insert((72, 101), 5);
        frequencies.insert((108, 108), 10);
        let result = byte_pair_biggest(&frequencies);
        assert_eq!(result, (108, 108));
    }
    
    #[test]
    fn test_merge() {
        let mut input = vec![vec![72, 101, 108, 108, 111]];
        merge(&mut input, &(108, 108), 256);
        assert_eq!(input, vec![vec![72, 101, 256, 111]]);
    }
    
    #[test]
    fn test_encoder() {
        let mut input = vec![vec![72, 101, 108, 108, 111]];
        let result = encoder(&mut input, 280);
        assert!(!result.is_empty());
    }
    
    #[test]
    fn test_recursive_decompose() {
        let mut merges = HashMap::new();
        merges.insert((72, 101), 256);
        let mut result = vec![];
        recursive_decompose(&merges, &mut result, 256);
        assert_eq!(result, vec![72, 101]);
    }
    
    #[test]
    fn test_decoder() {
        let input = vec![vec![72, 101, 256, 111]];
        let mut merges = HashMap::new();
        merges.insert((108, 108), 256);
        let result = decoder(input, merges);
        assert_eq!(result, vec![vec!["HeĀo".to_string()]]);
    }
}
