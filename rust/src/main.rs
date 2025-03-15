//rustup doc --book

use regex::Regex;
use std::{collections::{HashMap, HashSet}, vec};

fn string_to_chars(input: Vec<&str>) -> Vec<Vec<String>> {
    let re = Regex::new(r"[A-Za-z0-9]+|[[:punct:]]+").unwrap();
    
    let result: Vec<Vec<String>> = input
        .iter()
        .flat_map(|phrase| {
            re.find_iter(phrase)  
                .map(|mat| mat.as_str())  
                .map(|token| {
                    token.chars() 
                        .map(|c| c.to_string())  
                        .collect::<Vec<String>>()  
                })
        })
        .collect();  

    result
}

fn initialize_vocabulary(input: &Vec<Vec<String>>) -> Vec<String> {
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

fn char_pair_frequency(phrases: &Vec<Vec<String>>) -> HashMap<Vec<char>, i32> {
    let mut result: HashMap<Vec<char>, i32> = HashMap::new();

    for word in phrases {
        for n in 0..(word.len() - 1) {
            let vector = [word[n].chars().collect::<Vec<char>>(), word[n + 1].chars().collect::<Vec<char>>()]
                .concat();
            
            let counter = result.entry(vector).or_insert(0);
            *counter += 1;
        }
    }

    result
}

fn char_pair_leader(vector: &HashMap<Vec<char>, i32>) -> (Vec<char>, i32) {
    let mut result: (Vec<char>, i32) = (Vec::new(), i32::MIN);

    for (key, &value) in vector.iter() {
        if value > result.1 {
            result = (key.clone(), value);
        }
    }

    result } 

fn merge_chars(input: Vec<char>) -> String{
     input.iter().collect() 
}

fn update_vocabulary(training_chars: &Vec<Vec<String>>, vocabulary: &mut Vec<String>){
    let result = char_pair_frequency(&training_chars);
    let finished = char_pair_leader(&result);
    vocabulary.push(merge_chars(finished.0));
}

fn update_phrases(training_chars: &mut Vec<Vec<String>>, vocabulary: &mut Vec<String>) {
    if vocabulary.is_empty(){
        return;
    }

    let last_character = vocabulary.last().unwrap();
    for word in training_chars.iter_mut(){
        let mut s = 0;
        while s < (word.len() - 1){
            let combined = word[s].clone() + &word[s + 1];
            if combined == *last_character{
                word.remove(s);
                word[s] = last_character.clone();
            }
            else{
                s+=1;
            }
        }
    }
}

fn update(training_chars: &mut Vec<Vec<String>>, vocabulary: &mut Vec<String>, size: usize) {
    (0..size).for_each(|_| {
        update_vocabulary(training_chars, vocabulary);
        update_phrases(training_chars, vocabulary);
    });
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


    //i need to remove the whitespaces.

    let mut reference_phrases = string_to_chars(training_data);
}
