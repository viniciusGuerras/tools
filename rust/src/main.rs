//rustup doc --book

use std::{collections::{HashMap, HashSet}, vec};

fn break_letters(input: Vec<&str>) -> Vec<Vec<char>> {
    let result: Vec<Vec<char>> = input
        .iter() 
        .map(|phrase| phrase.chars().collect()) 
        .collect(); 

    result
}

fn get_unique_symbols(input: &Vec<Vec<char>>) -> Vec<char> {
    let mut unique_chars = HashSet::new();
    let mut result = Vec::new();

    for phrase in input {
        for letter in phrase {
            if unique_chars.insert(*letter) { 
                result.push(*letter); 
            }
        }
    }
    result
}

fn pair_frequency(array: &Vec<Vec<char>>) -> HashMap<Vec<char>, i32> {
    let mut hash: HashMap<Vec<char>, i32> = HashMap::new();
    for word in array{
        for n in 0..(word.len()-1){
            let vector = vec![word[n], word[n+1]];
            let counter = hash.entry(vector).or_insert(0);
            *counter += 1;
        }
    }
    hash
}

fn biggest_pair(vector: &HashMap<Vec<char>, i32>) -> (Vec<char>, i32) {
    let mut biggest: (Vec<char>, i32) = (Vec::new(), i32::MIN);

    for (key, &value) in vector.iter() {
        if value > biggest.1 {
            biggest = (key.clone(), value);
        }
    }

    biggest
}

/* 
fn update_training_phrases() {

}

fn increase_vocabulary(input: Vec<Vec<char>>, vocabulary: Vec<char>, size: usize)  -> Vec<char>{
    let mut result: Vec<char> = Vec::new();
    while vocabulary.len() < size {
        let biggest_pair = pair_frequency(&input).keys();
        println!("{}", biggest_pair);
        // check the frequency needed
        // update the training phrases
        // repeat
    }

    result
}
*/

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

    let reference_phrases = break_letters(training_data);
    let _initial_vocabulary = get_unique_symbols(&reference_phrases);
    let result = pair_frequency(&reference_phrases);
    let finished = biggest_pair(&result);

    println!("{:?}", finished);

}
