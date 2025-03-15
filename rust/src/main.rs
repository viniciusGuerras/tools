//rustup doc --book

use std::{collections::{HashMap, HashSet}, vec};

fn string_to_chars(input: Vec<&str>) -> Vec<Vec<char>> {
    let result: Vec<Vec<char>> = input
        .iter() 
        .map(|phrase| phrase.chars().filter(|c| !c.is_whitespace()).collect()) 
        .collect(); 

    result
}

fn initialize_vocabulary(input: &Vec<Vec<char>>) -> Vec<char> {
    let mut unique_chars = HashSet::new();
    let mut result = Vec::new();

    for letters in input {
        for letter in letters {
            if unique_chars.insert(*letter) { 
                result.push(*letter); 
            }
        }
    }
    result
}

fn char_pair_frequency(phrases: &Vec<Vec<char>>) -> HashMap<Vec<char>, i32> {
    let mut result: HashMap<Vec<char>, i32> = HashMap::new();
    for word in phrases{
        for n in 0..(word.len()-1){
            let vector = vec![word[n], word[n+1]];
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

    result
}

/* 
fn update_training_phrases() {

}

fn increase_vocabulary(input: Vec<Vec<char>>, vocabulary: Vec<char>, size: usize)  -> Vec<char>{
    let mut result: Vec<char> = Vec::new();
    while vocabulary.len() < size {
        let char_pair_leader = pair_frequency(&input).keys();
        println!("{}", char_pair_leader);
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

    let reference_phrases = string_to_chars(training_data);
    let _initial_vocabulary = initialize_vocabulary(&reference_phrases);
    let result = char_pair_frequency(&reference_phrases);
    let finished = char_pair_leader(&result);

    println!("{:?}", finished);

}
