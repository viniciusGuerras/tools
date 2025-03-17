//rustup doc --book

use std::{collections::{HashMap, HashSet}, vec};
use lazy_static::lazy_static;
use fancy_regex::Regex;


fn string_to_u32(input: Vec<String>) -> Vec<Vec<u32>> {
    input.iter()
            .map(|word| word.chars()
                .map(|letter| letter as u32)
                .collect())
            .collect()
}


fn tokenize(input: &str) -> Vec<String> {
    lazy_static! {
        static ref RE: Regex = Regex::new(
            //need to take care of lowecase
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

fn byte_pair_frequency(phrases: &Vec<Vec<u32>>) -> HashMap<(u32, u32), i32> {
    let mut result: HashMap<(u32, u32), i32> = HashMap::new();

    for word in phrases {
        for n in 0..(word.len().saturating_sub(1)) {
            *result.entry((word[n], word[n + 1])).or_insert(0) += 1; 
        }
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

fn preprocessing(input: &str) -> Vec<Vec<u32>> {
    string_to_u32(adjust_tokens(tokenize(input)))
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
    let mut preproc = preprocessing("Language models predict the next word.",);
    let next = byte_pair_frequency(&preproc);
    let biggest = byte_pair_biggest(&next);
    let merges = encoder(&mut preproc, 280);
    println!("{:?}", merges);
    let decoded = decoder(preproc, merges);
    println!("{:?}", decoded);
}
