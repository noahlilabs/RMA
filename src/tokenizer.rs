use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Split the input text on whitespace. For each token, produce a numeric ID.
/// In a real system, you'd have a real vocab. This is just a naive demonstration.
pub fn tokenize(text: &str) -> Vec<usize> {
    let mut ids = Vec::new();
    for token in text.split_whitespace() {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let h = hasher.finish();
        ids.push(h as usize);
    }
    ids
}