//! Random number generation module

pub struct Random;

impl Random {
    pub fn rand_float() -> f64 {
        rand::random::<f64>()
    }
    pub fn rand_int(max: u64) -> u64 {
        rand::random::<u64>() % max
    }
}
