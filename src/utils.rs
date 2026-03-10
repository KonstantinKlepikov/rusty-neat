//! Utility functions ported from MultiNEAT Utils.h / Utils.cpp
use std::f64;

/// Get max and min from a slice of f64. Returns (min,max).
pub fn get_max_min(vals: &[f64]) -> Option<(f64, f64)> {
    if vals.is_empty() {
        return None;
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    Some((min, max))
}

/// Integer to string
pub fn itos(i: i32) -> String {
    i.to_string()
}

/// Float to string
pub fn ftos(f: f64) -> String {
    f.to_string()
}

/// Clamp helpers (explicit for types used in the project)
pub fn clamp_f64(x: f64, min: f64, max: f64) -> f64 {
    if min > max {
        panic!("clamp: min > max")
    }
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

pub fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
    if min > max {
        panic!("clamp: min > max")
    }
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

pub fn clamp_i32(x: i32, min: i32, max: i32) -> i32 {
    if min > max {
        panic!("clamp: min > max")
    }
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

pub fn clamp_i64(x: i64, min: i64, max: i64) -> i64 {
    if min > max {
        panic!("clamp: min > max")
    }
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// Round to nearest integer (ties up)
pub fn rounded(val: f64) -> i32 {
    let integ = val as i32;
    let mant = val - integ as f64;
    if mant < 0.5 { integ } else { integ + 1 }
}

/// Round under offset
pub fn round_under_offset(val: f64, offset: f64) -> i32 {
    let integ = val as i32;
    let mant = val - integ as f64;
    if mant < offset { integ } else { integ + 1 }
}

/// Scale a value from [a_min..a_max] into [tr_min..tr_max]
pub fn scale_f64(a: f64, a_min: f64, a_max: f64, tr_min: f64, tr_max: f64) -> f64 {
    let a_r = a_max - a_min;
    let r = tr_max - tr_min;
    if a_r == 0.0 {
        return tr_min;
    }
    let rel = (a - a_min) / a_r;
    tr_min + r * rel
}

pub fn scale_f32(a: f32, a_min: f32, a_max: f32, tr_min: f32, tr_max: f32) -> f32 {
    let a_r = a_max - a_min;
    let r = tr_max - tr_min;
    if a_r == 0.0 {
        return tr_min;
    }
    let rel = (a - a_min) / a_r;
    tr_min + r * rel
}

/// Absolute value
pub fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}
