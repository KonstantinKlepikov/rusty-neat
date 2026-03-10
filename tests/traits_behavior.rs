mod common;
use common::seeded_rng;
use rusty_neat::genes::{Gene, TraitDetail, TraitParameters, TraitValue};
use std::collections::HashMap as StdHashMap;

// Test deterministic roulette selection for set/string traits via init_traits
#[test]
fn set_trait_roulette_is_deterministic_with_fixed_seed() {
    let mut tp: StdHashMap<String, TraitParameters> = StdHashMap::new();
    tp.insert(
        "choice".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 0.0,
            detail: TraitDetail::Str {
                set: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                probs: vec![0.2, 0.3, 0.5],
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    let mut g1 = Gene::new();
    let mut g2 = Gene::new();

    let mut rng1 = seeded_rng(42);
    let mut rng2 = seeded_rng(42);

    g1.init_traits(&tp, &mut rng1);
    g2.init_traits(&tp, &mut rng2);

    let v1 = g1.get_trait("choice").cloned();
    let v2 = g2.get_trait("choice").cloned();

    assert!(v1.is_some());
    assert_eq!(
        v1, v2,
        "roulette selection should be deterministic given the same seed"
    );
}

// Test replace vs perturb semantics for Int and Float traits, including clamping
#[test]
fn int_and_float_replace_vs_perturb_and_clamping() {
    let mut tp: StdHashMap<String, TraitParameters> = StdHashMap::new();

    // Integer trait: allow range 0..10
    tp.insert(
        "ti".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 10,
                mut_power: 2,
                mut_replace_prob: 1.0, // force replace
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    // Float trait: range [0.0, 1.0]
    tp.insert(
        "tf".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Float {
                min: 0.0,
                max: 1.0,
                mut_power: 0.1,
                mut_replace_prob: 1.0, // force replace
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    let mut g = Gene::new();
    // set current values
    g.traits.insert("ti".to_string(), TraitValue::Int(5));
    g.traits.insert("tf".to_string(), TraitValue::Float(0.5));

    let mut rng = seeded_rng(7);

    // With mut_replace_prob = 1.0 values must change (replace) and stay within bounds
    assert!(g.mutate_traits(&tp, &mut rng));
    if let Some(TraitValue::Int(v)) = g.get_trait("ti") {
        assert!(*v >= 0 && *v <= 10, "int replaced out of bounds");
        assert!(
            *v != 5,
            "int should have been replaced when mut_replace_prob=1.0"
        );
    } else {
        panic!("ti should exist and be Int");
    }
    if let Some(TraitValue::Float(v)) = g.get_trait("tf") {
        assert!(*v >= 0.0 && *v <= 1.0, "float replaced out of bounds");
        assert!(
            (*v - 0.5).abs() > std::f64::EPSILON,
            "float should have been replaced"
        );
    } else {
        panic!("tf should exist and be Float");
    }

    // Now test perturb behavior: set mut_replace_prob = 0.0 and check delta bounded by mut_power
    let mut tp2: StdHashMap<String, TraitParameters> = StdHashMap::new();
    tp2.insert(
        "ti".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 10,
                mut_power: 2,
                mut_replace_prob: 0.0, // force perturb
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );
    tp2.insert(
        "tf".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Float {
                min: 0.0,
                max: 1.0,
                mut_power: 0.1,
                mut_replace_prob: 0.0, // force perturb
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    // set known current values
    g.traits.insert("ti".to_string(), TraitValue::Int(5));
    g.traits.insert("tf".to_string(), TraitValue::Float(0.5));

    let mut rng2 = seeded_rng(13);
    assert!(g.mutate_traits(&tp2, &mut rng2));

    // Int should have changed by at most mut_power (2)
    if let Some(TraitValue::Int(v)) = g.get_trait("ti") {
        let delta = (*v - 5).abs();
        assert!(delta <= 2, "int perturb exceeded mut_power");
        assert!(*v >= 0 && *v <= 10, "int perturb out of bounds");
    } else {
        panic!("ti should exist and be Int after perturb");
    }

    // Float should have changed by at most mut_power (0.1)
    if let Some(TraitValue::Float(v)) = g.get_trait("tf") {
        let delta = (*v - 0.5).abs();
        assert!(delta <= 0.100001, "float perturb exceeded mut_power");
        assert!(*v >= 0.0 && *v <= 1.0, "float perturb out of bounds");
    } else {
        panic!("tf should exist and be Float after perturb");
    }
}

// Test dependency key/values prevents mutation when not matched and allows when matched
#[test]
fn dep_key_and_dep_values_control_mutation() {
    let mut tp: StdHashMap<String, TraitParameters> = StdHashMap::new();

    // Trait t depends on dep having value Int(1)
    tp.insert(
        "t".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 3,
                mut_power: 1,
                mut_replace_prob: 1.0,
            },
            dep_key: Some("dep".to_string()),
            dep_values: vec![TraitValue::Int(1)],
        },
    );

    let mut g = Gene::new();
    // set t initial value
    g.traits.insert("t".to_string(), TraitValue::Int(0));

    // dep not present -> mutation should be skipped
    let mut rng = seeded_rng(23);
    let changed = g.mutate_traits(&tp, &mut rng);
    assert_eq!(
        changed, false,
        "mutation should be skipped when dependency missing"
    );

    // set dep to non-matching value -> still skipped
    g.traits.insert("dep".to_string(), TraitValue::Int(2));
    let mut rng2 = seeded_rng(23);
    let changed2 = g.mutate_traits(&tp, &mut rng2);
    assert_eq!(
        changed2, false,
        "mutation should be skipped when dependency value not in dep_values"
    );

    // set dep to matching value -> mutation should occur
    g.traits.insert("dep".to_string(), TraitValue::Int(1));
    let mut rng3 = seeded_rng(23);
    let changed3 = g.mutate_traits(&tp, &mut rng3);
    assert_eq!(
        changed3, true,
        "mutation should occur when dependency satisfied"
    );
}

// Test specific roulette outcomes for set/intset/floatset with a fixed seed
#[test]
fn set_intset_floatset_specific_choices() {
    // no-local-rng imports needed; use shared `seeded_rng` helper

    let mut tp: StdHashMap<String, TraitParameters> = StdHashMap::new();
    tp.insert(
        "s".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 0.0,
            detail: TraitDetail::Str {
                set: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                probs: vec![0.2, 0.3, 0.5],
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );
    tp.insert(
        "is".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 0.0,
            detail: TraitDetail::IntSet {
                set: vec![10, 20, 30],
                probs: vec![0.1, 0.2, 0.7],
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );
    tp.insert(
        "fs".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 0.0,
            detail: TraitDetail::FloatSet {
                set: vec![0.1, 0.5, 0.9],
                probs: vec![0.4, 0.4, 0.2],
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    let mut g = Gene::new();
    let mut rng = seeded_rng(2025);
    g.init_traits(&tp, &mut rng);

    // Expected results observed with StdRng seed 2025
    assert_eq!(g.get_trait("s"), Some(&TraitValue::Str("C".to_string())));
    assert_eq!(g.get_trait("is"), Some(&TraitValue::Int(30)));
    if let Some(TraitValue::Float(v)) = g.get_trait("fs") {
        assert!(
            (*v - 0.5).abs() < 1e-12,
            "expected 0.5 chosen from float set"
        );
    } else {
        panic!("fs should be present and Float");
    }
}
