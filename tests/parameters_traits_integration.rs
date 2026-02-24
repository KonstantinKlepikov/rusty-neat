use rand::SeedableRng;
use rand::rngs::StdRng;
use rusty_neat::genes::{TraitDetail, TraitParameters, TraitValue};
use rusty_neat::parameters::Parameters;
mod common;
use common::make_simple_genome_with_traits;

// Integration tests for Parameters <-> Traits
//
// These tests validate that configuring `Parameters` trait maps (neuron/link)
// affects genome-level operations. They exercise two flows:
// 1) `Genome::mutate_neuron_traits` / `Genome::mutate_link_traits` when
//    detailed `TraitParameters` are provided (mutation_prob=1.0, replace)
//    — assertions check that values change and remain inside configured ranges.
// 2) `Genome::randomize_traits` which populates trait values with simple
//    fallbacks — assertions check that produced values lie in expected basic
//    ranges (ints in [-5,5], floats in [-1,1]).

#[test]
// Test that per-trait `TraitParameters` (set on `Parameters`) drive
// mutation behaviour at the genome level. Uses `mutation_prob = 1.0`
// and `mut_replace_prob = 1.0` to force replacement semantics so the
// test is deterministic with a fixed RNG seed.
fn genome_mutate_traits_respects_trait_parameters() {
    let mut params = Parameters::default();

    // define a neuron float trait 'n1' that will be replaced (mut_replace_prob=1.0)
    params.neuron_trait_parameters.insert(
        "n1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Float {
                min: 0.0,
                max: 1.0,
                mut_power: 0.1,
                mut_replace_prob: 1.0,
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    // define a link int trait 'l1' that will be replaced
    params.link_trait_parameters.insert(
        "l1".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 10,
                mut_power: 2,
                mut_replace_prob: 1.0,
            },
            dep_key: None,
            dep_values: Vec::new(),
        },
    );

    // Build a genome with one neuron and one link populated with traits
    let mut g = make_simple_genome_with_traits(
        &[("n1", TraitValue::Float(0.5))],
        &[("l1", TraitValue::Int(5))],
        &[],
    );

    let mut rng = StdRng::seed_from_u64(2024);

    // Mutations should occur (mutation_prob = 1.0) and results should be within the configured ranges
    let changed_neuron = g.mutate_neuron_traits(&params, &mut rng);
    assert!(changed_neuron, "neuron traits should have mutated");
    if let Some(nv) = g.neuron_gene_at(0).traits.get("n1") {
        match nv {
            TraitValue::Float(v) => {
                assert!(*v >= 0.0 && *v <= 1.0, "neuron trait out of bounds");
                assert!(
                    (*v - 0.5).abs() > std::f64::EPSILON,
                    "neuron trait should have been replaced"
                );
            }
            _ => panic!("n1 should be Float"),
        }
    } else {
        panic!("n1 trait missing on neuron");
    }

    let changed_link = g.mutate_link_traits(&params, &mut rng);
    assert!(changed_link, "link traits should have mutated");
    if let Some(lv) = g.link_gene_at(0).traits.get("l1") {
        match lv {
            TraitValue::Int(v) => {
                assert!(*v >= 0 && *v <= 10, "link trait out of bounds");
                assert!(*v != 5, "link trait should have been replaced");
            }
            _ => panic!("l1 should be Int"),
        }
    } else {
        panic!("l1 trait missing on link");
    }
}

#[test]
// Test that `Genome::randomize_traits` assigns values within the simple
// fallback ranges implemented in `randomize_traits_map`.
// This verifies neuron, link and genome-level trait randomization.
fn genome_randomize_traits_sets_values_in_basic_ranges() {
    let mut g = make_simple_genome_with_traits(
        &[("ti", TraitValue::Int(0)), ("tf", TraitValue::Float(0.0))],
        &[("li", TraitValue::Int(0)), ("lf", TraitValue::Float(0.0))],
        &[("gi", TraitValue::Int(0)), ("gf", TraitValue::Float(0.0))],
    );

    let mut rng = StdRng::seed_from_u64(1337);
    g.randomize_traits(&mut rng);

    // check ranges: ints in [-5,5], floats in [-1,1]
    if let Some(TraitValue::Int(v)) = g.neuron_gene_at(0).traits.get("ti").cloned() {
        assert!(
            v >= -5 && v <= 5,
            "neuron int randomized out of expected range"
        );
    } else {
        panic!("ti missing or wrong type");
    }
    if let Some(TraitValue::Float(v)) = g.neuron_gene_at(0).traits.get("tf").cloned() {
        assert!(
            v >= -1.0 && v <= 1.0,
            "neuron float randomized out of expected range"
        );
    } else {
        panic!("tf missing or wrong type");
    }

    if let Some(TraitValue::Int(v)) = g.link_gene_at(0).traits.get("li") {
        assert!(
            *v >= -5 && *v <= 5,
            "link int randomized out of expected range"
        );
    } else {
        panic!("li missing or wrong type");
    }
    if let Some(TraitValue::Float(v)) = g.link_gene_at(0).traits.get("lf") {
        assert!(
            *v >= -1.0 && *v <= 1.0,
            "link float randomized out of expected range"
        );
    } else {
        panic!("lf missing or wrong type");
    }

    if let Some(gene) = g.genome_gene() {
        if let Some(TraitValue::Int(v)) = gene.traits.get("gi").cloned() {
            assert!(
                v >= -5 && v <= 5,
                "genome int randomized out of expected range"
            );
        } else {
            panic!("gi missing or wrong type");
        }
        if let Some(TraitValue::Float(v)) = gene.traits.get("gf").cloned() {
            assert!(
                v >= -1.0 && v <= 1.0,
                "genome float randomized out of expected range"
            );
        } else {
            panic!("gf missing or wrong type");
        }
    } else {
        panic!("genome_gene missing");
    }
}

// Integration test demonstrating dep_key / dep_values interaction at the
// Genome level: mutation should be skipped when the dependency is missing
// or has a non-matching value, and should occur when the dependency is satisfied.
#[test]
fn genome_mutate_traits_dep_key_dep_values_integration() {
    let mut params = Parameters::default();

    // Trait 't' depends on 'dep' == Int(1)
    params.neuron_trait_parameters.insert(
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

    params.link_trait_parameters.insert(
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

    let mut g = make_simple_genome_with_traits(
        &[("t", TraitValue::Int(0))],
        &[("t", TraitValue::Int(0))],
        &[],
    );

    let mut rng = StdRng::seed_from_u64(2026);

    // dep missing -> no mutation
    let changed_n = g.mutate_neuron_traits(&params, &mut rng);
    assert_eq!(
        changed_n, false,
        "neuron mutation should be skipped when dep missing"
    );

    let changed_l = g.mutate_link_traits(&params, &mut rng);
    assert_eq!(
        changed_l, false,
        "link mutation should be skipped when dep missing"
    );

    // set dep to non-matching value -> still skipped
    g.neuron_gene_at_mut(0)
        .traits
        .insert("dep".to_string(), TraitValue::Int(2));
    g.link_gene_at_mut(0)
        .traits
        .insert("dep".to_string(), TraitValue::Int(2));
    let mut rng2 = StdRng::seed_from_u64(2026);
    let changed_n2 = g.mutate_neuron_traits(&params, &mut rng2);
    assert_eq!(
        changed_n2, false,
        "neuron mutation should be skipped when dep value not matched"
    );

    let mut rng3 = StdRng::seed_from_u64(2026);
    let changed_l2 = g.mutate_link_traits(&params, &mut rng3);
    assert_eq!(
        changed_l2, false,
        "link mutation should be skipped when dep value not matched"
    );

    // set dep to matching value -> mutation should occur
    g.neuron_gene_at_mut(0)
        .traits
        .insert("dep".to_string(), TraitValue::Int(1));
    g.link_gene_at_mut(0)
        .traits
        .insert("dep".to_string(), TraitValue::Int(1));
    let mut rng4 = StdRng::seed_from_u64(2026);
    let changed_n3 = g.mutate_neuron_traits(&params, &mut rng4);
    assert_eq!(
        changed_n3, true,
        "neuron mutation should occur when dep satisfied"
    );

    let mut rng5 = StdRng::seed_from_u64(2026);
    let changed_l3 = g.mutate_link_traits(&params, &mut rng5);
    assert_eq!(
        changed_l3, true,
        "link mutation should occur when dep satisfied"
    );
}

// Similar dependency test for genome-level traits (`genome_gene`) to ensure
// `mutate_genome_traits` respects dep_key/dep_values semantics.
#[test]
fn genome_mutate_genome_traits_dep_key_dep_values_integration() {
    let mut params = Parameters::default();

    params.genome_trait_parameters.insert(
        "gt".to_string(),
        TraitParameters {
            importance_coeff: 1.0,
            mutation_prob: 1.0,
            detail: TraitDetail::Int {
                min: 0,
                max: 3,
                mut_power: 1,
                mut_replace_prob: 1.0,
            },
            dep_key: Some("gdep".to_string()),
            dep_values: vec![TraitValue::Int(1)],
        },
    );

    let mut g = make_simple_genome_with_traits(&[], &[], &[("gt", TraitValue::Int(0))]);

    let mut rng = StdRng::seed_from_u64(4242);

    // dep missing -> no mutation
    let changed0 = g.mutate_genome_traits(&params, &mut rng);
    assert_eq!(
        changed0, false,
        "genome mutation should be skipped when dep missing"
    );

    // set non-matching dep -> still skipped
    if let Some(gg) = g.genome_gene_mut() {
        gg.traits.insert("gdep".to_string(), TraitValue::Int(2));
    }
    let mut rng2 = StdRng::seed_from_u64(4242);
    let changed1 = g.mutate_genome_traits(&params, &mut rng2);
    assert_eq!(
        changed1, false,
        "genome mutation should be skipped when dep value not matched"
    );

    // set matching dep -> mutation should occur
    if let Some(gg) = g.genome_gene_mut() {
        gg.traits.insert("gdep".to_string(), TraitValue::Int(1));
    }
    let mut rng3 = StdRng::seed_from_u64(4242);
    let changed2 = g.mutate_genome_traits(&params, &mut rng3);
    assert_eq!(
        changed2, true,
        "genome mutation should occur when dep satisfied"
    );
}
