use rusty_neat::Genome;
use rusty_neat::PhenotypeBehavior;

// Verify default PhenotypeBehavior contracts:
// - newly created instance has empty `m_data`
// - `successful()` returns true by default
// - `acquire()` returns false with default implementation
// Also validate equality via `m_data` and that `distance_to()` returns 0.0
// for identical behavior matrices.
#[test]
fn phenotype_behavior_defaults_and_equality() {
    let mut pb = PhenotypeBehavior::new();
    assert!(pb.m_data.is_empty());
    assert_eq!(pb.successful(), true);
    assert_eq!(pb.acquire(&Genome::default()), false);

    // distance_to identical
    let mut a = PhenotypeBehavior::new();
    a.m_data.push(vec![1.0, 2.0]);
    let mut b = PhenotypeBehavior::new();
    b.m_data.push(vec![1.0, 2.0]);

    assert_eq!(a, b);
    assert_eq!(a.distance_to(&b), 0.0);
}
