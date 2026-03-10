use crate::genes::TraitParameters;
use std::collections::HashMap as StdHashMap;

/// Full `Parameters` ported from MultiNEAT C++ Defaults (see `cneat/MultiNEAT/src/Parameters.cpp`).
#[derive(Clone)]
pub struct Parameters {
    // Basic parameters
    /// Size of population
    pub population_size: u32,
    /// Controls the use of speciation. When off, the population will consist of only one species.
    pub speciation: bool,
    /// If true, enables dynamic compatibility thresholding (keep species count between Min/MaxSpecies)
    pub dynamic_compatibility: bool,
    /// Minimum number of species
    pub min_species: u32,
    /// Maximum number of species
    pub max_species: u32,
    /// Don't wipe the innovation database each generation?
    pub innovations_forever: bool,
    /// Allow clones (nearly identical genomes) to exist simultaneously
    /// This is useful for non-deterministic environments,
    /// as the same individual will get more than one chance to prove himself, also
    /// there will be more chances the same individual to mutate in different ways.
    /// The drawback is greatly increased time for reproduction. If you want to
    /// search quickly, yet less efficient, leave this to true.
    pub allow_clones: bool,
    /// Keep an archive of genomes and don't allow new genomes in the archive/population
    pub archive_enforcement: bool,
    /// Normalize genome size when calculating compatibility
    pub normalize_genome_size: bool,
    /// Optional custom topology/trait constraints: should return true if genome FAILS constraint
    /// Stored as Option/Arc to avoid requiring Clone/Debug on closure.
    pub custom_constraints:
        Option<std::sync::Arc<dyn Fn(&crate::genome::Genome) -> bool + Send + Sync>>,

    // GA parameters
    /// Age treshold, meaning if a species is below it, it is considered young
    pub young_age_treshold: u32,
    /// Fitness boost multiplier for young species (1.0 means no boost)
    /// Make sure it is >= 1.0 to avoid confusion
    pub young_age_fitness_boost: f64,
    /// Number of generations without improvement allowed for a species
    pub species_max_stagnation: u32,
    /// Minimum jump in fitness necessary to be considered improvement
    pub stagnation_delta: f64,
    /// Age threshold, meaning if a species if above it, it is considered old
    pub old_age_treshold: u32,
    /// Multiplier that penalizes old species.
    /// Make sure it is < 1.0 to avoid confusion.
    pub old_age_penalty: f64,
    /// Detect competetive coevolution stagnation
    /// This kills the worst species of age >N (each X generations)
    pub detect_competetive_coevolution_stagnation: bool,
    /// Kill worst species every N generations (used if competitive detection enabled)
    pub kill_worst_species_each: i32,
    /// Kill species older than this age
    pub kill_worst_age: i32,
    /// Percent of best individuals that are allowed to reproduce. 1.0 = 100%
    pub survival_rate: f64,
    /// Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
    pub crossover_rate: f64,
    /// If a baby results from sexual reproduction, this probability determines if mutation will
    /// be performed after crossover. 1.0 = 100% (always mutate after crossover)
    pub overall_mutation_rate: f64,
    /// Probability for inter-species mating
    pub interspecies_crossover_rate: f64,
    /// Probability for a baby gene to result from Multipoint Crossover when mating. 1.0 = 100%
    /// The default if the Average mating.
    pub multipoint_crossover_rate: f64,
    /// Probability that when doing multipoint crossover,
    /// the gene of the fitter parent will be prefered, instead of choosing one at random
    pub prefer_fitter_parent_rate: f64,
    /// Use roulette wheel selection
    pub roulette_wheel_selection: bool,
    /// Use tournament selection instead
    pub tournament_selection: bool,
    /// Tournament selection size
    pub tournament_size: u32,
    /// Fraction of individuals copied unchanged
    pub elite_fraction: f64,

    // Phased search parameters
    /// Using phased search
    pub phased_searching: bool,
    /// Using delta coding
    pub delta_coding: bool,
    /// MPC threshold to begin simplifying phase
    pub simplifying_phase_mpc_treshold: u32,
    /// How many generations of stagnation to enter simplifying phase
    pub simplifying_phase_stagnation_treshold: u32,
    /// How many generations of MPC stagnation needed to re-enable complexifying
    pub complexity_floor_generations: u32,

    // Novelty search parameters
    /// K constant
    pub novelty_search_k: u32,
    /// Sparseness threshold (add to archive if above)
    pub novelty_search_p_min: f64,
    /// Dynamic Pmin flag
    pub novelty_search_dynamic_pmin: bool,
    /// How many evaluations should pass without adding to the archive in order to lower Pmin
    pub novelty_search_no_archiving_stagnation_treshold: u32,
    /// How should it be multiplied (make it less than 1.0)
    pub novelty_search_pmin_lowering_multiplier: f64,
    /// Minimum Pmin allowed (Not lower than this value)
    pub novelty_search_pmin_min: f64,
    /// How many one-after-another additions to the archive should pass in order to raise Pmin
    pub novelty_search_quick_archiving_min_evaluations: u32,
    /// How should it be multiplied (make it more than 1.0)
    pub novelty_search_pmin_raising_multiplier: f64,
    /// Per how many evaluations to recompute the sparseness
    pub novelty_search_recompute_sparseness_each: u32,

    // Mutation parameters

    // Structural mutation parameters
    /// Probability for Add-Neuron mutation
    pub mutate_add_neuron_prob: f64,
    /// Allow splitting of recurrent links
    pub split_recurrent: bool,
    /// Allow splitting of looped recurrent links
    pub split_looped_recurrent: bool,
    /// Max tries to find neuron to split
    pub neuron_tries: i32,
    /// Probability for a baby to be mutated with the Add-Link mutation
    pub mutate_add_link_prob: f64,
    /// Probability new incoming link comes from bias neuron
    pub mutate_add_link_from_bias_prob: f64,
    /// Probability for Remove-Link mutation
    pub mutate_rem_link_prob: f64,
    /// Probability replace a simple neuron with a link
    pub mutate_rem_simple_neuron_prob: f64,
    /// Max tries to find two neurons for link operations
    pub link_tries: u32,
    /// Max links in genome (-1 unlimited)
    pub max_links: i32,
    /// Max neurons in genome (-1 unlimited)
    pub max_neurons: i32,
    /// Probability a link mutation will be recurrent
    pub recurrent_prob: f64,
    /// Probability a recurrent link mutation will be looped
    pub recurrent_loop_prob: f64,

    // Weight mutation parameters
    /// Probability for a baby's weights to be mutated
    pub mutate_weights_prob: f64,
    /// Probability of severe (shaking) weight mutation
    pub mutate_weights_severe_prob: f64,
    /// Probability for a particular gene to be mutated. 1.0 = 100%
    pub weight_mutation_rate: f64,
    /// Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
    pub weight_replacement_rate: f64,
    /// Max perturbation for weight mutation
    pub weight_mutation_max_power: f64,
    /// Max magnitude of replaced weight
    pub weight_replacement_max_power: f64,
    /// Maximum weight
    pub max_weight: f64,
    /// Minimum weight
    pub min_weight: f64,

    // Activation / time-constant / bias mutation parameters
    /// robability for a baby's A activation function parameters to be perturbed
    pub mutate_activation_a_prob: f64,
    /// Probability for a baby's B activation function parameters to be perturbed
    pub mutate_activation_b_prob: f64,
    /// Max magnitude for A parameter perturbation
    pub activation_a_mutation_max_power: f64,
    /// Max magnitude for B parameter perturbation
    pub activation_b_mutation_max_power: f64,
    /// Max magnitude for time constants perturbation
    pub time_constant_mutation_max_power: f64,
    /// Max magnitude for biases perturbation
    pub bias_mutation_max_power: f64,
    /// Activation parameter A min
    pub min_activation_a: f64,
    /// Activation parameter A max
    pub max_activation_a: f64,
    /// Activation parameter B min
    pub min_activation_b: f64,
    /// Activation parameter B max
    pub max_activation_b: f64,
    /// Probability for a baby that an activation function type will be changed for a single neuron
    /// considered a structural mutation because of the large impact on fitness
    pub mutate_neuron_activation_type_prob: f64,
    /// Probabilities for a particular activation function appearance
    pub activationfunction_signedsigmoid_prob: f64,
    pub activationfunction_unsignedsigmoid_prob: f64,
    pub activationfunction_tanh_prob: f64,
    pub activationfunction_tanhcubic_prob: f64,
    pub activationfunction_signedstep_prob: f64,
    pub activationfunction_unsignedstep_prob: f64,
    pub activationfunction_signedgauss_prob: f64,
    pub activationfunction_unsignedgauss_prob: f64,
    pub activationfunction_abs_prob: f64,
    pub activationfunction_signedsine_prob: f64,
    pub activationfunction_unsignedsine_prob: f64,
    pub activationfunction_linear_prob: f64,
    pub activationfunction_relu_prob: f64,
    pub activationfunction_softplus_prob: f64,

    /// FIXME: Legacy vector of activation function probabilities (used by some code paths)
    pub activation_function_probs: Vec<f64>,

    /// FIXME: Alias for legacy field name `timeconstant_mutation_max_power` used in C++/genome.rs
    pub timeconstant_mutation_max_power: f64,

    /// Probability for a baby's neuron time constant values to be mutated
    pub mutate_neuron_time_constants_prob: f64,
    /// Probability for a baby's neuron bias values to be mutated
    pub mutate_neuron_biases_prob: f64,

    /// Time constant range
    /// Min neuron time constanta
    pub min_neuron_time_constant: f64,
    /// Max neuron time constanta
    pub max_neuron_time_constant: f64,
    /// Min neuron bias
    pub min_neuron_bias: f64,
    /// Max neuron bias
    pub max_neuron_bias: f64,

    // Speciation / compatibility parameters
    /// Percent of disjoint genes importance
    pub disjoint_coeff: f64,
    /// Percent of excess genes importance
    pub excess_coeff: f64,
    /// Node-specific activation parameter A difference importance
    pub activation_a_diff_coeff: f64,
    /// Node-specific activation parameter B difference importance
    pub activation_b_diff_coeff: f64,
    /// Average weight difference importance
    pub weight_diff_coeff: f64,
    /// Average time constant difference importance
    pub time_constant_diff_coeff: f64,
    /// Average bias difference importance
    pub bias_diff_coeff: f64,
    /// Activation function type difference importance
    pub activation_function_diff_coeff: f64,
    /// Compatibility treshold
    pub compat_treshold: f64,
    /// Minumal value of the compatibility treshold
    pub min_compat_treshold: f64,
    /// Modifier per generation for keeping the species stable
    pub compat_treshold_modifier: f64,
    /// Per how many generations to change the treshold
    pub compat_tresh_change_interval_generations: u32,
    /// Per how many evaluations to change the treshold
    pub compat_tresh_change_interval_evaluations: u32,
    /// What is the minimal difference needed for not to be a clone
    pub min_delta_compat_equal_genomes: f64,
    /// How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
    pub constraint_trials: i32,

    // Genome properties / flags parameters
    /// When true, don't have a special bias neuron and treat all inputs equal
    pub dont_use_bias_neuron: bool,
    pub allow_loops: bool,

    // ES / HyperNEAT parameters
    pub division_threshold: f64,
    pub variance_threshold: f64,
    /// Used for Band prunning.
    pub band_threshold: f64,
    /// Max and Min Depths of the quadtree
    pub initial_depth: u32,
    pub max_depth: u32,
    /// How many hidden layers before connecting nodes to output. At 0 there is
    /// one hidden layer. At 1, there are two and so on.
    pub iteration_level: u32,
    /// The Bias value for the CPPN queries.
    pub cppn_bias: f64,
    /// Quadtree Dimensions
    /// The range of the tree. Typically set to 2,
    pub width: f64,
    pub height: f64,
    /// The (x, y) coordinates of the tree
    pub qtree_x: f64,
    pub qtree_y: f64,
    /// Use Link Expression output
    pub leo: bool,
    /// Threshold above which a connection is expressed
    pub leo_threshold: f64,
    /// Use geometric seeding. Currently only along the X axis. 1
    pub leo_seed: bool,
    pub geometry_seed: bool,

    // Universal traits
    pub neuron_trait_parameters: StdHashMap<String, TraitParameters>,
    pub link_trait_parameters: StdHashMap<String, TraitParameters>,
    pub genome_trait_parameters: StdHashMap<String, TraitParameters>,
    pub mutate_neuron_traits_prob: f64,
    pub mutate_link_traits_prob: f64,
    pub mutate_genome_traits_prob: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            population_size: 300,
            speciation: true,
            dynamic_compatibility: true,
            min_species: 5,
            max_species: 10,
            innovations_forever: true,
            allow_clones: true,
            archive_enforcement: false,
            normalize_genome_size: false,
            custom_constraints: None,

            young_age_treshold: 5,
            young_age_fitness_boost: 1.1,
            species_max_stagnation: 25000,
            stagnation_delta: 0.0,
            old_age_treshold: 30,
            old_age_penalty: 0.5,
            detect_competetive_coevolution_stagnation: false,
            kill_worst_species_each: 15,
            kill_worst_age: 10,
            survival_rate: 0.2,
            crossover_rate: 0.7,
            overall_mutation_rate: 0.75,
            interspecies_crossover_rate: 0.0001,
            multipoint_crossover_rate: 0.75,
            prefer_fitter_parent_rate: 0.25,
            roulette_wheel_selection: false,
            tournament_selection: true,
            tournament_size: 5,
            elite_fraction: 0.000001,
            constraint_trials: 2000000,

            phased_searching: false,
            delta_coding: false,
            simplifying_phase_mpc_treshold: 20,
            simplifying_phase_stagnation_treshold: 30,
            complexity_floor_generations: 40,

            novelty_search_k: 15,
            novelty_search_p_min: 0.5,
            novelty_search_dynamic_pmin: true,
            novelty_search_no_archiving_stagnation_treshold: 150,
            novelty_search_pmin_lowering_multiplier: 0.9,
            novelty_search_pmin_min: 0.05,
            novelty_search_quick_archiving_min_evaluations: 8,
            novelty_search_pmin_raising_multiplier: 1.1,
            novelty_search_recompute_sparseness_each: 25,

            mutate_add_neuron_prob: 0.01,
            split_recurrent: false,
            split_looped_recurrent: false,
            neuron_tries: 0,
            mutate_add_link_prob: 0.03,
            mutate_add_link_from_bias_prob: 0.0,
            mutate_rem_link_prob: 0.0,
            mutate_rem_simple_neuron_prob: 0.0,
            link_tries: 64,
            max_links: -1,
            max_neurons: -1,
            recurrent_prob: 0.25,
            recurrent_loop_prob: 0.25,

            mutate_weights_prob: 0.90,
            mutate_weights_severe_prob: 0.25,
            weight_mutation_rate: 1.0,
            weight_replacement_rate: 0.2,
            weight_mutation_max_power: 1.0,
            weight_replacement_max_power: 1.0,
            max_weight: 8.0,
            min_weight: -8.0,

            mutate_activation_a_prob: 0.0,
            mutate_activation_b_prob: 0.0,
            activation_a_mutation_max_power: 0.0,
            activation_b_mutation_max_power: 0.0,
            time_constant_mutation_max_power: 0.0,
            bias_mutation_max_power: 1.0,
            min_activation_a: 1.0,
            max_activation_a: 1.0,
            min_activation_b: 0.0,
            max_activation_b: 0.0,
            mutate_neuron_activation_type_prob: 0.0,
            activationfunction_signedsigmoid_prob: 0.0,
            activationfunction_unsignedsigmoid_prob: 1.0,
            activationfunction_tanh_prob: 0.0,
            activationfunction_tanhcubic_prob: 0.0,
            activationfunction_signedstep_prob: 0.0,
            activationfunction_unsignedstep_prob: 0.0,
            activationfunction_signedgauss_prob: 0.0,
            activationfunction_unsignedgauss_prob: 0.0,
            activationfunction_abs_prob: 0.0,
            activationfunction_signedsine_prob: 0.0,
            activationfunction_unsignedsine_prob: 0.0,
            activationfunction_linear_prob: 0.0,
            activationfunction_relu_prob: 0.0,
            activationfunction_softplus_prob: 0.0,

            // FIXME: Legacy vector: UnsignedSigmoid probability = 1.0
            // at index 1 (C++ default mapping)
            activation_function_probs: vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],

            // FIXME: Alias for legacy time-constant field
            timeconstant_mutation_max_power: 0.0,

            mutate_neuron_time_constants_prob: 0.0,
            mutate_neuron_biases_prob: 0.0,

            min_neuron_time_constant: 0.0,
            max_neuron_time_constant: 0.0,
            min_neuron_bias: 0.0,
            max_neuron_bias: 0.0,

            disjoint_coeff: 1.0,
            excess_coeff: 1.0,
            activation_a_diff_coeff: 0.0,
            activation_b_diff_coeff: 0.0,
            weight_diff_coeff: 0.5,
            time_constant_diff_coeff: 0.0,
            bias_diff_coeff: 0.0,
            activation_function_diff_coeff: 0.0,
            compat_treshold: 3.0,
            min_compat_treshold: 0.0,
            compat_treshold_modifier: 0.1,
            compat_tresh_change_interval_generations: 1,
            compat_tresh_change_interval_evaluations: 1,
            min_delta_compat_equal_genomes: 0.0000001,

            dont_use_bias_neuron: false,
            allow_loops: true,

            division_threshold: 0.03,
            variance_threshold: 0.03,
            band_threshold: 0.3,
            initial_depth: 3,
            max_depth: 3,
            iteration_level: 1,
            cppn_bias: 1.0,
            width: 2.0,
            height: 2.0,
            qtree_x: 0.0,
            qtree_y: 0.0,
            leo: false,
            leo_threshold: 0.1,
            leo_seed: false,
            geometry_seed: false,

            neuron_trait_parameters: StdHashMap::new(),
            link_trait_parameters: StdHashMap::new(),
            genome_trait_parameters: StdHashMap::new(),
            mutate_neuron_traits_prob: 0.0,
            mutate_link_traits_prob: 0.0,
            mutate_genome_traits_prob: 0.0,
        }
    }
}

impl Parameters {
    /// Reset parameters to defaults (uses `Default` implementation)
    pub fn reset(&mut self) {
        *self = Parameters::default();
    }

    /// Load parameters from a file path. Returns io::Error on failure.
    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        let s = std::fs::read_to_string(path)?;
        self.load_from_str(&s)
    }

    /// Load parameters from a string (token-based parser, compatible with C++ format)
    pub fn load_from_str(&mut self, data: &str) -> std::io::Result<()> {
        fn parse_bool(tf: &str) -> bool {
            tf == "true" || tf == "1" || tf == "1.0"
        }

        let tokens: Vec<&str> = data.split_whitespace().collect();
        // find start
        let mut i = 0usize;
        while i < tokens.len() && tokens[i] != "NEAT_ParametersStart" {
            i += 1;
        }
        if i >= tokens.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "NEAT_ParametersStart not found",
            ));
        }
        i += 1;

        while i < tokens.len() {
            let s = tokens[i];
            if s == "NEAT_ParametersEnd" {
                break;
            }
            i += 1;
            if i >= tokens.len() {
                break;
            }
            match s {
                "PopulationSize" => {
                    self.population_size = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "Speciation" => {
                    self.speciation = parse_bool(tokens[i]);
                    i += 1;
                }
                "DynamicCompatibility" => {
                    self.dynamic_compatibility = parse_bool(tokens[i]);
                    i += 1;
                }
                "MinSpecies" => {
                    self.min_species = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxSpecies" => {
                    self.max_species = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "InnovationsForever" => {
                    self.innovations_forever = parse_bool(tokens[i]);
                    i += 1;
                }
                "AllowClones" => {
                    self.allow_clones = parse_bool(tokens[i]);
                    i += 1;
                }
                "NormalizeGenomeSize" => {
                    self.normalize_genome_size = parse_bool(tokens[i]);
                    i += 1;
                }
                "ConstraintTrials" => {
                    self.constraint_trials = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "YoungAgeTreshold" => {
                    self.young_age_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "YoungAgeFitnessBoost" => {
                    self.young_age_fitness_boost = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "SpeciesMaxStagnation" => {
                    self.species_max_stagnation = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "StagnationDelta" => {
                    self.stagnation_delta = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "OldAgeTreshold" => {
                    self.old_age_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "OldAgePenalty" => {
                    self.old_age_penalty = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "DetectCompetetiveCoevolutionStagnation" => {
                    self.detect_competetive_coevolution_stagnation = parse_bool(tokens[i]);
                    i += 1;
                }
                "KillWorstSpeciesEach" => {
                    self.kill_worst_species_each = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "KillWorstAge" => {
                    self.kill_worst_age = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "SurvivalRate" => {
                    self.survival_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "CrossoverRate" => {
                    self.crossover_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "OverallMutationRate" => {
                    self.overall_mutation_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "InterspeciesCrossoverRate" => {
                    self.interspecies_crossover_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MultipointCrossoverRate" => {
                    self.multipoint_crossover_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "PreferFitterParentRate" => {
                    self.prefer_fitter_parent_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "RouletteWheelSelection" => {
                    self.roulette_wheel_selection = parse_bool(tokens[i]);
                    i += 1;
                }
                "TournamentSelection" => {
                    self.tournament_selection = parse_bool(tokens[i]);
                    i += 1;
                }
                "PhasedSearching" => {
                    self.phased_searching = parse_bool(tokens[i]);
                    i += 1;
                }
                "DeltaCoding" => {
                    self.delta_coding = parse_bool(tokens[i]);
                    i += 1;
                }
                "SimplifyingPhaseMPCTreshold" => {
                    self.simplifying_phase_mpc_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "SimplifyingPhaseStagnationTreshold" => {
                    self.simplifying_phase_stagnation_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ComplexityFloorGenerations" => {
                    self.complexity_floor_generations = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_K" => {
                    self.novelty_search_k = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_P_min" => {
                    self.novelty_search_p_min = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_Dynamic_Pmin" => {
                    self.novelty_search_dynamic_pmin = parse_bool(tokens[i]);
                    i += 1;
                }
                "NoveltySearch_No_Archiving_Stagnation_Treshold" => {
                    self.novelty_search_no_archiving_stagnation_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_Pmin_lowering_multiplier" => {
                    self.novelty_search_pmin_lowering_multiplier = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_Pmin_min" => {
                    self.novelty_search_pmin_min = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_Quick_Archiving_Min_Evaluations" => {
                    self.novelty_search_quick_archiving_min_evaluations = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_Pmin_raising_multiplier" => {
                    self.novelty_search_pmin_raising_multiplier = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "NoveltySearch_Recompute_Sparseness_Each" => {
                    self.novelty_search_recompute_sparseness_each = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateAddNeuronProb" => {
                    self.mutate_add_neuron_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "SplitRecurrent" => {
                    self.split_recurrent = parse_bool(tokens[i]);
                    i += 1;
                }
                "SplitLoopedRecurrent" => {
                    self.split_looped_recurrent = parse_bool(tokens[i]);
                    i += 1;
                }
                "MutateAddLinkProb" => {
                    self.mutate_add_link_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateAddLinkFromBiasProb" => {
                    self.mutate_add_link_from_bias_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateRemLinkProb" => {
                    self.mutate_rem_link_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateRemSimpleNeuronProb" => {
                    self.mutate_rem_simple_neuron_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "LinkTries" => {
                    self.link_tries = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxLinks" => {
                    self.max_links = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxNeurons" => {
                    self.max_neurons = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "RecurrentProb" => {
                    self.recurrent_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "RecurrentLoopProb" => {
                    self.recurrent_loop_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateWeightsProb" => {
                    self.mutate_weights_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateWeightsSevereProb" => {
                    self.mutate_weights_severe_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "WeightMutationRate" => {
                    self.weight_mutation_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "WeightMutationMaxPower" => {
                    self.weight_mutation_max_power = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "WeightReplacementRate" => {
                    self.weight_replacement_rate = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "WeightReplacementMaxPower" => {
                    self.weight_replacement_max_power = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxWeight" => {
                    self.max_weight = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinWeight" => {
                    self.min_weight = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateActivationAProb" => {
                    self.mutate_activation_a_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateActivationBProb" => {
                    self.mutate_activation_b_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationAMutationMaxPower" => {
                    self.activation_a_mutation_max_power = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationBMutationMaxPower" => {
                    self.activation_b_mutation_max_power = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinActivationA" => {
                    self.min_activation_a = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxActivationA" => {
                    self.max_activation_a = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinActivationB" => {
                    self.min_activation_b = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxActivationB" => {
                    self.max_activation_b = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "TimeConstantMutationMaxPower" => {
                    self.time_constant_mutation_max_power = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "BiasMutationMaxPower" => {
                    self.bias_mutation_max_power = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateNeuronTimeConstantsProb" => {
                    self.mutate_neuron_time_constants_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MutateNeuronBiasesProb" => {
                    self.mutate_neuron_biases_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinNeuronTimeConstant" => {
                    self.min_neuron_time_constant = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxNeuronTimeConstant" => {
                    self.max_neuron_time_constant = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinNeuronBias" => {
                    self.min_neuron_bias = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxNeuronBias" => {
                    self.max_neuron_bias = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_SignedSigmoid_Prob" => {
                    self.activationfunction_signedsigmoid_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_UnsignedSigmoid_Prob" => {
                    self.activationfunction_unsignedsigmoid_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_Tanh_Prob" => {
                    self.activationfunction_tanh_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_TanhCubic_Prob" => {
                    self.activationfunction_tanhcubic_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_SignedStep_Prob" => {
                    self.activationfunction_signedstep_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_UnsignedStep_Prob" => {
                    self.activationfunction_unsignedstep_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_SignedGauss_Prob" => {
                    self.activationfunction_signedgauss_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_UnsignedGauss_Prob" => {
                    self.activationfunction_unsignedgauss_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_Abs_Prob" => {
                    self.activationfunction_abs_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_SignedSine_Prob" => {
                    self.activationfunction_signedsine_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_UnsignedSine_Prob" => {
                    self.activationfunction_unsignedsine_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_Linear_Prob" => {
                    self.activationfunction_linear_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_Relu_Prob" => {
                    self.activationfunction_relu_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunction_Softplus_Prob" => {
                    self.activationfunction_softplus_prob = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "DontUseBiasNeuron" => {
                    self.dont_use_bias_neuron = parse_bool(tokens[i]);
                    i += 1;
                }
                "AllowLoops" => {
                    self.allow_loops = parse_bool(tokens[i]);
                    i += 1;
                }
                "ArchiveEnforcement" => {
                    self.archive_enforcement = parse_bool(tokens[i]);
                    i += 1;
                }
                "DisjointCoeff" => {
                    self.disjoint_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ExcessCoeff" => {
                    self.excess_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "WeightDiffCoeff" => {
                    self.weight_diff_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationADiffCoeff" => {
                    self.activation_a_diff_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationBDiffCoeff" => {
                    self.activation_b_diff_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "TimeConstantDiffCoeff" => {
                    self.time_constant_diff_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "BiasDiffCoeff" => {
                    self.bias_diff_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "ActivationFunctionDiffCoeff" => {
                    self.activation_function_diff_coeff = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "CompatTreshold" => {
                    self.compat_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinCompatTreshold" => {
                    self.min_compat_treshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "CompatTresholdModifier" => {
                    self.compat_treshold_modifier = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "CompatTreshChangeInterval_Generations" => {
                    self.compat_tresh_change_interval_generations = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "CompatTreshChangeInterval_Evaluations" => {
                    self.compat_tresh_change_interval_evaluations = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MinDeltaCompatEqualGenomes" => {
                    self.min_delta_compat_equal_genomes = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "DivisionThreshold" => {
                    self.division_threshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "VarianceThreshold" => {
                    self.variance_threshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "BandThreshold" => {
                    self.band_threshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "InitialDepth" => {
                    self.initial_depth = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "MaxDepth" => {
                    self.max_depth = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "IterationLevel" => {
                    self.iteration_level = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "TournamentSize" => {
                    self.tournament_size = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "CPPN_Bias" => {
                    self.cppn_bias = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "Width" => {
                    self.width = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "Height" => {
                    self.height = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "Qtree_X" => {
                    self.qtree_x = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "Qtree_Y" => {
                    self.qtree_y = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "Leo" => {
                    self.leo = parse_bool(tokens[i]);
                    i += 1;
                }
                "GeometrySeed" => {
                    self.geometry_seed = parse_bool(tokens[i]);
                    i += 1;
                }
                "LeoThreshold" => {
                    self.leo_threshold = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                "LeoSeed" => {
                    self.leo_seed = parse_bool(tokens[i]);
                    i += 1;
                }
                "Elitism" => {
                    self.elite_fraction = tokens[i]
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    i += 1;
                }
                _ => { /* unknown token: skip */ }
            }
        }

        Ok(())
    }

    /// Save parameters to a file path
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.save_to_writer(&mut f)
    }

    /// Save parameters to a writer in the C++ format
    pub fn save_to_writer<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        macro_rules! wln { ($fmt:expr $(, $arg:expr)*) => { write!(w, concat!($fmt, "\n") $(, $arg)*)?; } }

        wln!("NEAT_ParametersStart");
        wln!("PopulationSize {}", self.population_size);
        wln!(
            "Speciation {}",
            if self.speciation { "true" } else { "false" }
        );
        wln!(
            "DynamicCompatibility {}",
            if self.dynamic_compatibility {
                "true"
            } else {
                "false"
            }
        );
        wln!("MinSpecies {}", self.min_species);
        wln!("MaxSpecies {}", self.max_species);
        wln!(
            "InnovationsForever {}",
            if self.innovations_forever {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "AllowClones {}",
            if self.allow_clones { "true" } else { "false" }
        );
        wln!(
            "NormalizeGenomeSize {}",
            if self.normalize_genome_size {
                "true"
            } else {
                "false"
            }
        );
        wln!("ConstraintTrials {}", self.constraint_trials);
        wln!("YoungAgeTreshold {}", self.young_age_treshold);
        wln!("YoungAgeFitnessBoost {:.20}", self.young_age_fitness_boost);
        wln!("SpeciesMaxStagnation {}", self.species_max_stagnation);
        wln!("StagnationDelta {:.20}", self.stagnation_delta);
        wln!("OldAgeTreshold {}", self.old_age_treshold);
        wln!("OldAgePenalty {:.20}", self.old_age_penalty);
        wln!(
            "DetectCompetetiveCoevolutionStagnation {}",
            if self.detect_competetive_coevolution_stagnation {
                "true"
            } else {
                "false"
            }
        );
        wln!("KillWorstSpeciesEach {}", self.kill_worst_species_each);
        wln!("KillWorstAge {}", self.kill_worst_age);
        wln!("SurvivalRate {:.20}", self.survival_rate);
        wln!("CrossoverRate {:.20}", self.crossover_rate);
        wln!("OverallMutationRate {:.20}", self.overall_mutation_rate);
        wln!(
            "InterspeciesCrossoverRate {:.20}",
            self.interspecies_crossover_rate
        );
        wln!(
            "MultipointCrossoverRate {:.20}",
            self.multipoint_crossover_rate
        );
        wln!(
            "PreferFitterParentRate {:.20}",
            self.prefer_fitter_parent_rate
        );
        wln!(
            "RouletteWheelSelection {}",
            if self.roulette_wheel_selection {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "PhasedSearching {}",
            if self.phased_searching {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "DeltaCoding {}",
            if self.delta_coding { "true" } else { "false" }
        );
        wln!(
            "SimplifyingPhaseMPCTreshold {}",
            self.simplifying_phase_mpc_treshold
        );
        wln!(
            "SimplifyingPhaseStagnationTreshold {}",
            self.simplifying_phase_stagnation_treshold
        );
        wln!(
            "ComplexityFloorGenerations {}",
            self.complexity_floor_generations
        );
        wln!("NoveltySearch_K {}", self.novelty_search_k);
        wln!("NoveltySearch_P_min {:.20}", self.novelty_search_p_min);
        wln!(
            "NoveltySearch_Dynamic_Pmin {}",
            if self.novelty_search_dynamic_pmin {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "NoveltySearch_No_Archiving_Stagnation_Treshold {}",
            self.novelty_search_no_archiving_stagnation_treshold
        );
        wln!(
            "NoveltySearch_Pmin_lowering_multiplier {:.20}",
            self.novelty_search_pmin_lowering_multiplier
        );
        wln!(
            "NoveltySearch_Pmin_min {:.20}",
            self.novelty_search_pmin_min
        );
        wln!(
            "NoveltySearch_Quick_Archiving_Min_Evaluations {}",
            self.novelty_search_quick_archiving_min_evaluations
        );
        wln!(
            "NoveltySearch_Pmin_raising_multiplier {:.20}",
            self.novelty_search_pmin_raising_multiplier
        );
        wln!(
            "NoveltySearch_Recompute_Sparseness_Each {}",
            self.novelty_search_recompute_sparseness_each
        );
        wln!("MutateAddNeuronProb {:.20}", self.mutate_add_neuron_prob);
        wln!(
            "SplitRecurrent {}",
            if self.split_recurrent {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "SplitLoopedRecurrent {}",
            if self.split_looped_recurrent {
                "true"
            } else {
                "false"
            }
        );
        wln!("NeuronTries {}", self.neuron_tries);
        wln!("MutateAddLinkProb {:.20}", self.mutate_add_link_prob);
        wln!(
            "MutateAddLinkFromBiasProb {:.20}",
            self.mutate_add_link_from_bias_prob
        );
        wln!("MutateRemLinkProb {:.20}", self.mutate_rem_link_prob);
        wln!(
            "MutateRemSimpleNeuronProb {:.20}",
            self.mutate_rem_simple_neuron_prob
        );
        wln!("LinkTries {}", self.link_tries);
        wln!("MaxLinks {}", self.max_links);
        wln!("MaxNeurons {}", self.max_neurons);
        wln!("RecurrentProb {:.20}", self.recurrent_prob);
        wln!("RecurrentLoopProb {:.20}", self.recurrent_loop_prob);
        wln!("MutateWeightsProb {:.20}", self.mutate_weights_prob);
        wln!(
            "MutateWeightsSevereProb {:.20}",
            self.mutate_weights_severe_prob
        );
        wln!("WeightMutationRate {:.20}", self.weight_mutation_rate);
        wln!(
            "WeightMutationMaxPower {:.20}",
            self.weight_mutation_max_power
        );
        wln!("WeightReplacementRate {:.20}", self.weight_replacement_rate);
        wln!(
            "WeightReplacementMaxPower {:.20}",
            self.weight_replacement_max_power
        );
        wln!("MaxWeight {:.20}", self.max_weight);
        wln!("MinWeight {:.20}", self.min_weight);
        wln!(
            "MutateActivationAProb {:.20}",
            self.mutate_activation_a_prob
        );
        wln!(
            "MutateActivationBProb {:.20}",
            self.mutate_activation_b_prob
        );
        wln!(
            "ActivationAMutationMaxPower {:.20}",
            self.activation_a_mutation_max_power
        );
        wln!(
            "ActivationBMutationMaxPower {:.20}",
            self.activation_b_mutation_max_power
        );
        wln!(
            "TimeConstantMutationMaxPower {:.20}",
            self.time_constant_mutation_max_power
        );
        wln!("BiasMutationMaxPower {:.20}", self.bias_mutation_max_power);
        wln!("MinActivationA {:.20}", self.min_activation_a);
        wln!("MaxActivationA {:.20}", self.max_activation_a);
        wln!("MinActivationB {:.20}", self.min_activation_b);
        wln!("MaxActivationB {:.20}", self.max_activation_b);
        wln!(
            "MutateNeuronActivationTypeProb {:.20}",
            self.mutate_neuron_activation_type_prob
        );
        wln!(
            "ActivationFunction_SignedSigmoid_Prob {:.20}",
            self.activationfunction_signedsigmoid_prob
        );
        wln!(
            "ActivationFunction_UnsignedSigmoid_Prob {:.20}",
            self.activationfunction_unsignedsigmoid_prob
        );
        wln!(
            "ActivationFunction_Tanh_Prob {:.20}",
            self.activationfunction_tanh_prob
        );
        wln!(
            "ActivationFunction_TanhCubic_Prob {:.20}",
            self.activationfunction_tanhcubic_prob
        );
        wln!(
            "ActivationFunction_SignedStep_Prob {:.20}",
            self.activationfunction_signedstep_prob
        );
        wln!(
            "ActivationFunction_UnsignedStep_Prob {:.20}",
            self.activationfunction_unsignedstep_prob
        );
        wln!(
            "ActivationFunction_SignedGauss_Prob {:.20}",
            self.activationfunction_signedgauss_prob
        );
        wln!(
            "ActivationFunction_UnsignedGauss_Prob {:.20}",
            self.activationfunction_unsignedgauss_prob
        );
        wln!(
            "ActivationFunction_Abs_Prob {:.20}",
            self.activationfunction_abs_prob
        );
        wln!(
            "ActivationFunction_SignedSine_Prob {:.20}",
            self.activationfunction_signedsine_prob
        );
        wln!(
            "ActivationFunction_UnsignedSine_Prob {:.20}",
            self.activationfunction_unsignedsine_prob
        );
        wln!(
            "ActivationFunction_Linear_Prob {:.20}",
            self.activationfunction_linear_prob
        );
        wln!(
            "ActivationFunction_Relu_Prob {:.20}",
            self.activationfunction_relu_prob
        );
        wln!(
            "ActivationFunction_Softplus_Prob {:.20}",
            self.activationfunction_softplus_prob
        );
        wln!(
            "MutateNeuronTimeConstantsProb {:.20}",
            self.mutate_neuron_time_constants_prob
        );
        wln!(
            "MutateNeuronBiasesProb {:.20}",
            self.mutate_neuron_biases_prob
        );
        wln!(
            "MinNeuronTimeConstant {:.20}",
            self.min_neuron_time_constant
        );
        wln!(
            "MaxNeuronTimeConstant {:.20}",
            self.max_neuron_time_constant
        );
        wln!("MinNeuronBias {:.20}", self.min_neuron_bias);
        wln!("MaxNeuronBias {:.20}", self.max_neuron_bias);
        wln!(
            "DontUseBiasNeuron {}",
            if self.dont_use_bias_neuron {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "ArchiveEnforcement {}",
            if self.archive_enforcement {
                "true"
            } else {
                "false"
            }
        );
        wln!(
            "AllowLoops {}",
            if self.allow_loops { "true" } else { "false" }
        );
        wln!("DisjointCoeff {:.20}", self.disjoint_coeff);
        wln!("ExcessCoeff {:.20}", self.excess_coeff);
        wln!("ActivationADiffCoeff {:.20}", self.activation_a_diff_coeff);
        wln!("ActivationBDiffCoeff {:.20}", self.activation_b_diff_coeff);
        wln!("WeightDiffCoeff {:.20}", self.weight_diff_coeff);
        wln!(
            "TimeConstantDiffCoeff {:.20}",
            self.time_constant_diff_coeff
        );
        wln!("BiasDiffCoeff {:.20}", self.bias_diff_coeff);
        wln!(
            "ActivationFunctionDiffCoeff {:.20}",
            self.activation_function_diff_coeff
        );
        wln!("CompatTreshold {:.20}", self.compat_treshold);
        wln!("MinCompatTreshold {:.20}", self.min_compat_treshold);
        wln!(
            "CompatTresholdModifier {:.20}",
            self.compat_treshold_modifier
        );
        wln!(
            "CompatTreshChangeInterval_Generations {}",
            self.compat_tresh_change_interval_generations
        );
        wln!(
            "CompatTreshChangeInterval_Evaluations {}",
            self.compat_tresh_change_interval_evaluations
        );
        wln!(
            "MinDeltaCompatEqualGenomes {:.20}",
            self.min_delta_compat_equal_genomes
        );
        wln!("DivisionThreshold {:.20}", self.division_threshold);
        wln!("VarianceThreshold {:.20}", self.variance_threshold);
        wln!("BandThreshold {:.20}", self.band_threshold);
        wln!("InitialDepth {}", self.initial_depth);
        wln!("MaxDepth {}", self.max_depth);
        wln!("IterationLevel {}", self.iteration_level);
        wln!(
            "TournamentSelection {}",
            if self.tournament_selection {
                "true"
            } else {
                "false"
            }
        );
        wln!("TournamentSize {}", self.tournament_size);
        wln!("CPPN_Bias {:.20}", self.cppn_bias);
        wln!("Width {:.20}", self.width);
        wln!("Height {:.20}", self.height);
        wln!("Qtree_X {:.20}", self.qtree_x);
        wln!("Qtree_Y {:.20}", self.qtree_y);
        wln!("Leo {}", if self.leo { "true" } else { "false" });
        wln!("LeoThreshold {:.20}", self.leo_threshold);
        wln!("LeoSeed {}", if self.leo_seed { "true" } else { "false" });
        wln!(
            "GeometrySeed {}",
            if self.geometry_seed { "true" } else { "false" }
        );
        wln!("Elitism {:.20}", self.elite_fraction);
        wln!("NEAT_ParametersEnd");

        Ok(())
    }
}

impl std::fmt::Debug for Parameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parameters")
            .field("population_size", &self.population_size)
            .field("speciation", &self.speciation)
            .field("min_species", &self.min_species)
            .field("max_species", &self.max_species)
            .finish()
    }
}
