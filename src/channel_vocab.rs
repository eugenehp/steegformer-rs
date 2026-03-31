//! Global channel name vocabulary for ST-EEGFormer.
//!
//! ST-EEGFormer supports up to 142 EEG channels. Each channel maps to
//! a learned embedding index (0–141). The mapping is from the pre-training
//! pickle file `sen_chan_idx.pkl`.
//!
//! The `enc_channel_emd` Embedding table has shape [145, embed_dim] where
//! indices 0–141 are used channels and 142–144 are padding/reserved.

/// All 142 channel names indexed by their embedding index.
/// Index `i` maps to `CHANNEL_VOCAB[i]`.
pub const CHANNEL_VOCAB: &[&str] = &[
    "C1",       // 0
    "Pz",       // 1
    "C4",       // 2
    "F6",       // 3
    "FTT8h",    // 4
    "Oz",       // 5
    "Fp1",      // 6
    "FCC5h",    // 7
    "TPP8h",    // 8
    "CPP6h",    // 9
    "C2",       // 10
    "F4",       // 11
    "OI2h",     // 12
    "AF4",      // 13
    "FCz",      // 14
    "CCP6h",    // 15
    "TP8",      // 16
    "POO10h",   // 17
    "FC1",      // 18
    "FC6",      // 19
    "C5",       // 20
    "P8",       // 21
    "FT8",      // 22
    "P6",       // 23
    "P9",       // 24
    "Fz",       // 25
    "AFF1",     // 26
    "TPP10h",   // 27
    "AFF2",     // 28
    "P10",      // 29
    "CPP2h",    // 30
    "M1",       // 31
    "FCC6h",    // 32
    "FTT7h",    // 33
    "FC2",      // 34
    "PPO2",     // 35
    "AFp3h",    // 36
    "AF7",      // 37
    "PO10",     // 38
    "AF8",      // 39
    "CPP1h",    // 40
    "P7",       // 41
    "F1",       // 42
    "AFp4h",    // 43
    "PO9",      // 44
    "FT9",      // 45
    "CP2",      // 46
    "Iz",       // 47
    "FCC1h",    // 48
    "FC5",      // 49
    "T5",       // 50
    "CP5",      // 51
    "CP6",      // 52
    "FFC5h",    // 53
    "F2",       // 54
    "M2",       // 55
    "POO9h",    // 56
    "AFF5h",    // 57
    "PO4",      // 58
    "POO3h",    // 59
    "Fp2",      // 60
    "T3",       // 61
    "CP4",      // 62
    "POz",      // 63
    "TTP7h",    // 64
    "T7",       // 65
    "A2",       // 66
    "CCP4h",    // 67
    "T8",       // 68
    "PPO10h",   // 69
    "FC3",      // 70
    "F3",       // 71
    "F5",       // 72
    "A1",       // 73
    "P3",       // 74
    "FC4",      // 75
    "FCC2h",    // 76
    "FFC6h",    // 77
    "FFT8h",    // 78
    "CCP2h",    // 79
    "CPP4h",    // 80
    "T6",       // 81
    "FTT9h",    // 82
    "PPO6h",    // 83
    "CP3",      // 84
    "CP1",      // 85
    "AF3",      // 86
    "FT10",     // 87
    "OI1h",     // 88
    "TPP9h",    // 89
    "P5",       // 90
    "I2",       // 91
    "CCP1h",    // 92
    "T4",       // 93
    "CCP3h",    // 94
    "O1",       // 95
    "PO5",      // 96
    "PPO9h",    // 97
    "PPO5h",    // 98
    "P1",       // 99
    "AFz",      // 100
    "PO6",      // 101
    "PO3",      // 102
    "O2",       // 103
    "CPP5h",    // 104
    "FFC1h",    // 105
    "FCC4h",    // 106
    "FFT7h",    // 107
    "FFC2h",    // 108
    "FFC4h",    // 109
    "Cz",       // 110
    "TP7",      // 111
    "Fpz",      // 112
    "FTT10h",   // 113
    "PO7",      // 114
    "CPP3h",    // 115
    "P4",       // 116
    "P2",       // 117
    "F8",       // 118
    "CPz",      // 119
    "FCC3h",    // 120
    "FFC3h",    // 121
    "FT7",      // 122
    "I1",       // 123
    "TTP8h",    // 124
    "AFF6h",    // 125
    "CCP5h",    // 126
    "C6",       // 127
    "PPO1",     // 128
    "PO8",      // 129
    "C3",       // 130
    "POO4h",    // 131
    "TPP7h",    // 132
    "F7",       // 133
    "T9",       // 134
    "TP9",      // 135
    "T10",      // 136
    "TP10",     // 137
    "POO1",     // 138
    "POO2",     // 139
    "PPO1h",    // 140
    "PPO2h",    // 141
];

/// Vocabulary size (number of unique channels).
pub const VOCAB_SIZE: usize = 142;

/// Maximum embedding table size (matches nn.Embedding(145, D) in Python).
pub const EMBEDDING_TABLE_SIZE: usize = 145;

/// Look up the vocabulary index for a channel name (case-insensitive).
pub fn channel_index(name: &str) -> Option<usize> {
    let upper = name.to_uppercase();
    CHANNEL_VOCAB.iter().position(|&v| v.to_uppercase() == upper)
}

/// Look up vocabulary indices for multiple channel names.
pub fn channel_indices(names: &[&str]) -> Vec<Option<usize>> {
    names.iter().map(|n| channel_index(n)).collect()
}

/// Look up vocabulary indices, panicking on missing names.
pub fn channel_indices_unwrap(names: &[&str]) -> Vec<i64> {
    names.iter().map(|n| {
        channel_index(n)
            .unwrap_or_else(|| panic!("Channel '{}' not in ST-EEGFormer vocabulary", n)) as i64
    }).collect()
}

/// Common channel subsets used in BCI datasets.

/// BCI Competition IV-2a: 22 channels.
pub const BCI_COMP_IV_2A: &[&str] = &[
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
];

/// Standard 10-20 system (19 channels + 2 reference).
pub const STANDARD_10_20: &[&str] = &[
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        assert_eq!(CHANNEL_VOCAB.len(), VOCAB_SIZE);
    }

    #[test]
    fn test_channel_index() {
        assert_eq!(channel_index("C1"), Some(0));
        assert_eq!(channel_index("Pz"), Some(1));
        assert_eq!(channel_index("C3"), Some(130));
        assert_eq!(channel_index("NONEXISTENT"), None);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(channel_index("c3"), Some(130));
        assert_eq!(channel_index("FZ"), Some(25));
    }

    #[test]
    fn test_bci_comp_channels() {
        for ch in BCI_COMP_IV_2A {
            assert!(channel_index(ch).is_some(), "BCI IV-2a channel {ch} not found");
        }
    }
}
