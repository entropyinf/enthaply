
/// Configuration for SenseVoiceEncoderSmall
#[derive(PartialEq)]
pub struct EncoderConfig {
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
    /// Number of attention heads
    pub attention_heads: usize,
    /// Number of linear units
    pub linear_units: usize,
    /// Number of encoder blocks
    pub num_blocks: usize,
    /// Number of TP blocks
    pub tp_blocks: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Attention dropout rate
    pub attention_dropout_rate: f32,
    /// Kernel size
    pub kernel_size: usize,
    /// SANM shift
    pub sanm_shfit: usize,
    /// Whether to normalize before computation
    pub normalize_before: bool,
    /// Whether to concatenate after attention
    pub concat_after: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_size: 0,
            output_size: 0,
            attention_heads: 4,
            linear_units: 2048,
            num_blocks: 6,
            tp_blocks: 0,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
            kernel_size: 11,
            sanm_shfit: 0,
            normalize_before: true,
            concat_after: false,
        }
    }
}