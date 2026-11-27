use crate::core::{Error, Result};
use super::arch_enum::Arch;
use super::kv::Tensor;

/// The layer in which a tensor is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorLayer {
    Input,
    Repeating,
    Output,
}

/// ggml ops referenced by the tensor classification table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Div,
    GetRows,
    Im2Col,
    Mul,
    MulMat,
    MulMatId,
    None,
    Rope,
    RwkvWkv6,
    Scale,
    SetRows,
    SsmConv,
    SsmScan,
}

/// Where a tensor is used and the op that consumes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub layer: TensorLayer,
    pub op: Op,
}

/// Tensor classification used for weight buffer-type assignment.
///
/// A mismatch can lead to sub-optimal buffer assignment and extra overhead.
pub fn tensor_info_for(tensor: Tensor) -> Result<&'static TensorInfo> {
    match tensor {
        Tensor::TokenEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::GetRows,
        }),
        Tensor::TokenEmbdNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::TokenEmbdProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TokenTypes => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::GetRows,
        }),
        Tensor::PosEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::GetRows,
        }),
        Tensor::Dense2Out => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::Dense3Out => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::Output => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::OutputNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::OutputNormLfm2 => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::OutputResQuery => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::RopeFreqs => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Rope,
        }),
        Tensor::RopeFactorsLong => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Rope,
        }),
        Tensor::RopeFactorsShort => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Rope,
        }),
        Tensor::AttnQ => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQkv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnNorm2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnOutNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnPostNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnSinks => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Scale,
        }),
        Tensor::AttnSsmax => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnResidAlpha => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnResidBeta => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnRel => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnRelBias => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQClip => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnRelClip => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnSinkProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQInpNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnKInpNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnVInpNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnKConv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::AttnVConv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::AttnKConvProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnVConvProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnPostConv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::AttnResQuery => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnGateInp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnGateInpShexp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnPostNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnPostNorm1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnPostNorm2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnNormAdaDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnNormAdaUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnPreNorm2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnAct => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Div,
        }),
        Tensor::FfnNormExps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnDownExps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnGateExps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnUpExps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnGateUpExps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnDownShexp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnGateShexp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnUpShexp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnDownChexps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnGateChexps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnUpChexps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        Tensor::FfnExpProbsB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::FfnExpProbsBVl => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::FfnLatentDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnLatentUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnPostConv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::FfnResQuery => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopMsgQuery => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopQCtrlDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopQCtrlGain => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopQCtrlForget => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopKCtrlDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopKCtrlGain => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopKCtrlForget => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopVCtrlDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopVCtrlGain => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopVCtrlForget => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopFfnCtrlDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopFfnCtrlGain => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LoopFfnCtrlForget => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::StreamEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Add,
        }),
        Tensor::DepthMemStateWrite => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemMetaWrite => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemEntryBias => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemQuery => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemKey => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemValue => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemOut => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemRouteState => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemRouteRetr => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemRouteDiff => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemRouteHist => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemRoute => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DepthMemStreamGate => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ContHiddenState => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ContHiddenRetr => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ContHiddenDiff => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ContHiddenRoute => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ContHidden => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ContOutput => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::HcAttnMix => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcAttnRead => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcAttnWrite => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnMix => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnRead => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnWrite => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnPassLoraDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnPassLoraUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnPassGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnExpEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::FfnPreExpNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnPostExpNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnResidAlpha => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnResidBeta => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnQNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnKNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::LayerOutNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::LayerOutScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnVresLambda => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PerLayerTokenEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::GetRows,
        }),
        Tensor::PerLayerModelProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PerLayerInpGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PerLayerProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PerLayerProjNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PerLayerPostNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AltupProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::AltupUnembdProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::AltupCorrectCoef => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AltupCorrectScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AltupPredictCoef => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AltupRouter => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AltupRouterNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::LaurelL => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LaurelR => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LaurelPostNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmIn => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmConv1d => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::SsmX => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmDt => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmDtNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmA => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmScan,
        }),
        Tensor::SsmANoscan => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmBNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmCNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmD => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmBetaAlpha => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmAlpha => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmConv1dQ => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmConv1dK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmConv1dV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::SsmFA => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmFB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmBeta => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmRda => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmGA => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmGB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmG => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnResScore => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnResScore => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::OutputResScore => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::FfnRoutedDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnRoutedUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnRoutedNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::ValueEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::GetRows,
        }),
        Tensor::AttnVeGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnXsaAlpha => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnResidScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnResidScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::MuddDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::MuddUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixW0 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixW1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixW2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixA0 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixA1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixA2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixV0 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixV1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixV2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixG1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixG2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixKK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::TimeMixKA => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::TimeMixRK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::TimeMixLerpX => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::TimeMixLerpW => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixLerpK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixLerpV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixLerpR => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixLerpG => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixLerpFused => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixFirst => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::RwkvWkv6,
        }),
        Tensor::TimeMixDecay => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::TimeMixDecayW1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixDecayW2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixKey => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixValue => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixReceptance => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::TimeMixLn => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::TimeMixOutput => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ChannelMixLerpK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::ChannelMixLerpR => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::ChannelMixKey => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ChannelMixReceptance => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ChannelMixValue => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQA => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnKvAMqa => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnKvB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnKv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnKvNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnOutA => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnOutB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnKB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnVB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQANorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnKvANorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::HcHeadFn => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::HcHeadBase => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Add,
        }),
        Tensor::HcHeadScale => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::HcHeadNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::HcHeadDown => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::HcHeadUp => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::HcAttnFn => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcAttnBase => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::HcAttnScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::HcFfnFn => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnBase => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::HcFfnScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::HcAttnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::HcAttnDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcAttnUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcAttnInject => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::HcFfnDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnInject => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcAttnDownInject => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::HcFfnDownInject => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnQkvGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::SsmBetaAlphaCat => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PleKey => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PleValue => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PleNormKey => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PleNormQuery => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PleNormConv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PleConv1d => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::AttnCompressorWkv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnCompressorWgate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnCompressorApe => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::AttnCompressorNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnSubNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnSubNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DecAttnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DecAttnQ => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecAttnK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecAttnV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecAttnOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecAttnRelB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::DecCrossAttnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DecCrossAttnNormKv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DecCrossAttnQ => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecCrossAttnK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecCrossAttnV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecCrossAttnOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecCrossAttnRelB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::None,
        }),
        Tensor::DecFfnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DecFfnGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecFfnDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecFfnUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DecOutputNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::EncAttnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::EncAttnQ => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncAttnK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncAttnV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncAttnOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncAttnRelB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::EncFfnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::EncFfnGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncFfnDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncFfnUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EncOutputNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::Cls => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ClsOut => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::ClsNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionHiddenNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionProjMemory => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionProjQuestion => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionProjOptionQuestion => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionProjGlobal => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionProjOptionContext => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionProjOptionLexical => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionOptionSummaryNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionFieldNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionOptionNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionScales => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionScorer => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionScorerOut => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionQueryNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::DecisionKey => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionQuery => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionOptionMlp => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DecisionQueryMlp => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::Conv1d => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Im2Col,
        }),
        Tensor::ConvnextDw => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Im2Col,
        }),
        Tensor::ConvnextNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::ConvnextPw1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ConvnextPw2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ConvnextGamma => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PosNetConv1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Im2Col,
        }),
        Tensor::PosNetConv2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Im2Col,
        }),
        Tensor::PosNetNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PosNetNorm1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PosNetNorm2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PosNetAttnNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::PosNetAttnQ => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PosNetAttnK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PosNetAttnV => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::PosNetAttnOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ShortconvConv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::SsmConv,
        }),
        Tensor::ShortconvInproj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::ShortconvOutproj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::VisexpAttnQkv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::VisexpAttnOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::VisexpFfnGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::VisexpFfnDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::VisexpFfnUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerKNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::IndexerProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerAttnK => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerAttnQB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerQProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerKProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerQNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::IndexerSsmax => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::IndexerCompressorWkv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerCompressorWgate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerCompressorApe => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::IndexerCompressorNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::IndexerKpoolGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::IndexerKpoolApe => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::IndexerKpoolValue => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::FfnGateTid2eid => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::NextnProjPre => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::NextnProjPost => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::NextnEhProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::NextnEmbedTokens => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::GetRows,
        }),
        Tensor::NextnEnorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::NextnHnorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::NextnSharedHeadHead => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::NextnSharedHeadNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::NextnHcHeadNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::NextnHcHeadDown => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::NextnHcHeadUp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::MaskedEmbdCentroids => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::None,
        }),
        Tensor::MaskedEmbdOrdering => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::None,
        }),
        Tensor::HrmZLInit => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::Fc => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::D2t => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::SetRows,
        }),
        Tensor::DsparkMarkovW1 => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::GetRows,
        }),
        Tensor::DsparkMarkovW2 => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DsparkConfProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DsparkJointProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DsparkMaskEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Input,
            op: Op::GetRows,
        }),
        Tensor::DsparkMarkovCtxMix => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DsparkMarkovCtxNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::EngramKv => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::EngramGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::EngramKvB => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::EngramLookupNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::EngramPreGamma => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::EngramHcScale => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Add,
        }),
        Tensor::DflashAttnConvBase => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DflashAttnConvProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DflashFfnConvBase => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DflashFfnConvProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DflashSelectorPrev => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::GetRows,
        }),
        Tensor::DflashSelectorNext => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::GetRows,
        }),
        Tensor::DflashSelectorHidden => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::DflashCtxMix => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DflashCtxNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::DflashAttnKCtx => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DflashAttnVCtx => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::DflashAttnKCtxNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::LilicorrTokenProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::LilicorrPassHiddenProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::LilicorrSlotEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Add,
        }),
        Tensor::LilicorrRankEmbd => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Add,
        }),
        Tensor::LilicorrAttnBias => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Add,
        }),
        Tensor::LilicorrContextProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::LilicorrAttnInProj => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::LilicorrOutputNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::LilicorrAnchorNorm => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::Mul,
        }),
        Tensor::LilicorrFactorInputProj => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::LilicorrOutHead => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::LilicorrInHead => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::LilicorrAnchorOutHead => Ok(&TensorInfo {
            layer: TensorLayer::Output,
            op: Op::MulMat,
        }),
        Tensor::CcaConvGrp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::CcaTemp => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::ResScaleIn => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::AttnResScaleHs => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::FfnResScaleHs => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::RouterEda => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::RouterNorm => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::Mul,
        }),
        Tensor::RouterFc1 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::RouterFc2 => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::RouterOut => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnVGate => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMat,
        }),
        Tensor::AttnVExps => Ok(&TensorInfo {
            layer: TensorLayer::Repeating,
            op: Op::MulMatId,
        }),
        _ => Err(Error::new(format!("no tensor info for {:?}", tensor))),
    }
}

// ---------------------------------------------------------------------------
// Architecture capability flags
// ---------------------------------------------------------------------------

/// Whether the architecture is fully recurrent.
pub fn arch_is_recurrent(arch: Arch) -> bool {
    match arch {
        Arch::Mamba => true,
        Arch::Mamba2 => true,
        Arch::Rwkv6 => true,
        Arch::Rwkv6qwen2 => true,
        Arch::Rwkv7 => true,
        Arch::Arwkv7 => true,
        _ => false,
    }
}

/// Whether the architecture mixes attention with recurrent/state layers.
pub fn arch_is_hybrid(arch: Arch) -> bool {
    match arch {
        Arch::Jamba => true,
        Arch::FalconH1 => true,
        Arch::Plamo2 => true,
        Arch::GraniteHybrid => true,
        Arch::Lfm2 => true,
        Arch::Lfm2moe => true,
        Arch::Lfm3 => true,
        Arch::Zaya1 => true,
        Arch::NemotronH => true,
        Arch::NemotronHMoe => true,
        Arch::Metis16 => true,
        Arch::Qwen3next => true,
        Arch::KimiLinear => true,
        Arch::Bailingmoe3 => true,
        Arch::KimiK3 => true,
        Arch::KimiK31 => true,
        Arch::Raon1Mini => true,
        Arch::Glm5Next => true,
        Arch::Qwen35 => true,
        Arch::Qwen35moe => true,
        Arch::CloverMoe => true,
        Arch::Qwen4exp => true,
        Arch::Deepseek4 => true,
        Arch::Deepseek41 => true,
        Arch::Minimax01 => true,
        Arch::Zonos3 => true,
        Arch::Inkling2 => true,
        Arch::Inkling2Loop => true,
        _ => false,
    }
}

/// Whether the architecture is a diffusion language model.
pub fn arch_is_diffusion(arch: Arch) -> bool {
    match arch {
        Arch::Dream => true,
        Arch::Llada => true,
        Arch::LladaMoe => true,
        Arch::Rnd1 => true,
        _ => false,
    }
}

/// Whether roll-back of the recurrent state is supported.
pub fn arch_supports_rs_rollback(arch: Arch) -> bool {
    match arch {
        Arch::KimiK3 => true,
        Arch::KimiK31 => true,
        Arch::Raon1Mini => true,
        Arch::Qwen35 => true,
        Arch::Qwen35moe => true,
        Arch::CloverMoe => true,
        Arch::Qwen4exp => true,
        Arch::Deepseek4 => true,
        Arch::Deepseek41 => true,
        Arch::Glm5Next => true,
        Arch::NemotronH => true,
        Arch::NemotronHMoe => true,
        Arch::Lfm2 => true,
        Arch::Lfm2moe => true,
        Arch::Lfm3 => true,
        Arch::Bailingmoe3 => true,
        Arch::Inkling2 => true,
        Arch::Inkling2Loop => true,
        _ => false,
    }
}

/// These models pick weights, routing or input meaning per ubatch based on
/// token vs embd input.
/// Whether the architecture supports mixed token/embd batches.
pub fn arch_supports_mixed_batch(arch: Arch) -> bool {
    match arch {
        Arch::Cogvlm => false,
        Arch::Deepseek4 => false,
        Arch::Deepseek41 => false,
        Arch::GraniteSwitch => false,
        Arch::Eagle3 => false,
        Arch::Dflash => false,
        Arch::Gemma4Assistant => false,
        Arch::Limite => false,
        Arch::Inkling2 => false,
        Arch::Inkling2Loop => false,
        _ => true,
    }
}

/// Whether the architecture supports state-machine tensors.
pub fn arch_supports_sm_tensor(arch: Arch) -> bool {
    match arch {
        Arch::CloverMoe => false,
        Arch::Zaya1 => false,
        Arch::Grok => false,
        Arch::Mpt => false,
        Arch::Plamo2 => false,
        Arch::Minicpm3 => false,
        Arch::Gemma3n => false,
        Arch::Mamba => false,
        Arch::Mamba2 => false,
        Arch::Jamba => false,
        Arch::FalconH1 => false,
        Arch::Inkling2 => false,
        Arch::Inkling2Loop => false,
        Arch::Olmo2 => false,
        Arch::Olmoe => false,
        Arch::Deepseek2 => false,
        Arch::Deepseek32 => false,
        Arch::HyV4 => false,
        Arch::HyV4Flash => false,
        Arch::Dots3note => false,
        Arch::Xing40 => false,
        Arch::GlmDsa => false,
        Arch::Bitnet => false,
        Arch::T5 => false,
        Arch::NemotronH => false,
        Arch::NemotronHMoe => false,
        Arch::GraniteHybrid => false,
        Arch::Minimax01 => false,
        Arch::MinimaxM2 => false,
        Arch::MinimaxM3 => false,
        Arch::Mistral4 => false,
        Arch::MistralLarge4 => false,
        Arch::Mellum => false,
        Arch::Flint => false,
        Arch::Laguna => false,
        Arch::KimiLinear => false,
        Arch::Bailingmoe3 => false,
        Arch::KimiK3 => false,
        Arch::KimiK31 => false,
        Arch::Raon1Mini => false,
        Arch::Qwen3tts => false,
        Arch::Limite => false,
        _ => true,
    }
}

/// Whether the down projection must stay in bf16.
pub fn arch_mmid_down_bf16(arch: Arch) -> bool {
    match arch {
        Arch::Mistral4 => true,
        Arch::Mellum => true,
        Arch::Flint => true,
        Arch::Laguna => true,
        _ => false,
    }
}

/// Whether the architecture can be saved back to GGUF.
pub fn arch_supports_save(arch: Arch) -> bool {
    match arch {
        Arch::SensenovaU1 => false,
        Arch::Gemma3n => false,
        Arch::Bitnet => false,
        Arch::T5 => false,
        Arch::Apertus => false,
        Arch::Step35 => false,
        Arch::Clef => false,
        Arch::Inkling2Loop => false,
        Arch::VllmDecision2 => false,
        _ => true,
    }
}

/// Alias of `arch_supports_sm_tensor`.
pub fn llama_arch_supports_sm_tensor(arch: Arch) -> bool { arch_supports_sm_tensor(arch) }

/// Alias of `arch_supports_mixed_batch`.
pub fn llama_arch_supports_mixed_batch(arch: Arch) -> bool { arch_supports_mixed_batch(arch) }
