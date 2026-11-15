# Scientific attribution

The G-Transformer core in `src/c2csynth/vendor/xiong_gt.py` is an independent
implementation of the architecture and algorithms in Xiong et al.,
*G-Transformer: Counterfactual Outcome Prediction under Dynamic and Time-varying
Treatment Regimes*, MLHC 2024, PMLR 252.
<https://proceedings.mlr.press/v252/xiong24a.html>

It is not the original authors' software. The published Compact GT baseline
uses the factual-history encoder, a centered current-assignment response
component, and the common marketplace state updates described in this
repository. Its configuration is recorded in the experiment manifest.

External Python dependencies are installed separately and retain their own
licenses. No dependency binaries are included in this source distribution.
