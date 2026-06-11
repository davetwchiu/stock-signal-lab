# Decision Mode

Decision Mode is the default view for Stock Signal Lab. It is designed to answer one practical question: what is the system suggesting today, and why?

It is research and decision support only. It is not financial advice, not a live trading system, not a brokerage connection, and not an automatic stock picker.

## ML Score

The ML score is a 0-100 research score based on the model's estimated forward relative strength and drawdown-risk state.

A higher score means the selected model and feature group currently see a better balance of opportunity and risk. It is not a price target, not a guaranteed forecast, and not proof that a stock will outperform.

## Drawdown-Risk Probability

Drawdown-risk probability is the model's estimated probability that the ticker may experience a material forward drawdown under the selected research setup.

High drawdown-risk probability can reduce or block exposure even when the ML score is strong.

## Suggested Actions

- `Add`: target exposure is meaningfully above the current or assumed position.
- `Hold`: current evidence is constructive enough to keep exposure, but not strong enough for aggressive adding.
- `Trim`: target exposure is below current or assumed exposure because risk has risen.
- `Exit`: target exposure is zero while current exposure is meaningful.
- `Watch`: target exposure is zero and the ticker does not currently justify exposure.

## Target Exposure Buckets

Decision Mode shows target exposure as buckets: `0%`, `25%`, `50%`, `75%`, or `100%`.

The bucket is relative to the configured maximum single-position exposure, not the whole portfolio. For example, if max single-position exposure is 12%, a `75%` bucket means roughly 9% target weight before portfolio-level caps.

## Confidence

Confidence is `Low`, `Medium`, or `High`. It reflects how far the score and risk probability are from the undecided zone. Low confidence means the system is closer to a toss-up and the action should be interpreted cautiously.

## What The System Does Not Know

The system does not know future news, earnings surprises, macro shocks, liquidity events, execution quality, taxes, your personal risk tolerance, or your full financial situation.

It also does not prove predictive power. Repeated experiments can overfit. Treat suggestions as structured research prompts, not instructions.

## Locked Defaults And Profiles

Decision Mode uses `config/default_decision_mode.yaml` by default. Preset profiles keep the interface simple:

- Conservative: more cash, smaller positions, stricter drawdown-risk threshold.
- Balanced: default.
- Aggressive: less cash, larger positions, more tolerance for risk.

Advanced parameter tuning is intentionally hidden unless `Advanced override` is enabled or the user opens Research Lab.

