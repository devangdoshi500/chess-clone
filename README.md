# Chess Clone

A chess move predictor that learns how a particular person tends to play, not just which move an engine prefers.

[Try the interactive demo](https://chess-clone-style-explorer.vercel.app)

A population policy ranks every legal move; a small player-specific adapter learns from earlier games. On 34,350 held-out decisions from 10 players, personalization reduced measured style error **26.2%** versus the shared policy. This is a replay-position result, not proof of convincing full-game cloning.

The demo compares three policies on nine curated positions. Those examples are illustrative, not an accuracy benchmark.

Built with Python, CatBoost, python-chess, React, and Vercel. To run the demo locally:

```bash
./scripts/run_demo.sh
```
