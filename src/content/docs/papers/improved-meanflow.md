---
title: Improved Mean Flows — On the Challenges of Fastforward Generative Models
sidebar:
  order: 2
---

## Paper status

- Venue: CVPR 2026 Spotlight
- Role in this handbook: direct revision of original MeanFlow

## One-sentence summary

iMF reformulates the original network-dependent MeanFlow target as a standard-looking velocity regression parameterized through an average-velocity network, replaces the JVP's sample-specific conditional tangent with an estimated marginal velocity, and makes CFG flexible through conditioning.

## Main issues addressed

1. Original MF target depends on the current network.
2. Original MF fixes CFG scale at training time.

## Main reported result

ImageNet 256×256, one NFE, trained from scratch, no distillation: FID 1.72.

## Research interpretation

After iMF, “fix the network-dependent target” should no longer be treated as an open novelty by itself. More promising directions include temporal difficulty, endpoint-quality mismatch, gradient geometry, and flow-map parameterization.
