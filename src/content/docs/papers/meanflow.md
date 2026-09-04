---
title: Mean Flows for One-step Generative Modeling
sidebar:
  order: 1
---

## Paper status

- Venue: NeurIPS 2025 Oral
- Role in this handbook: original MeanFlow baseline

## One-sentence summary

MeanFlow replaces a purely instantaneous velocity predictor with an interval-average velocity field and derives a differential identity that allows this average field to be trained without explicitly integrating the ground-truth trajectory during training.

## Reading checklist

- Motivation: reduce NFE for generative ODE sampling
- Core object: average velocity $u(z_t,r,t)$
- Key derivation: MeanFlow Identity
- Training mechanism: JVP-based target construction
- Inference: $z_0=z_1-u_\theta(z_1,0,1)$
- Reported ImageNet-256 one-step FID: 3.43 for MF-XL/2

## My interpretation

The central shift is not “a better local velocity,” but learning an object whose semantics already include finite-time transport.
