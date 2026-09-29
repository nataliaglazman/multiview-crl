**Content/style implementation and literature audit — 29 September 2026**

The evidence points first to the objective and the decoder's information paths, rather than an identifiable numerical choice of loss weights. There are two distinct problems: the supplied step-69001 checkpoint routes ventricular changes mainly through style; the current working tree also introduces a cross-subject reconstruction target that conflicts with preserving individual-specific appearance. The newer change should not be used to explain the older checkpoint.

This audit uses the two supplied reports, static inspection of the current source, and primary papers. It does not rerun the checkpoint. Its saved settings and weights were not available locally, and the inspected Python environments lack PyTorch. Existing source edits were preserved; only this report and an independent analytical example were added.

**What the measurements establish**

| Mean response to a ventricular-size intervention | T1 | FLAIR |
|---|---:|---:|
| Joint decoder response gain | 0.9067 | 0.7293 |
| Content contribution gain | -0.00105 | -0.00073 |
| Style contribution gain | 0.9077 | 0.7300 |
| Interaction RMS ratio | 0.00412 | 0.00547 |

These means cover 62 measurable subjects per view. The intervention changed ventricular anatomy while holding acquisition settings fixed. Thus, legitimate differences in gain between people do not explain this style response: the style path also carries the changed anatomy. The useful average directional response is overwhelmingly attributed to style in this test. Gains are projections onto the input intervention effect, not percentages of information. These results do not establish that every subject behaves identically or that content contains no ventricular information.

The weak held-out probes support concern about content, but cannot establish information absence. The T1 statistics/RBF probe reaches R² 0.126; its improvement over GAP has a confidence interval crossing zero. Poorer high-dimensional probes can reflect limited samples and readout geometry. The native content response being larger than its GAP response also warns against treating GAP as a complete representation audit.

**Individual-specific style changes the valid training target**

Write an image as `x[i,v] = render_v(anatomy_i, acquisition_i,v)`. A paired T1/FLAIR subject shares anatomy; another person's acquisition parameters need not match. Domain-specific style means a domain has a style space, not that every image in that domain has the same style value.

For `decode(content_i,T1, style_j,FLAIR)`, the intended target is `render_FLAIR(anatomy_i, acquisition_j,FLAIR)`. It is generally neither subject i's original FLAIR nor subject j's original FLAIR.

The current working tree selects donor j in [training/main_multimodal.py](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/training/main_multimodal.py:271), swaps those style rows in [models/vqvae.py](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/models/vqvae.py:1669), but always uses subject i's original other view as the target in [training/losses.py](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/training/losses.py:1907). `other_subject` is now the parser/default-YAML choice. The loss documentation acknowledges a nonzero floor, but this is more than irreducible error: the objective can prefer ignoring donor appearance. With detached injection, that pressure directly affects the decoder and content path rather than backpropagating through the style injection.

The accompanying [gain example](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/out/content_style_audit_20260929/gain_target_counterexample.py) isolates this issue. For independent gains 0.7, 1.0 and 1.3, a perfect donor-style transfer scores MAE 0.2667 against the recipient's original appearance. Ignoring the donor and always using median gain scores better: 0.2000. With the correct mixed target, perfect transfer instead scores 0 and the median scores 0.2000. This is an exact toy calculation, not a checkpoint measurement.

If the goal were deliberately to remove individual appearance, this could be a different regularizer. It is inconsistent with a style code intended to retain and transfer that appearance. Merely changing the target to the donor image would introduce the opposite error: donor anatomy.

**Relevant papers and their implementation differences**

| Paper | Mechanism worth comparing | Implication here |
|---|---|---|
| [MUNIT — Huang et al., ECCV 2018](https://arxiv.org/html/1804.04732) | Spatial content; globally pooled style vector; style-conditioned AdaIN; content and style reconstruction after translation (Eqs. 2–3). | Style varies within a domain. Re-encoding a generated mixture is used to recover its source content and chosen style; its pixels are not required to reproduce the recipient's original style. |
| [DRIT — Lee et al., ECCV 2018](https://arxiv.org/html/1808.00948) | Swaps attributes between noncorresponding images, re-encodes, and swaps back before comparing with the originals; also uses attribute regression and adversarial objectives (§3.2–3.3). | Its cross-cycle loss is different from one swap followed by an original-image pixel target. |
| [SDNet — Chartsias et al., Medical Image Analysis 2019](https://arxiv.org/html/1903.09467) | Spatial anatomical maps and a nonspatial modality vector; global channel-wise FiLM; modality-code reconstruction; anatomical supervision in its medical experiments. | Closest anatomical analogy, but its supervision and anatomical bottleneck are substantially stronger than yours. Its reported separation is not evidence that an unconstrained spatial style branch should separate on its own. |
| [Swapping Autoencoder — Park et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/50905d7b2216bfeccb5b41016357176b-Paper.pdf) | Spatial structure, global texture, and a patch co-occurrence discriminator connecting generated texture to the donor (§3.3). | The paper explicitly discusses that swapping alone does not assign the intended structure/texture meanings. Its additional constraint is task-specific; importing a GAN is not automatically necessary here. |
| [Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style — von Kügelgen et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/8929c70f8d710e412d38da624b21c3c8-Paper.pdf) | Content recovery uses assumptions about the generator, style variation, and invertibility or entropy maximization. Its discussion after Theorem 4.4 distinguishes the entropy objective from Barlow Twins. | Good covariance/alignment scores do not by themselves prove that every anatomical factor is retained. This is a theoretical limitation, not evidence that replacing Barlow Twins alone will fix this run. |

These papers motivate architectural and objective choices; none proves that a global style vector cannot encode ventricular size. A scalar anatomical factor can fit inside a small vector.

**Why the older checkpoint can exploit style**

1. **Style has a spatial route into the decoder.** The model and CLI default `style_spatial_size` to 0, preserving the encoder's spatial grid. The style codebook assigns a categorical code at each site. Four raw style channels therefore do not imply four scalar style variables. The default style embedding size and vocabulary inherit the content settings unless overridden ([codebook construction](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/models/vqvae.py:858)). Quantization limits capacity but does not assign semantics. With S sites and K entries, the upper bound on discrete capacity is S log2(K) bits, not log2(K); realized information can be lower.

   Some experiment YAML files override the spatial default to 1 or 4. Therefore, the checkpoint's saved `settings.json` and actual injected tensor shapes must decide its configuration. The reported unequal native and GAP style response RMS is consistent with a spatial code and inconsistent with an actual 1×1×1 tensor at that tap.

2. **Same-subject cross reconstruction leaves a shortcut.** The older objective decodes `content_i,T1` with `style_i,FLAIR` and targets `x_i,FLAIR`. A style representation carrying most of that FLAIR image can solve the objective with little useful content contribution. Increasing the weight rewards accurate output but does not distinguish this shortcut from the desired anatomy/style decomposition. The near-identical reconstruction and cross-reconstruction encoder-gradient directions in the supplied audit are consistent with redundant pressure; they are not a proof of this mechanism on their own.

3. **Detachment is not information removal.** [The detach call](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/models/vqvae.py:1659) stops gradients through one edge. The decoder still reads the values and learns to use them. Style commitment and codebook EMA updates happen upstream, and shared encoder features can change through other losses. Consequently, `detach_style_injection=true` cannot certify that style lacks anatomy. The supplied gradient audit reports no active style-alignment/independence contribution; it provides no evidence of a complementary active constraint there.

4. **The content loss and the decoder see different stages.** Contrastive pooling uses encoder features saved before split normalization, while decoding uses normalized, projected, quantized features ([encoder taps](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/models/vqvae.py:1383)). A signal available upstream may be harder to recover downstream. This is a candidate failure point to measure, not a demonstrated normalization or VQ bug. Also, switching to the current `film` option while keeping spatial style does not reproduce global AdaIN/FiLM: the local implementation explicitly supports spatially varying modulation.

Several comments overstate what these mechanisms enforce. Cross reconstruction need not remove modality information from content if the decoder ignores it; spatial bottlenecking encourages a separation but cannot guarantee it. These claims should not be treated as invariants established by the existing reconstruction tests.

**Recommended sequence before tuning weights**

1. For a matched baseline, explicitly keep `cross_recon_style_source=same_subject`. Do not silently include the newer donor-target mismatch in the capacity comparison. Preserve the older settings/checkpoint provenance.
2. Run the existing frozen stage and acquisition audits on the training machine:

```bash
python -m eval.content_path_probe \
  --run-dir results/synthetic/synthetic-clean-content-causal-cross-recon \
  --num-samples 512 --batch-size 8 --grids 1 8 --causal iid

python -m eval.style_path_audit \
  --run-dir results/synthetic/synthetic-clean-content-causal-cross-recon \
  --num-samples 256 --swap-samples 64 --batch-size 4 --causal iid
```

The first locates changes in readout across pre-normalization, post-normalization, projection and decoder input. The second tests whether style actually retains and controls legitimate gain/bias, and whether those changes leak into content. IID testing reduces factor-correlation shortcuts; it can differ from the training distribution, so follow with `--causal match` when distribution sensitivity matters. A probe-score drop does not alone establish information destruction.

3. If the saved style spatial size is 0 or greater than 1, train a matched run with `style_spatial_size=1`, preserving the other weights, quantization and detach settings. Keep dropout at 0 for this comparison. Compare matched training steps and seeds. Evaluate both ventricular routing and acquisition routing: moving anatomy into content while destroying acquisition information would not achieve the intended decomposition. This is the first architectural ablation, not a guaranteed final solution.
4. For synthetic data, a stronger correctly specified experiment is to render recipient anatomy with donor acquisition and train against that mixed image, including appropriate mask and normalization. Noise realization and acquisition parameters must be handled deliberately. Alternatively, the existing [same-acquisition pair constructor](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/data/datasets.py:918) changes anatomy while reusing the anchor's actual acquisition settings. Its [within-modality style alignment](/Users/nataliaglazman/Desktop/PhD/projects/multiview-crl/training/style_alignment.py:9) is compatible with natural between-person style variation because it constructs equal-style pairs; it does not assume unrelated people naturally match. It uses additional simulator-controlled correspondence and should be described as such.
5. For data without those render controls, MUNIT/DRIT-style latent reconstruction or cross-cycle consistency is a defensible next design to evaluate. These are additional constraints, not a proof of anatomical identifiability. Keep the anatomical intervention tests as the deciding evidence.

The reconstruction gradient is larger than the content gradient in the supplied report (roughly 17.3 versus 9.0 RMS), but the scalar content loss being around 201 is not evidence that content is weighted too strongly. First correct target semantics and test style capacity; then tune weights using both pathway fidelity and reconstruction quality. No defensible exact new loss weights follow from the supplied measurements alone.
