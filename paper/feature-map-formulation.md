# Scalar latent factors represented by spatial feature maps

Working mathematical formulation, 21 September 2026. This note proposes an extension of the representation space; it does not claim that the current training implementation satisfies an identifiability theorem. The existing methods draft is not modified.

Source: Yao et al., *Multi-View Causal Representation Learning with Partial Observability*, supplied PDF `2311.04056v2 (7).pdf`. Relevant statements are Assumption 2.1 and Definition 2.3 (p. 3), Theorem 3.2 (p. 4), Theorem 3.8 (p. 5), and Lemmas C.2–C.3 and the proof of Theorem 3.2 (pp. 20–22).

## 1. Keep the latent variables scalar; make their representation spatial

Let a subject have shared scalar latent factors

\[
c=(c_1,\ldots,c_d)\in\mathcal C\subset\mathbb R^d
\]

and view-specific variables \(s_k\in\mathcal S_k\). On a finite image grid \(\mathcal V\), write

\[
X_k=f_k(c,s_k)\in\mathbb R^{q_k\times|\mathcal V|},
\qquad
X_k(v)=f_k(c,s_k)(v).
\]

Here \(v\) indexes where an image is evaluated. It does not index a new independent realization of the subject's latent factors. For example, one scalar atrophy parameter can alter many voxels through the rendering function.

For an encoder grid \(\Omega\), define the feature space and encoder by

\[
\mathcal F=(\mathbb R^m)^\Omega\cong\mathbb R^{m|\Omega|},
\qquad E_k:\mathcal X_k\longrightarrow\mathcal F,
\qquad F_k=E_k(X_k).
\]

The indices have distinct meanings:

- \(j\): ground-truth scalar factor;
- \(a\): learned feature channel;
- \(u\in\Omega\): spatial location;
- \(k\): observed view.

In general there is no one-to-one correspondence between \(j\) and \(a\).

## 2. The appropriate global identification target

**Definition (identification of a latent block by a feature tensor).** The content feature tensor identifies \(c\) if

\[
\boxed{F_k=\Phi_k(c),\qquad D_k(\Phi_k(c))=c,}
\]

where \(\Phi_k:\mathcal C\to\mathcal F\) is a smooth embedding and \(D_k\) is its smooth inverse on the image \(\mathcal M_k=\Phi_k(\mathcal C)\).

The first equality excludes functional dependence on \(s_k\). The second preserves all of \(c\). This does not require statistical independence of \(F_k\) and \(s_k\), since \(c\) and \(s_k\) may be statistically dependent.

With exact feature alignment in common channel and spatial coordinates, the \(\Phi_k\) agree, so one can write a common \(\Phi\). With separate coordinate conventions, retain \(\Phi_k\).

Equivalently,

\[
\operatorname{vec}(F_k)=\widetilde\Phi_k(c)\in\mathbb R^M,
\qquad M=m|\Omega|.
\]

For \(M>d\), this is a bijection between \(\mathcal C\) and a \(d\)-dimensional embedded manifold of tensors, not a bijection onto the whole ambient space \(\mathbb R^M\). A necessary dimension condition is \(M\ge d\); it is not sufficient for identification.

This is the natural overcomplete analogue of Definition 2.3. Reshaping a vector of dimension \(d\) into a tensor with exactly \(d\) entries changes only notation. A practical tensor with many more than \(d\) entries requires the embedded-manifold formulation above.

## 3. Concrete example: two scalar factors and two feature maps

Let \(c_1\) and \(c_2\) denote two anatomical scalar parameters. Let \(b_1,b_2:\Omega\to\mathbb R\) be fixed, nonzero spatial patterns associated with two regions. Consider

\[
F_1(u)=c_1b_1(u)+c_2b_2(u),\qquad
F_2(u)=c_1b_1(u)-c_2b_2(u).
\]

The subscripts here indicate channels, not imaging views. Then

\[
c_1=\frac{\langle F_1+F_2,b_1\rangle}{2\|b_1\|^2},
\qquad
c_2=\frac{\langle F_1-F_2,b_2\rangle}{2\|b_2\|^2}.
\]

Thus the tensor identifies the two-dimensional block. Neither channel isolates a single factor. At a location where both patterns vanish, neither factor can be recovered from that location, yet the full tensor remains sufficient.

This is an illustrative encoding, not an assumption of additive MRI physics or a guarantee about learned channels.

## 4. What transfers from the paper's alignment argument

For registered views, use whole-field alignment

\[
\mathcal L_{\mathrm{align}}^{\mathrm{field}}
=\sum_{k<\ell}\mathbb E\left[
\frac1{|\Omega|}\sum_{u\in\Omega}
\|E_k(X_k)(u)-E_\ell(X_\ell)(u)\|_2^2
\right].
\]

Under the paper's smoothness and full-support assumptions, zero loss implies

\[
E_k(f_k(c,s_k))=\Phi(c).
\]

For two views, the argument is particularly transparent: hold \(c,s_2\) fixed and vary \(s_1\). The second encoder's output cannot change. Exact alignment therefore forces the first output to remain unchanged. Repeat with \(s_2\). Full support and continuity turn almost-sure alignment into the needed functional statement. Independence of the styles is not required, but deterministic constraints that prevent their variation can invalidate this reasoning.

Apply this argument to every coordinate \((a,u)\) of the tensor. This is the dimension-independent content-exclusion argument of Lemma C.3; it does not use a one-channel-per-factor assumption.

**It proves exclusion, not preservation.** A constant field and a field encoding only a subset of \(c\) both satisfy alignment. Joint image reconstruction is also insufficient if a separate style path can carry the missing content.

## 5. A precise route from Theorem 3.2 to feature tensors

The following is a proposed corollary using the paper's theorem as an input, not a theorem stated in the paper.

Let \(P:\mathcal F\to(0,1)^d\) be a common smooth readout, and set

\[
g_k=P\circ E_k.
\]

**Proposition.** Suppose:

1. The generative model and composed encoders satisfy the applicable conditions of Yao et al.'s Theorem 3.2, with the correct shared-content dimension \(d\).
2. The full feature tensors align almost surely across views.
3. The composed encoders \(g_k\) attain the global minimum in Theorem 3.2.

Then the full tensor identifies the shared latent block in the sense of Section 2.

**Proof.** Full alignment gives \(E_k(X_k)=\Phi(c)\) by the exclusion argument. Theorem 3.2 gives

\[
P(\Phi(c))=h(c)
\]

for a smooth invertible \(h\). Consequently,

\[
D(F)=h^{-1}(P(F)),\qquad D(\Phi(c))=c.
\]

In particular, \(\Phi(c)=\Phi(c')\) implies \(h(c)=h(c')\), hence \(c=c'\). The smooth left inverse also implies full differential rank: \(J_DJ_\Phi=I_d\). The restriction of \(D\) to the image gives the inverse, so \(\Phi\) is a smooth embedding. \(\square\)

One idealized objective realizing these conditions is

\[
\mathcal L_{\mathrm{ideal}}
=\mathcal L_{\mathrm{align}}^{\mathrm{field}}
-\lambda\sum_k H(P(E_k(X_k))),\qquad\lambda>0.
\]

If a zero-valued solution is attainable in the chosen model/readout class and a population global minimizer is reached, every term is minimized: the fields align and the readouts are uniform on the unit cube. Thus \(P\circ E_k\) also minimizes the paper's objective, and the proposition applies. Attainability must be checked; it is not automatic for an arbitrary architecture or readout.

This is a clean theoretical construction, not a claim about finite-sample InfoNCE, Barlow Twins, VICReg, or the current implementation. A broadcast encoding \(F(u)=h(c)\) can satisfy these requirements when the readout permits it. Therefore this construction establishes global content identification without establishing anatomical localization.

### Why readout alignment alone is weaker

For a mean readout, take

\[
F_k(u)=h(c)+s_k b(u),\qquad \sum_u b(u)=0.
\]

Then \(\operatorname{mean}_u F_k(u)=h(c)\), while the maps still contain style. This example illustrates the gap between a theorem about a pooled/projected representation and a theorem about the complete tensor.

The readout \(P\) need not be global average pooling. Pooling can discard useful spatial patterns, including zero-mean ones; a theorem demanding a sufficient pooled readout may impose a storage format that is undesirable for local anatomy.

## 6. Why entropy and channel counting need care

If \(F=\Phi(c)\) is a smooth embedding and \(M>d\), the feature distribution is supported on a lower-dimensional manifold. It has no density with respect to \(M\)-dimensional Lebesgue measure. Therefore one cannot directly maximize ambient differential entropy to obtain a uniform distribution on \((0,1)^M\) as in Theorem 3.2. Manifold entropy would require a specified reference measure and a separate argument; inserting an entropy term does not itself establish injectivity.

The entropy of \(P(F)\in(0,1)^d\) above avoids this dimensional mismatch. The value \(d\) must include all shared latent degrees of freedom, including independently sampled deformation fields when present. The number of labelled scalar targets need not equal \(d\).

Variation across spatial sites is also different from variation across subjects. A subject-independent positional template \(F_n(u)=T(u)\) can have substantial variance when \((n,u)\) are pooled, while carrying no subject information. Nonzero variance or channel decorrelation does not establish content sufficiency.

For a channel selector \(\varphi\in\{0,1\}^m\), the feature-map analogue is

\[
(\varphi\odot F)(a,u)=\varphi_aF(a,u),
\]

or equivalently retaining the selected channels. This is a valid architectural operation. However, \(\|\varphi\|_0\) counts maps, not scalar degrees of freedom. One map can encode several scalars, and several maps can redundantly encode one. Thus the dimension-counting argument behind Theorem 3.8's selector regularizer does not transfer merely by replacing vector entries with feature maps.

## 7. Adding a spatial claim requires an additional target

The general representation is

\[
F(u)=\Phi(c)(u)=\phi_u(c).
\]

Only the assembled field must be injective in \(c\). Individual \(\phi_u\) need not be injective. Arbitrary invertible rearrangements of the full field preserve the global property, so global identification does not identify anatomical support.

To describe spatial sensitivity of a scalar, define

\[
J_j(u;c)=\frac{\partial\Phi(c)(u)}{\partial c_j}\in\mathbb R^m.
\]

This shows where the representation changes when that scalar changes, holding the other latent coordinates fixed. It is a coordinate sensitivity, not automatically the total causal effect of an intervention when latent factors have causal descendants. Without known factors or further identification, individual columns of this Jacobian cannot be given unique biological labels from learned channels alone.

A locality assumption can limit where dependence occurs. If the feature at \(u\) uses only an input neighborhood \(N(u)\), then

\[
\frac{\partial f_k(c,s_k)(v)}{\partial c_j}=0
\text{ for every }v\in N(u)
\quad\Longrightarrow\quad
\frac{\partial E_k(f_k(c,s_k))(u)}{\partial c_j}=0.
\]

This follows by the chain rule for a strictly local encoder. It provides a support-inclusion statement. It does not provide equality, guarantee retention of all locally available information, or establish a universal lower bound on localization accuracy. Normalization using global spatial statistics or global attention can invalidate strict locality even when convolution kernels are small.

### Optional stronger local formulation

Specify a local anatomical state \(a_u(c)\in\mathcal A_u\), such as an explicitly defined local geometric descriptor or a vector of truly local latent variables. Then a stronger target is

\[
F(u)=h_u(a_u(c)),\qquad h_u\text{ a smooth embedding}.
\]

It preserves the specified local state at that location. For recovery of all global scalars from the whole map, additionally require \(c\mapsto(a_u(c))_{u\in\Omega}\) to be injective. A common map \(h_u=h_0\) is a further assumption, requiring justification from the local generative mechanism and architecture; convolutional weight sharing alone does not establish it.

If \(a_u(c)\) is a deterministic spatial expression of scalar factors, its entries are generally dependent and may occupy a lower-dimensional set. They must not silently be treated as arbitrary independent latent variables with a positive density on the full field space.

Alternatively, genuinely random fields can be included among the generative latents. On a finite lattice these can be vectorized and their true intrinsic dimensions counted, but this changes the generative model and its assumptions.

## 8. Corrections needed before using the existing pointwise draft

The existing `paper/methods-formulation.tex` contains useful targets but should not yet be used as a proved feature-map extension:

1. Replacing invertibility by injectivity does not permit a smooth embedding of an open \((d_c+n_g)\)-dimensional local latent space into \(m<d_c+n_g\) channels. A lower-dimensional manifold assumption must be explicit and exact for an exact claim. Approximate smoothness or a Taylor approximation does not establish it.
2. Recovering every global factor \(\gamma\) at every site requires local observability of those factors. Whole-map identification does not supply that assumption.
3. Subtracting a positional mean does not establish a stationary local generative mechanism. For example, \(X_n(u)=a(u)c_n+b(u)\) becomes \(a(u)(c_n-\mathbb E c)\) after centering; the position-dependent sensitivity remains.
4. A local common map and support-recovery equality require their own assumptions. A finite receptive field alone gives an upper limit on possible dependency spread, not an unavoidable minimum localization width.
5. Agreement between two encoders, even up to one common per-channel scale, is a cross-view statement. It does not by itself establish an injective relationship to the true content block.

## 9. Suggested paper wording

> We retain a finite-dimensional latent model in which shared scalar factors generate spatially structured observations. Our encoders represent these factors by feature fields on a common spatial grid. We seek a content field that depends only on the shared latent block and from which that block is recoverable. For an overcomplete feature tensor, this corresponds to a smooth embedding of the content space into the space of feature fields, with an inverse defined on its image. Individual channels and locations need not identify individual scalar factors. Anatomical localization constitutes an additional structural requirement beyond global block-identifiability.

The associated evaluation questions are distinct: does the whole content field exclude view-specific variation; does it preserve the shared factors; and where is the retained anatomical information accessible? The first two correspond to the global identification target. The third requires the additional spatial formulation and evidence.
