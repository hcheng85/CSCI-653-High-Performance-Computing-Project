# CSCI-653 High Performance Computing — Fish Schooling in Complex Geometries

**Participants:** Hao Cheng

---

## 0. Problem Description

Fish typically congregate into cohesive groups to explore and navigate environments safely and efficiently, exhibiting collective phases such as polarized schooling, milling [@gautrais2012deciphering], turning [@filella2018model], and swarming. Most existing models focus on unbounded domains. However, many natural environments contain complex confinement, such as undersea tunnels and caves. Understanding how fish behave under geometric constraints requires new simulation frameworks.

In this project, we use high-order boundary integral equations to impose non-penetration boundary conditions for potential dipole flow fields and investigate fish-schooling dynamics inside a bi-chamber geometry consisting of two chambers connected by a narrow channel. We examine population oscillations between chambers, the influence of group size, geometric effects on transitions, and the interplay between hydrodynamic interactions and confinement.

---

## 1. Simulation Methods

Fish are modeled as active particles. Filella *et al.* [@filella2018model] demonstrated how hydrodynamic interactions and behavioral rules generate phases such as collective turning and milling. Gautrais *et al.* [@gautrais2012deciphering] extracted interaction rules directly from experimental data, showing how positional and orientational cues determine turning behavior. Calovi *et al.* [@Cavaiola2021] studied responses to external perturbations, finding sensitivity peaks near transitions between schooling and milling. Huang *et al.* [@Huang2017] showed that confinement generates new behaviors such as double milling and wall-following. Tsang and Kanso [@Tsang2015] investigated confined microswimmers and demonstrated transitions between chaotic swirling, stable circulation, and boundary aggregation.

We use the Euler–Maruyama method to simulate the stochastic differential equations. Since noise is present in the turning dynamics, we run multiple Monte Carlo (MC) simulations distributed over OpenMPI on USC CARC. Data analysis includes spatial probability density heat maps, velocity-correlation statistics, and flow visualization. All code (existing and new) is written in Python and MATLAB.

---

## Equation of Fish Motion

Let each fish \( n \) be at position  
\[
\mathbf{r}_n = (x_n, y_n),
\qquad
\mathbf{p}_n = (\cos\theta_n,\, \sin\theta_n),
\]
and swim at speed \( U = 0.2\ \mathrm{m/s} \) relative to the flow.

### Equations of motion

$$
\begin{aligned}
\dot{\mathbf{r}}_n &= U\,\mathbf{p}_n + \mathbf{U}_n, \\
\mathrm{d}\theta_n &= 
\Big[
(\mathbf{p}_n \cdot \nabla)\mathbf{U}_n \cdot \mathbf{p}_n^\perp
+ k_w \frac{\operatorname{sgn}(\phi_{wn})}{d_{wn}}
\Big]\,\mathrm{d}t \\
&\quad
+ \left\langle
k_p\, d_{nk}\,\sin\theta_{nk} +
k_v\,\sin\phi_{nk}
\right\rangle\,\mathrm{d}t
+ \sigma\,\mathrm{d}W.
\end{aligned}
$$

Here:

- \( \mathbf{U}_n = (\mathbf{u} + \mathbf{u}_{\mathrm{BEM}})|_{\mathbf{r}_n} \) is the combined disturbance from hydrodynamic dipoles and boundary singularities.
- \( k_w \) controls wall avoidance; \( d_{wn} \) is the distance to the boundary hit point; \( \phi_{wn} \) is the incident angle.
- \( \mathrm{d}W = \sqrt{\mathrm{d}t}\,\mathcal{N}(0,1) \) is a Wiener process.
- \( k_p \) and \( k_v \) scale attraction and alignment to Voronoi neighbors.

### Voronoi-based behavioral operator

$$
\langle \circ \rangle
= \frac{
\sum_{j\in\mathcal{V}_n} \circ\,(1+\cos\theta_{nk})
}{
\sum_{j\in\mathcal{V}_n} (1+\cos\theta_{nk})
}.
$$

Neighbor definitions:

- \( \theta_{nk} = \operatorname{atan2}(y_k - y_n,\, x_k - x_n) - \theta_n \)
- \( \phi_{nk} = \theta_k - \theta_n \)
- \( d_{nk} = \|\mathbf{r}_n - \mathbf{r}_k\| \)

---

## Hydrodynamic Dipole Interactions

The disturbance induced by fish \( k \) on fish \( n \) is modeled as:

$$
\mathbf{u}_{kn}
= \frac{k_f}{\pi}
\frac{
\mathbf{p}_n \sin 2\theta_{kn}
+ \mathbf{p}_n^\perp \cos 2\theta_{kn}
}{
d_{nk}^2
}.
$$

Here \( k_f = S v \), where \( S = \pi l^2 / 4 \) is the fish’s cross-sectional area and \( l \) is body length.

### Complex representation of dipoles

Using complex notation \( z_n = x_n + i y_n \):

$$
F(z)
= -\sum_{n=1}^N
\frac{k_f e^{i\theta_n}}{\pi (z - z_n)},
\qquad
w(z)
= \frac{\partial F}{\partial z}
= \sum_{n=1}^N
\frac{k_f e^{i\theta_n}}{\pi (z - z_n)^2}.
$$

To avoid singularities when swimmers approach each other, we use the regularized form:

$$
w^\delta(z)
= \sum_{n=1}^N
\frac{k_f e^{i\theta_n}}{
\pi\big[(z - z_n)^2 + \delta^2\big]
}.
$$

---

## 2. Expected Results

We expect that a leader fish can guide the group significantly more efficiently than a linear controller. This framework enables us to study how hydrodynamic interactions, geometry, and group size influence:

- transition dynamics,
- information propagation speed,
- population oscillation between chambers,
- and collective navigation strategies.

These results reveal how confinement and flow-mediated interactions shape collective behavior in complex environments.

