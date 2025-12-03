CSCI-653 High Performance Computing -- Fish schooling in complex geometries 

Participants: Hao Cheng
## 0. Problem description:
Fish typically congregate into cohesive groups to explore and navigate environments safely and more efficiently, exhibiting various collective phases such as polarized schooling, milling\cite{gautrais2012deciphering}, and tunning\cite{filella2018model}, and swarming when there is low environment information. In recent years, lots of fish schooling models have been proposed to reveal the mechanisms behind these fascinating phenomena, while most focus on fish schooling in unbounded domains. However, to understand how fish groups behave under complex geometry confinement, for example, in complex undersea tunnels and caves, it is important to formulate models to simulate fish schooling dynamics in confined domains. We use high-order boundary integral equations to achieve non-penetration boundary conditions for potential dipole fields and investigate how fish schools behave under bi-chamber confinement, with two chambers connected by a narrow channel. We examine the population oscillation between the two chambers and the effects of group size and geometry shape, and we find a surprising interplay between group size and transition dynamics.

## 1. Simulation methods:
The dynamics of fish schooling can be modeled as active particles, Filella \emph{et al.}\cite{filella2018model} proposed a novel model incorporating hydrodynamic interactions with behavioral rules. The model demonstrated how flow features influence collective behavior, revealing new phases such as "collective turning" and enhanced swimming efficiency. Hydrodynamic effects, often ignored in traditional models, were shown to introduce self-induced noise and contribute to emergent behaviors like schooling and milling. Gautrais \emph{et al.}\cite{gautrais2012deciphering} employed a bottom-up methodology to infer interaction rules directly from empirical data. Their findings highlighted the role of positional and orientational effects in fish turning speeds, emphasizing that collective properties emerge from density-dependent behavioral changes. This approach provided a robust framework for linking individual-level interactions to group dynamics, particularly in confined environments where group size alters social interactions. Calovi \emph{et al.}\cite{Cavaiola2021} investigated fish schools' responses to external perturbations using a validated data-driven model. Their study demonstrated that responsiveness peaks in transition regions between schooling and milling states, where fluctuations in alignment and milling order parameters are highest. Perturbations were shown to propagate effectively in states exhibiting multistability, highlighting the intrinsic link between response sensitivity and phase transitions in collective dynamics. Huang \emph{et al.}\cite{Huang2017} explored the effects of confinement on collective behavior in fish schools. Incorporating geometric boundaries and hydrodynamic interactions, their model uncovered new collective phases, such as "double milling" and "wall-following." These emergent patterns demonstrated that confinement can drive spontaneous transitions between schooling and milling without individual-level adjustments. The study underscored the role of environmental factors in shaping collective behaviors, offering insights into the dynamics of confined biological systems.Tsang and Kanso \cite{Tsang2015} investigate the collective behavior of microswimmers in circularly confined domains, demonstrating transitions between chaotic swirling, stable circulation, and boundary aggregation based on flagellar activity. This work bridges the gap between geometric confinement and emergent hydrodynamic patterns, underscoring the influence of boundary effects on microscale systems.

Euler–Maruyama method to get a numerical solution of the Stochastic differential equations. Since stochasticity is involved in the swimmer's equations of motion, we will run multiple Monte-Carlo simulations (MCs). The MCs will be run using distributed computing through the OpenMPI platform on CARC. Data visualisation and analysis might include the heat map of the spatial probability density distribution, correlation in the velocity of focal fish, etc. All the codes that are used in this project, including some existing code, are in Python and MATLAB.

\paragraph{Equation of Fish motion}
Considering a system of $N$ swimmers, where each fish is represented by a self-propelled particle that moving along its heading direction at speed $U =0.2$ (units: $\mathrm{ms^{-1}}$) relative to the flow velocity\cite{Gautrais2012}. Each swimmer generates a flow field that is represented by far-field potential dipole~\cite{Kanso2014,Filella2018} and behaves according to behavioral rules derived from shallow water experiments in a circular tank~\cite{Gautrais2012,Gautrais2009}. Therefore, each swimmer translates under the influence of local flow disturbances generated by all the other swimmers and reorients its heading to both approach and align with its Voronoi neighbors, reflecting the attraction and alignment behaviors~\cite{Gautrais2012,Filella2018}. Let fish $n$ be positioned at ${\mathbf{r}}_n = (x_n,y_n)$ , moving with translational velocity, denoted by $\dot {\mathbf{r}}_n$, where the dot stands for time derivative. The heading direction is expressed by a unit vector $\mathbf{p}_n = (\mathrm{cos}\;\theta_n, \mathrm{sin}\;\theta_n)$, where $\theta_n$ is the heading angle measured from $x$-axis ranging from $-\pi \; \text{to} \; \pi$. Based on the prescribed system, the equations of motion of fish can be expressed dimensionally as follow:

\begin{equation}
\begin{aligned}
    \dot {\mathbf{r}}_n& =  U\mathbf{p}_n + \mathbf{U}_n \\
    \mathrm{d}{\theta_n} &=  ((\mathbf{p}_n \cdot \nabla)\mathbf{U}_n\cdot\mathbf{p}_n^\perp
+k_w \frac{\text{sgn}(\phi_{wn})}{d_{wn}}) \mathrm{d}t 
+ \langle k_{p} d_{nk} \sin\theta_{nk} + k_{v} \sin{\phi_{nk}}\rangle \mathrm{d}t +\sigma\, dW.
\end{aligned} \label{eqn_SI:motion}
\end{equation}
where $\mathbf{U}_n = (\mathbf{u}+\mathbf{u}_{\text{BEM}})\bigl\rvert_{\mathbf{r}_n}$ is the hydrodynamic disturbance coupled both dipolar field and flow field generated by boundary singularities.
The wall-avoidance intensity $k_w$  (units: $\mathrm{m}\mathrm{s}^{-1}$) scales with a reorienting process to avoid collision with the wall. Here, $\phi_{wn}$ is the angle between fish \(n\)’s heading direction and the outward normal at the point of impact on the boundary, and $d_{wn}$ is the distance from the fish to that point of impact (Fig S.~\ref{fig_SI:geo}A). A standard Wiener process $\mathrm{d}W = \sqrt{\mathrm{d}t}\mathcal{N}(0, 1)$ characterized by standard deviation $\sigma$ (units: $\text{rad} \cdot \mathrm{s^{-1/2}}$) is employed to model the fish's ``free will”.

The behavioral rules specify a tendency to align with and be attracted to its Voronoi neighbors $\mathcal{V}_n$ by reorienting its heading angle, scaled with alignment intensity $k_v$ (units: $\mathrm{m^{-1}}$) and attraction $k_p$ (units: $\mathrm{m^{-1}s^{-1}}$). The operator $\langle \circ \rangle $ represents a weighted average using $(1+\text{cos}(\theta_{nk}))$, modelling continuously a rear blind angle~\cite{Calovi2014}.

%------
\begin{equation}
\begin{aligned}
\langle\circ\rangle=\sum_{j \in \mathcal{V}_n} \circ\left(1+\cos \theta_{nk}\right) / \sum_{j \in \mathcal{V}_n}\left(1+\cos \theta_{nk}\right).
\end{aligned}
\label{eqn_SI:blind_angle}
\end{equation}
Furthermore, $\theta_{nk} = \arctan2 \left(y_k-y_n,x_k-x_n \right)-\theta_n$, $\phi_{nk} = \theta_k-\theta_n$ and $d_{nk} =\| \mathbf{r}_n - \mathbf{r}_k\|$ are the viewing angle, relative alignment angle and inter-swimmer distance (see Fig.~1 \cite{Filella2018}), respectively. 
The flow disturbance generated by all other swimmers on the focal swimmer's location $\mathbf{r}_n$ is represented by $\mathbf{u}\bigl\rvert_{\mathbf{r}_n}$, it is superposed under potential flow approximation. We can write the flow interactions in a  pairwise form 
\begin{equation}
 \mathbf{u}\bigl\rvert_{\mathbf{r}_n} = \sum_{k \neq n} \mathbf{u}_{kn}, \quad \text{where} \quad \mathbf{u}_{kn} = \frac{k_f}{\pi} \frac{ \mathbf{p}_n \sin 2\theta_{kn} + \mathbf{p}_{n}^{\perp} \cos 2\theta_{kn}}{ d_{nk}^2},
\end{equation}
where the intensity $k_f$ (units: $\mathrm{m}^3 \mathrm{s}^{-1}$) of the dipolar field produced by the swimmers is defined as $Sv$, and it depends on both the swimming speed and the swim surface cross-sectional area, $S = \pi l^2/4$, with $l$ representing the fish's body length \cite{Gautrais2012,Filella2018}. 
For convenience, the dipolar field is rewritten in complex notation. If $N$ swimmers are present at positions $z_n = x_n+iy_n$ in the complex plane with orientations $\theta_n$, for $n = 1, \dots, N$, the complex potential $F(z)$ is formulated by linearly superimposing the dipolar flow fields emerging from each swimmer at its position $z_n$ to form
\begin{equation}
    F(z) = - \sum_{n=1}^{N} \frac{k_f e^{i \theta_n}}{\pi (z - z_n)}, \quad 
       w(z) = \frac{\partial F(z)}{\partial z}=\sum_{n=1}^{N} \frac{k_f e^{i \theta_n}}{\pi (z - z_n)^2} = u_x(z)-iu_y(z) \label{eqn_SI:dip_unreg}
\end{equation}
As every dipole induces a repulsive velocity affecting all other swimmers, it diminishes with $1/d^2$ and becomes singular as $d \rightarrow 0$. While it is rare for two swimmers to occupy precisely the exact location computationally, the significant repulsive force that occurs when swimmers approach one another or the boundary of confinement can cause the swimmers to be repelled over a considerable distance.
Following Huang~\emph{et al.}\cite{Huang2024}, the point-dipole model is regularized by adding a regularization parameter $\delta$, then we can rewrite \ref{eqn_SI:dip_unreg} into 
\begin{equation}
     w^{\delta}(z) = \sum_{n=1}^{N} \frac{k_f e^{i \theta_n}}{\pi \left[(z - z_n)^2 +\delta^2\right]} 
\end{equation}


 ## 2. Expected results:
 We expect that there is an optimal strategy for the leader fish to guide the entire fish school to the destination much more efficiently than the linear controller, which acts as a comparison for all the RL results. And through this, we can study the basic mechanism of why it is efficient through the information propagation speed, etc.
