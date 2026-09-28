# Parameters (`Param.py`)

The following scripts are all constants, conventions, and fitted values used to reconstruct the strain-dependent tight-binding Hamiltonian for transition-metal dichalcogenides (TMDCs) as described in [Fang et al. (2018), Sec. II C](references.md#fang2018). We focused on $MoS_2$ as an generalizable example for all other TMDCs. 

## Orbital Groups
<a id="orbital-Groups"></a>

As in the paper, we take the $MoS_2$ layer to lie in the $xy$-plane. So the hamiltonian $\hat{H}$ will have the $xy$ mirror symmetry, i.e. $\hat{H}$ commutes with the reflection operator $\hat{r}: (x, y, z) \to (x, y, -z)$. And so the eigenstates $\left|n\right>$ of $\hat{H}$ are also the eigenstates of $\hat{r}$. A key observation is that **orbital states $\left|\phi\right>$ are also eigenstates of $\hat{r}$**, since $\hat{r}\left|\phi\right> = \pm \left|\phi\right>$. So eigenstates $\left|n\right>$ are restricted to linear combinations of only the even orbitals $(\hat{r}\left|\phi\right> = \left|\phi\right>)$ or linear combinations of only the odd orbitals $(\hat{r}\left|\phi\right> = -\left|\phi\right>)$. Using this understanding, the 11 orbitals of a single $MoS_2$ (1 unitcell) that form the [relavant bands](references.md#fang2018-bands) near the band gap are classified based on their even/odd parity (denote as $o/e$ superscript).

$$
\begin{align*}
\Psi_A &= \begin{bmatrix} \ d_{xz}^o \\ d_{yz}^o \end{bmatrix} &
\Psi_B &= \begin{bmatrix} \ p_{x}^o \\ p_{y}^o \\ \ p_{z}^o \ \end{bmatrix} &
\Psi_C &= \begin{bmatrix} \ d_{xy}^e \\ d_{x^2-y^2}^e \\ d_{z^2}^e \ \end{bmatrix} &
\Psi_D &= \begin{bmatrix} \ p_{x}^e \\ p_{y}^e \\ \ p_{z}^e \ \end{bmatrix}
\end{align*}
$$

Following their convension, we also introduced those groups in our script:
```python
--8<-- "Library/Param.py:group-size"
```
!!! Note "By the way"
    Note here that $d$ orbitals are centered on $Mo$ so $\hat{r}d_{xz} \sim x(-z) = -d_{xz}$. But $p$ orbitals here are in fact superpositions of the two $p$ orbitals in top and bottom $S$, so $p_z^e \sim (p_z^{\text{top}} - p_z^{\text{bot}})$, implying 


    $$\hat{r}p_z^e \sim \left(\hat{r}p_z^{\text{top}} - \hat{r}p_z^{\text{bot}}\right) = \left((-p_z^{\text{bot}}) - (-p_z^{\text{top}})\right) \sim p_z^e.$$

    Similarly $p_z^o \sim (p_z^{\text{top}} + p_z^{\text{bot}})$. One should be able to imagine superpositions of other $p$ orbitals (left as an exercise for the reader ☺). 

## Valid Neighbors

Shown here are the 11 orbital by 11 orbital submatrix between two unitcells. Note that $[0]$ are used to represent zero matrix.

* $H^{(0)}$: the onsite term where the the two unitcells are the same. 
* $H^{(1)}$: the hopping term from $Mo$ to nearest $S_2$. So sometimes the two unitcell are the same, sometimes not.
* $H^{(2)}$: the hopping term from $Mo$ to another $Mo$ or from $S_2$ to another $S_2$. So the two unitcell aren't the same. 
* $H^{(3)}$: the hopping term from $Mo$ to the next-nearest $S_2$ position. So the two unitcell aren't the same. 

<figure markdown="block">

$$
H^{(0)} = \begin{bmatrix} 
              H_{AA}^{(0)} & [0] & [0] & [0] \\ 
              [0] & H_{BB}^{(0)} & [0] & [0] \\ 
              [0] & [0] & H_{CC}^{(0)} & [0] \\ 
              [0] & [0] & [0] & H_{DD}^{(0)} \ 
          \end{bmatrix}
\quad,\quad
H^{(1)} = \begin{bmatrix} 
              [0] & H_{BA}^{(1)} & [0] & [0] \\ 
              [0] & [0] & [0] & [0] \\ 
              [0] & [0] & [0] & H_{DC}^{(1)} \\ 
              [0] & [0] & [0] & [0]
          \end{bmatrix}
\quad,\quad
H^{(2)} = \begin{bmatrix} 
              H_{AA}^{(2)} & [0] & [0] & [0] \\ 
              [0] & H_{BB}^{(2)} & [0] & [0] \\ 
              [0] & [0] & H_{CC}^{(2)} & [0] \\ 
              [0] & [0] & [0] & H_{DD}^{(2)} \ 
          \end{bmatrix}
\quad,\quad
H^{(3)} = \begin{bmatrix} 
              \ [0] & [0] & [0] & [0] \\ 
              [0] & [0] & [0] & [0] \\ 
              [0] & [0] & [0] & H_{DC}^{(3)} \\ 
              [0] & [0] & [0] & [0] \ 
          \end{bmatrix}
$$

<figcaption markdown="span">
Note here that $H_{BA}^{(3)}$ is neglected on purpose, as it is too small (for some reason, maybe the author calculated and think it was too small).
</figcaption>
</figure>

In our script, `VALID` is thus used to restrict the nonzero entries of $H^{(i)}$ to only those location.
```python
--8<-- "Library/Param.py:valid-neighbors"
```

!!! Note "By the way"
    Remember the Hermitian requirement is only for the full hamiltonian $\hat{H}$, so such requirement need not apply to individual $H^{(i)}$. And this is why $H^{(1)}$ only recorded the hopping $Mo \to S_2$, as the reverse direction $S_2 \to Mo$ will be included from the conjugate requirement of $H$. Coding wise, this means $(H^{(1)})^{\dagger}$ will be added to the transpose position (i.e. reflected across $H$'s diagonal) of where $H^{(1)}$ lies in $H$. 


## Fitted Entries of Orbital Couplings
Let $\alpha, \beta$ be elements of $\{A, B, C, D\}$, and $i \in \{0, 1, 2, 3\}$. Then we can write $H_{\alpha, \beta}^{(i)}$ as one of the submatrix entries in $H^{(i)}$. The individual entries of $H_{\alpha, \beta}^{(i)}$ will be numbers that depends of the strain of the system. 

To find those numbers, the paper used some density field theory + localized Wannier functions black magic ([Fang et al. (2018), APPENDIX A](references.md#fang2018-DFT+Wannier)). I am not entirely clear on how they did it, but since they provided those numbers, I just happily copied as shown in the script below. 

<div class="expandable-code" data-lines="5" markdown="1">
```python
--8<-- "Library/Param.py:fitted-values"
```
</div>

Note here `np.zeros((3,4,9), dtype=float)` is following the convention that the needed parameters has 

* 3 kinds of $C_3$ invariant strain: unstrained ($1$), the scalar strain ($u_{xx}+u_{yy}$) which is a trace over the strain tensor, the plane span by differential and shear strain ($u_{xx}-u_{yy}$ and $2u_{xy}$). (Further information look into character table for irreducible representations of $C_3$ group restricted to $\mathbb{R}$ matrix entries.)
* 4 kinds of neighbors: $i \in \{0, 1, 2, 3\}$
* 9 entries for a 3x3 matrix. 
