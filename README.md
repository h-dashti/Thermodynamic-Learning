

# Thermodynamic Learning: The EntropyFlow Framework

**Thermodynamic Learning** is an exploratory framework for modeling learning as a **thermodynamic process** rather than a static optimization problem. Inspired by principles from **statistical physics**, this paradigm treats data as particles in a physical system, where structure emerges gradually through the interplay of **energy**, **entropy**, and **temperature**.

Within this broad framework, **EntropyFlow** is a specific implementation that focuses on **unsupervised structure discovery**, demonstrating how learning systems can evolve over time through **temperature-driven dynamics**—exploring freely at high entropy and settling into stable patterns as the system cools.


## The Core Idea: Learning as Thermodynamic Evolution

In thermodynamics, the **Helmholtz free energy** is defined as:

$$
F = U - T S
$$

Where:

* **U**: Internal energy (quantifies how well a configuration fits the system, e.g., squared distance to cluster centers)
* **S**: Entropy (represents uncertainty in assignments)
* **T**: Temperature (controls the trade-off between exploration and exploitation)
* **F**: Free energy (the system’s total cost, balancing fit and flexibility)

**Thermodynamic Learning** treats the learning process as an evolving system, guided by an annealing schedule where:

* High temperature allows for **exploration** and flexibility.
* Low temperature promotes **convergence** and stable structures.



## EntropyFlow: A Clustering Model within Thermodynamic Learning

**EntropyFlow** implements these thermodynamic principles specifically for **clustering and structure discovery** tasks. It models learning as an **annealing process**, where soft, probabilistic assignments emerge based on energy-entropy trade-offs.

### The Equations

#### Soft Assignments (Boltzmann Distribution)

For data point $x_i$ and cluster center $\mu_k$:

$$
E_{ik} = \|x_i - \mu_k\|^2
$$

$$
p_{ik} = \frac{\exp\left(-\frac{E_{ik}}{T}\right)}{\sum_k \exp\left(-\frac{E_{ik}}{T}\right)}
$$

#### Thermodynamic Quantities

At each iteration:

$$
U = \sum_{i,k} p_{ik} E_{ik}
$$

$$
S = -\sum_{i,k} p_{ik} \log p_{ik}
$$

$$
F = U - T S
$$

These quantities evolve over time, revealing how the system transitions from **disorder** to **structure**.



## The Code: EntropyFlow in Action

```python
class EntropyFlow:
    def __init__(self, n_clusters, initial_temp, cooling_rate, min_temp, max_iter):
        self.n_clusters = n_clusters
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter = max_iter
        self.temperature = initial_temp
        self.cluster_centers_ = None
        self.labels_ = None
        self.history = {
            'temperature': [], 'energy': [], 'entropy': [], 'free_energy': [], 'centers': []
        }
```

### Energy Matrix

$$E_{ik} = \|x_i - \mu_k\|^2$$

```python
def calculate_energy_matrix(self, X, centers):
        distances = cdist(X, centers)  # shape (N, K)
        return distances**2
```

$$ p_{ik} = \frac{\exp(-E_{ik}/T)}{Z}, \quad Z = \sum_k \exp(-E_{ik}/T)  $$

```python
def calculate_probabilities(self, energy_matrix):
    exp_term = np.exp(-energy_matrix / temperature)
    Z = np.sum(exp_term, axis=1, keepdims=True)
    return exp_term / Z
```


### Thermodynamic State

$$ 
U = \sum p_{ik} E_{ik}, \quad S = -\sum p_{ik} \log p_{ik}, \quad F = U - T S
$$

```python
def _calculate_thermodynamics(self, X):

    # energy
    energy_matrix = self.calculate_energy_matrix(X, self.cluster_centers_)
    probs = self.calculate_probabilities(energy_matrix)
    energy = np.sum(probs * energy_matrix)

    # entropy
    probs = np.clip(probs, 1e-10, 1.0) # avoid log(0)
    entropy = -np.sum(probs * np.log(probs))

    # free energy
    free_energy = energy - self.temperature * entropy

    return free_energy, energy, entropy, probs
```


### Update Step

$$
\mu_k = \frac{\sum_i p_{ik} x_i}{\sum_i p_{ik}}
$$

```python
def _get_updated_centers(self, X, probabilities):
    weighted_sum = np.dot(probabilities.T, X)
    weights_sum = np.sum(probabilities, axis=0)
    weights_sum = np.maximum(weights_sum, 1e-8) # avoid division by zero
    return weighted_sum / weights_sum[:, np.newaxis]
```


### Fit Loop: Annealing-Based Evolution

```python
def fit(self, X):
    n_samples, n_features = X.shape

    # initialize cluster centers
    idx = np.random.choice(n_samples, self.n_clusters, replace=False)
    self.cluster_centers_ = X[idx].copy()

    iteration = 0
    while self.temperature > self.min_temp and iteration < self.max_iter:

        F, U, S, probs = self._calculate_thermodynamics(X)

        self.history['temperature'].append(self.temperature)
        self.history['energy'].append(U)
        self.history['entropy'].append(S)
        self.history['free_energy'].append(F)
        self.history['centers'].append(self.cluster_centers_.copy())

        # updates
        self.cluster_centers_ = self._get_updated_centers(X, probs)
        self.temperature *= self.cooling_rate
        iteration += 1


    self.labels_ = self.predict(X)
    return self
```


### Predict Labels

```python
def predict(self, X):
    distances = cdist(X, self.cluster_centers_)
    return np.argmin(distances, axis=1)
```



## Experimental Results

EntropyFlow has been benchmarked against **k-means clustering** on synthetic datasets, demonstrating improved performance in capturing subtle, emergent structures:

| Model       | ARI   | NMI   |
| ----------- | ----- | ----- |
| EntropyFlow | 0.941 | 0.940 |
| k-Means     | 0.724 | 0.856 |



## Visualisation & Interpretation

EntropyFlow provides insights into the **evolution of learning systems** by visualizing:

* The dynamics of **energy**, **entropy**, and **free energy** over temperature
* **Phase transitions** during annealing
* The formation of clusters in complex data landscapes
* Comparisons to baseline methods like k-means


## Future Directions

**Thermodynamic Learning** is a broad, extensible framework. EntropyFlow demonstrates its potential in clustering, but future work could explore:

* **Latent variable models** (e.g., thermodynamically guided VAEs)
* **Attention mechanisms** (soft attention as Boltzmann weights)
* **Reinforcement learning** (reward as energy, exploration as entropy)
* **Graph and sequential data** (thermodynamic message-passing)



## Contributions

This project is a work in progress. Feedback, ideas, and contributions are welcome, especially from those interested in **machine learning**, **statistical physics**, or **emergent systems**.




