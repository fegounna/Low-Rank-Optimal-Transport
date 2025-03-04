import factor_relaxation as fr
import numpy as np
import ot
import matplotlib.pyplot as plt

# Générer des points aléatoires dans le plan euclidien
n, m = 50, 50  # Nombre de points dans a et b
np.random.seed(42)

def generate_gaussian_mixture(n, centers, std=0.2):
    """
    Génère un mélange de gaussiennes avec plusieurs centres.

    Parameters:
    - n : nombre total de points.
    - centers : liste des centres des gaussiennes (list of tuples).
    - std : écart-type des gaussiennes.

    Returns:
    - points : array (n,2) des coordonnées des points.
    """
    num_clusters = len(centers)
    points = []
    
    for i in range(n):
        center = centers[np.random.randint(num_clusters)]  # Choisir un centre aléatoire
        point = np.random.randn(2) * std + center  # Générer un point autour du centre
        points.append(point)
    
    return np.array(points)

def generate_spiral(n, num_turns=2, noise=0.5):
    """
    Génère une répartition en spirale.

    Parameters:
    - n : nombre de points.
    - num_turns : nombre de tours de la spirale.
    - noise : niveau de bruit ajouté aux points.

    Returns:
    - points : array (n,2) des coordonnées des points.
    """
    # t = np.linspace(0, num_turns * 2 * np.pi, n)  # Paramètre de la spirale
    t = np.random.rand(n) * num_turns * 1.5 * np.pi  # Paramètre de la spirale

    x = t * np.cos(t) + noise * np.random.randn(n)
    y = t * np.sin(t) + noise * np.random.randn(n)
    
    # Normalisation pour que les points restent dans une zone raisonnable
    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2
    
    x = x - np.average(x)
    y = y - np.average(y)
    x=x*2
    y=y*2
    
    return np.vstack((x, y)).T

centers_b = [(-1.5, -1.5), (3, 0), (-1, 2.5), (.6, 1)]

points_a = generate_spiral(n, num_turns=2)
points_b = generate_gaussian_mixture(m, centers_b)

# Calculer la matrice de coût (distance euclidienne)
cost_matrix = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)

# Générer des distributions de poids
a = np.ones(n) / n
b = np.ones(m) / m



# Fonction d'affichage du transport optimal
def plot_transport(points_a, points_b, transport_matrix, title, filename):
    """Affiche le transport optimal donné par une matrice de transport."""
    max_transport = np.max(transport_matrix)
    if max_transport > 0:
        transport_matrix /= max_transport  # Normalisation pour l'affichage

    plt.figure(figsize=(8, 6))
    plt.scatter(points_a[:, 0], points_a[:, 1], c='blue', label='Source points (a)')
    plt.scatter(points_b[:, 0], points_b[:, 1], c='red', label='Target points (b)')

    # Afficher les connexions entre points selon transport_matrix
    for i in range(n):
        for j in range(m):
            if transport_matrix[i, j] > 1e-3:  # Seuil pour éviter trop de traits
                line_width = 5 * transport_matrix[i, j]  # Épaisseur proportionnelle au transport
                alpha_value = 0.6 * transport_matrix[i, j]  # Opacité proportionnelle
                plt.plot([points_a[i, 0], points_b[j, 0]], [points_a[i, 1], points_b[j, 1]], 
                         'black', alpha=alpha_value, linewidth=line_width)

    plt.legend()
    plt.title(title)
    plt.savefig(filename)  # Sauvegarde de l'image
    plt.show()

def plot_loss(loss_liste, title, filename):
    """
    Affiche et sauvegarde l'évolution de la loss pendant l'optimisation.

    Parameters:
    - loss_liste : liste des valeurs de la loss à chaque itération.
    - title : titre du graphique.
    - filename : nom du fichier pour sauvegarder l'image.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_liste, label="Loss", color="blue")
    plt.xlabel("Itération")
    plt.ylabel("Loss")
    plt.yscale("log")  # Échelle logarithmique pour mieux voir la convergence
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(filename)  # Sauvegarde de l'image
    plt.show()

# Résolution du problème de transport optimal avec FRLC
# for tau in [100, 10, 1, .1, 1e-2, 1e-3, 1e-4]:
#     for gamma in [1e5, 1e4, 1e3, 1e2, 10, 1]:

#         r = 10
#         tau = tau
#         gamma = gamma
#         delta = -1 # sert à rien
#         epsilon = 1e-6
#         max_iter = 50
        
#         transport_matrix_frlc, loss_liste = fr.solve_balanced_FRLC(
#             C=cost_matrix, 
#             r=r, 
#             a=a, 
#             b=b, 
#             tau=tau, 
#             gamma=gamma, 
#             delta=delta, 
#             epsilon=epsilon, 
#             max_iter=max_iter
#         )
#         print(f"gamma:{gamma}, tau:{tau}", loss_liste)
#         plot_loss(loss_liste, "Transport Optimal Low-Rank (FRLC), loss evolution", f"loss_transport_frlc_tau_{tau}.png")
#     # plot_transport(points_a, points_b, transport_matrix_frlc, "Transport Optimal Low-Rank (FRLC)", f"transport_frlc_tau_{tau}.png")


r = 5
tau = 100
gamma = 100
delta = 10 # sert à rien
epsilon = 1e-20
max_iter = 200


transport_matrix_frlc, loss_liste = fr.solve_balanced_FRLC(
    C=cost_matrix, 
    r=r, 
    a=a, 
    b=b, 
    tau=tau, 
    gamma=gamma, 
    delta=delta, 
    epsilon=epsilon, 
    max_iter=max_iter
)
print(f"gamma:{gamma}, tau:{tau}", loss_liste)
plot_loss(loss_liste, "Transport Optimal Low-Rank (FRLC), loss evolution", f"./output/loss_transport_frlc_tau_{tau}.png")
print("ploted")
plot_transport(points_a, points_b, transport_matrix_frlc, "Transport Optimal Low-Rank (FRLC)", f"./output/transport_frlc_tau_{tau}.png")
print("ploted*2")

# Résolution du problème de transport optimal avec Sinkhorn (via POT)
reg_sinkhorn = 1e-2


# Paramètre de régularisation
transport_matrix_sinkhorn = ot.sinkhorn(a, b, cost_matrix, reg_sinkhorn)
print('loss_sinkhorn', np.sum(transport_matrix_sinkhorn*cost_matrix))

plot_transport(points_a, points_b, transport_matrix_sinkhorn, "Transport Optimal Sinkhorn", "./output/transport_sinkhorn.png")
