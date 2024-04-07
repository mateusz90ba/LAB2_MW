import tkinter as tk
from tkinter import filedialog
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
def read_xyz_file(file_path):
    # Funkcja do odczytu danych z pliku XYZ
    points = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 3:  # Upewnij się, że są trzy współrzędne XYZ
                points.append(list(map(float, row)))
    return points


def plot_3d_points(points_list, ax=None):
    # Funkcja do wyświetlenia chmury punktów 3D.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for points in points_list:
        xs, ys, zs = zip(*points)
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def fit_plane_ransac(points, iterations=1000, tolerance=0.1):
    # Funkcja do dopasowania płaszczyzny do chmury punktów za pomocą algorytmu RANSAC.
    best_plane = None
    best_inliers = []

    for i in range(iterations):
        # Wybierz losowo trzy punkty, aby utworzyć model płaszczyzny
        sample = random.sample(points, 3)

        # Oblicz równanie płaszczyzny za pomocą wybranej próbki
        v1 = np.array(sample[1]) - np.array(sample[0])
        v2 = np.array(sample[2]) - np.array(sample[0])
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, sample[0])

        # Oblicz odległość punktów od płaszczyzny
        inliers = []
        for point in points:
            distance = np.abs(np.dot(normal, point) + d) / np.linalg.norm(normal)
            if distance < tolerance:
                inliers.append(point)

        # Aktualizuj najlepszy model, jeśli obecny model ma więcej inlierów
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers

def browse_files():
    #Funkcja do wybierania plików z rozszerzeniem .xyz
    file_paths = filedialog.askopenfilenames(filetypes=[("XYZ files", "*.xyz")])
    points_list = []
    for file_path in file_paths:
        points = read_xyz_file(file_path)
        points_list.append(points)

    # Dopasuj płaszczyznę do punktów za pomocą algorytmu RANSAC
    plane, inliers = fit_plane_ransac(points_list[0])  # Możesz użyć pierwszego zestawu punktów

    # Wyświetl wyniki dopasowania płaszczyzny
    print("Równanie dopasowanej płaszczyzny:", plane)
    print("Liczba (ang. inliers):", len(inliers))
    print("Współrzędne wektora normalnego:", plane[0])  # Dodaj wypisanie współrzędnych wektora normalnego, można je uzyskać z równania dopasowanej płaszczyzny
    # W równaniu płaszczyzny w postaci ogólnej Ax+By+Cz+D=0, współczynniki A, B i C odpowiadają współrzędnym wektora normalnego do płaszczyzny
    # Oblicz średnią odległość punktów od płaszczyzny
    distances = [np.abs(np.dot(plane[0], point) + plane[1]) / np.linalg.norm(plane[0]) for point in inliers]
    mean_distance = np.mean(distances)

    # Wyświetl informację czy chmura jest płaszczyzną oraz informację o płaszczyźnie
    if mean_distance < 0.1:  # Ustaw próg, poniżej którego uznajemy, że chmura jest płaszczyzną
        print("Chmura punktów jest płaszczyzną.")
        if abs(plane[0][2]) < 0.1:  # Sprawdź czy współrzędna Z wektora normalnego jest bliska zeru
            print("Płaszczyzna jest pionowa.")
        else:
            print("Płaszczyzna jest pozioma.")
    else:
        print("Chmura punktów nie jest płaszczyzną.")

    # Wyświetl chmurę punktów
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_points(points_list, ax)

    # Wyświetl dopasowaną płaszczyznę UWAGA! - DLA PIONOWEJ CHMURY PUNKTÓW NIE WYŚWIETLI SIĘ DOPASOWANA PŁASZCZYZNA, BO JEST PRÓBA DZIELENIA PRZEZ ZERO
    xs, ys, zs = zip(*inliers)
    X, Y = np.meshgrid(np.linspace(min(xs), max(xs), 10), np.linspace(min(ys), max(ys), 10))
    Z = (-plane[0][0] * X - plane[0][1] * Y - plane[1]) / plane[0][2]
    #ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.5)

    #tu się zaczyna algorytm k-średnich - minimalizuje odległości punktów od centrum klastra, chmura punktów dzieli się jakby na 3 mniejsze
    #################################################################################################################
    # Znajdź rozłączne klastry dla algorytmu k-średnich i wyświetl je
    cluster_centers, disjoint_clusters = find_disjoint_clusters(points_list)
    # Wyświetl centra klastrów
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='o', s=100)
    ##################################################################################################################
    # Znajdź rozłączne klastry za pomocą algorytmu DBSCAN
    # Sprawdź, czy istnieją klastry przed próbą ich wydobycia
    '''if points_list:
        # Znajdź rozłączne klastry za pomocą algorytmu DBSCAN
        disjoint_clusters = find_disjoint_clusters_dbscan(points_list)'''
    ####################################################################################################################
    # Wyświetl rozłączne klastry
    for cluster in disjoint_clusters:
        xs, ys, zs = zip(*cluster)
        ax.scatter(xs, ys, zs, marker='.')

    plt.show()

def find_disjoint_clusters(points_list):
    all_points = [point for sublist in points_list for point in sublist]  # Połącz wszystkie punkty z różnych chmur w jedną listę
    kmeans = KMeans(n_clusters=3)  # Ustawienie na sztywno liczby klastrów na 3
    kmeans.fit(all_points) #rozpoznawanie danych przy użyciu funkcji fit, znaleźć reguły, które rządzą podziałem danych
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Podziel punkty na klastry na podstawie przypisanych etykiet
    clusters = [[] for _ in range(3)]
    for i, point in enumerate(all_points):
        clusters[labels[i]].append(point)

    return cluster_centers, clusters

def find_disjoint_clusters_dbscan(points_list, eps=0.5, min_samples=3):
    all_points = np.concatenate(points_list)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(all_points)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Ustal liczbę klastrów, pomijając punkty szumu oznaczone jako -1

    # Podziel punkty na klastry na podstawie przypisanych etykiet
    clusters = [[] for _ in range(num_clusters)]
    for i, point in enumerate(all_points):
        if labels[i] != -1:  # Pomijaj punkty szumu
            clusters[labels[i]].append(point)

    return clusters


# Utwórz główne okno aplikacji tkinter
root = tk.Tk()
root.title("Wybierz pliki XYZ")

# Zmień rozmiar okna tkinter
root.geometry("400x150")  # Szerokość x Wysokość

# Przycisk do wybierania plików
browse_button = tk.Button(root, text="Wybierz pliki", command=browse_files)
browse_button.pack(pady=10)

# Uruchom pętlę główną aplikacji tkinter
root.mainloop()
