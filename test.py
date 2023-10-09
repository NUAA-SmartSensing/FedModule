new_clusters_dict = {f'array{i}': [] for i in range(3)}
new_clusters = [3, 1, 1, 2, 3, 2, 3]
print(new_clusters_dict)
for k in range(3):
    for i in range(len(new_clusters)):
        if new_clusters[i] - 1 == k:
            key = f'array{k}'
            new_clusters_dict[key].append(i)
print(new_clusters_dict)
