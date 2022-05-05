# import copy
# import random
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.datasets import cifar10
# import numpy as np
# import scipy.stats as stats
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# import matplotlib.pyplot as plt
#
# import Datasets.MNIST as MNIST
# import Datasets.FashionMNIST as FashionMNIST
# import Datasets.CIFAR10 as CIFAR10
# import Datasets.CIFAR100 as CIFAR100
# import Datasets.ImageNette as ImageNette
#
# from MainFrame.Client import Client
#
#
# def set_gpu_with_increasing_occupancy_mode():
#     # 设置 GPU 显存使用方式为：为增长式占用-----------------------
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:  # 设置 GPU 为增长式占用
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             # 打印异常
#             print(e)
#
#
# def sum_nd_array_lists(nd_array_lists):
#     # 将输入的多个nd_array_list累加成一个nd_array_list
#     nd_array_list_sum = None
#     for nl in nd_array_lists:
#         if nd_array_list_sum is None:
#             nd_array_list_sum = copy.deepcopy(nl)
#         else:
#             for j in range(len(nd_array_lists[0])):
#                 nd_array_list_sum[j] += nl[j]
#     return nd_array_list_sum
#
#
# def sum_nd_array_lists_with_weight(nd_array_lists, important_index, weight):
#     # 将输入的多个nd_array_list累加成一个nd_array_list
#     nd_array_list_sum = None
#     for i in range(len(nd_array_lists)):
#         if nd_array_list_sum is None:
#             nd_array_list_sum = copy.deepcopy(nd_array_lists[i])
#         else:
#             for j in range(len(nd_array_lists[0])):
#                 nd_array_list_sum[j] += nd_array_lists[i][j]
#
#         # 若第i号list刚好是重要的list，则再累加该list (weight - 1)次
#         if i == important_index:
#             for w in range(weight - 1):
#                 for j in range(len(nd_array_lists[0])):
#                     nd_array_list_sum[j] += nd_array_lists[i][j]
#
#     return nd_array_list_sum
#
#
# def avg_nd_array_list(nd_array_list, divider):
#     # 将输入的nd_array_list中的每个nd_array中的每个数字都按divider进行平均
#     averaged_nd_array_list = []
#     for nd_array in nd_array_list:
#         averaged_nd_array = nd_array / divider
#         averaged_nd_array_list.append(averaged_nd_array)
#     return averaged_nd_array_list
#
#
# def weight_nd_array_list(nd_array_list, weight):
#     # 将输入的nd_array_list中的每个array乘以一个权重系数weight
#     weighted_nd_array_list = []
#     for nd_array in nd_array_list:
#         averaged_nd_array = nd_array * weight
#         weighted_nd_array_list.append(averaged_nd_array)
#     return weighted_nd_array_list
#
#
# def count_in_list(number_list):
#     result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     for n in number_list:
#         c = int(n / 10)
#         result[c] += 1
#     return result
#
#
# def move_forward_in_list(input_list):
#     # 将一个列表中的所有元素前移一位并返回产生的新列表
#     p_array = np.array(input_list)
#     p_temp = p_array[0]
#     for i in range(len(p_array) - 1):
#         p_array[i] = p_array[i + 1]
#     p_array[len(p_array) - 1] = p_temp
#     result_list = p_array.tolist()
#     return result_list
#
#
# def find_range_in_list(target, range_list):
#     for rl in range(len(range_list)):
#         if target < range_list[rl]:
#             return rl
#     return -1
#
#
# def normal_distribution_from_two_lists(x_list, y_list, capacity, class_size, class_num):
#     sizes = (118, 108, 90, 69, 49, 31, 18, 10, 5, 2)  # 标准正态分布的10类别概率面积
#     result_list = []
#
#     size_sum = 0
#     for s in sizes:
#         size_sum += s
#
#     class_index = []
#     for ci in range(class_num):
#         class_index.append(int(ci * class_size))
#
#     probabilities = list(sizes)
#     for r_id in range(100):
#         # 创建一个决定类别的decide_list
#         decide_list = copy.deepcopy(probabilities)
#         for d in range(1, len(decide_list)):
#             decide_list[d] = decide_list[d] + decide_list[d - 1]
#
#         # 按每个结果元素的容量，反复进行随机抽取，以生产出一个结果元素
#         x_result = []
#         y_result = []
#         for cap in range(capacity):
#             dd = random.randint(0, size_sum - 1)
#             random_choice = find_range_in_list(dd, decide_list)
#             chosen_index = class_index[random_choice]
#             # 若该类还有剩余，则正常添加
#             if class_index[random_choice] < ((random_choice + 1) * class_size):
#                 x_result.append(x_list[chosen_index])
#                 y_result.append(y_list[chosen_index])
#                 class_index[random_choice] += 1
#             # 若该类以没有剩余，则在有剩余的类中选取一个
#             elif class_index[random_choice] == ((random_choice + 1) * class_size):
#                 next_choice = copy.deepcopy(random_choice)
#                 # 依次查询每个类，寻找剩余
#                 for s_c in range(class_num):
#                     next_choice = (next_choice + 1) % 10
#                     if class_index[next_choice] < ((next_choice + 1) * class_size):
#                         next_chosen_index = class_index[next_choice]
#                         x_result.append(x_list[next_chosen_index])
#                         y_result.append(y_list[next_chosen_index])
#                         class_index[next_choice] += 1
#                         break
#                     # 若将所有类遍历一遍都没有找到剩余，则说明出现了错误
#                     if s_c == class_num - 1:
#                         print("Error!", random_choice)
#             # 不应发生的情况，将报错
#             else:
#                 print("Error!!")
#
#         # 将生成好的结果元素加入结果list
#         result_list.append([x_result, y_result])
#
#         # 将probabilities列表所有元素前移一位，以便下次循环使用
#         probabilities = move_forward_in_list(probabilities)
#
#     return result_list
#
#
# def preprocess(x, y):
#     # [0~1]
#     x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
#     y = tf.cast(y, dtype=tf.int32)
#     return x, y
#
#
# def normalization(train_images, test_images):
#     mean = np.mean(train_images, axis=(0, 1, 2, 3))
#     std = np.std(train_images, axis=(0, 1, 2, 3))
#     train_images = (train_images - mean) / (std + 1e-7)
#     test_images = (test_images - mean) / (std + 1e-7)
#     return train_images, test_images
#
#
# def load_cifar10_images():
#     (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
#
#     train_images = train_images.astype(np.float32)
#     test_images = test_images.astype(np.float32)
#
#     (train_images, test_images) = normalization(train_images, test_images)
#
#     train_labels = to_categorical(train_labels, 10)
#     test_labels = to_categorical(test_labels, 10)
#
#     return train_images, train_labels, test_images, test_labels
#
#
# def build_optimizer(learning_rate=0.1, momentum=0.9):
#     learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#         [500, 32000, 48000],
#         [learning_rate / 10., learning_rate, learning_rate / 10., learning_rate / 100.])
#
#     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
#
#     return optimizer
#
#
# def generate_iid_clients(x, y, client_num, client_size):
#     clients_x = []
#     clients_y = []
#     for i in range(client_num):
#         begin_index = int(i * client_size)
#         end_index = int((i + 1) * client_size)
#         clients_x.append(x[begin_index: end_index])
#         clients_y.append(y[begin_index: end_index])
#
#     return clients_x, clients_y
#
#
# def generate_non_iid_clients(use_iid_clients, sorted_x, sorted_y, client_num, client_size, type_num):
#     clients_x = []
#     clients_y = []
#     if use_iid_clients == 1:
#         client_num_per_type = int(client_num / type_num)
#         for t in range(type_num):
#             sorted_y[t] = tf.reshape(sorted_y[t], [sorted_y[t].shape[0], 1])
#             sorted_y[t] = sorted_y[t].numpy()
#             for c_t in range(client_num_per_type):
#                 begin_index, end_index = int(c_t * client_size), int((c_t + 1) * client_size)
#                 clients_x.append(sorted_x[t][begin_index: end_index])
#                 clients_y.append(sorted_y[t][begin_index: end_index])
#     elif use_iid_clients == 2 or use_iid_clients == 3:
#         for t in range(type_num):
#             sorted_y[t] = tf.reshape(sorted_y[t], [sorted_y[t].shape[0], 1])
#             sorted_y[t] = sorted_y[t].numpy()
#         client_size1 = int(client_size / 2)
#         client_size2 = int(client_size - client_size1)
#         type_indexes = [0 for _ in range(type_num)]
#         type1, type2 = 0, 1
#         for c in range(client_num):
#             print(sorted_x[type1][type_indexes[type1]: type_indexes[type1] + client_size1].shape,
#                   sorted_y[type1][type_indexes[type1]: type_indexes[type1] + client_size1].shape)
#             clients_x.append(np.vstack([sorted_x[type1][type_indexes[type1]: type_indexes[type1] + client_size1],
#                                         sorted_x[type2][type_indexes[type2]: type_indexes[type2] + client_size2]]))
#             clients_y.append(np.vstack([sorted_y[type1][type_indexes[type1]: type_indexes[type1] + client_size1],
#                                         sorted_y[type2][type_indexes[type2]: type_indexes[type2] + client_size2]]))
#             print(np.vstack([sorted_x[type1][type_indexes[type1]: type_indexes[type1] + client_size1],
#                              sorted_x[type2][type_indexes[type2]: type_indexes[type2] + client_size2]]).shape,
#                   np.vstack([sorted_y[type1][type_indexes[type1]: type_indexes[type1] + client_size1],
#                              sorted_y[type2][type_indexes[type2]: type_indexes[type2] + client_size2]]).shape, "\n")
#             type_indexes[type1] += client_size1
#             type_indexes[type2] += client_size2
#             if use_iid_clients == 2:
#                 type1 += 1
#                 type2 += 1
#                 type1 = type1 % 10
#                 type2 = type2 % 10
#             elif use_iid_clients == 3:
#                 type2 += 1
#                 if type2 == type_num:
#                     type1 += 1
#                     type2 = type1 + 1
#                 if type1 == type_num - 1:
#                     type1, type2 = 0, 1
#     elif use_iid_clients == 4 or use_iid_clients == 5 or use_iid_clients == 6 or use_iid_clients == 7:
#         client_num_per_type = int(client_num / type_num)
#         sub_c = 0
#         if use_iid_clients == 4:
#             main_size = 0.8
#         elif use_iid_clients == 5:
#             main_size = 0.5
#         elif use_iid_clients == 6:
#             main_size = 0.3
#         elif use_iid_clients == 7:
#             main_size = 0.2
#         else:
#             main_size = 0.8
#         main_client_size = int(client_size * main_size)
#         sub_client_size = int(client_size * (1 - main_size) / type_num)
#         print("main_client_size =", main_client_size)
#         print("sub_client_size =", sub_client_size)
#         for t in range(type_num):
#             sorted_y[t] = tf.reshape(sorted_y[t], [sorted_y[t].shape[0], 1])
#             sorted_y[t] = sorted_y[t].numpy()
#         for t in range(type_num):
#             for c_t in range(client_num_per_type):
#                 begin_index, end_index = int(c_t * main_client_size), int((c_t + 1) * main_client_size)
#                 c_x = sorted_x[t][begin_index: end_index]
#                 c_y = sorted_y[t][begin_index: end_index]
#                 for tt in range(type_num):
#                     print(sub_c, ":", c_x.shape, c_y.shape, " -- ",
#                           sorted_x[tt][int(sub_c * sub_client_size): int((sub_c + 1) * sub_client_size)].shape,
#                           sorted_y[tt][int(sub_c * sub_client_size): int((sub_c + 1) * sub_client_size)].shape)
#                     c_x = np.vstack(
#                         [c_x, sorted_x[tt][int(sub_c * sub_client_size): int((sub_c + 1) * sub_client_size)]])
#                     c_y = np.vstack(
#                         [c_y, sorted_y[tt][int(sub_c * sub_client_size): int((sub_c + 1) * sub_client_size)]])
#                 sub_c += 1
#                 clients_x.append(c_x)
#                 clients_y.append(c_y)
#
#     return clients_x, clients_y
#
#
# def generate_sorted_dataset(x, y, data_num):
#     # 将数据集按类别排序
#     y = tf.reshape(y, data_num)
#     y = y.numpy()
#     sorted_index = np.argsort(y)
#
#     y = tf.reshape(y, (data_num, 1))
#     y = y.numpy()
#     x = x[sorted_index]
#     y = y[sorted_index]
#     y = tf.squeeze(y, axis=1)
#     y = y.numpy()
#
#     # 按顺序将数据分类存储在list中
#     sorted_x = []
#     sorted_y = []
#     count = 0
#     sum_c = 0
#     yy = int(y[0])
#     for i in range(data_num):
#         if yy != int(y[i]):
#             yy = int(y[i])
#             sorted_x.append(x[sum_c: sum_c + count])
#             sorted_y.append(y[sum_c: sum_c + count])
#             sum_c += count
#             count = 1
#         else:
#             count += 1
#     sorted_x.append(x[sum_c: sum_c + count])
#     sorted_y.append(y[sum_c: sum_c + count])
#     sum_c += count
#
#     return sorted_x, sorted_y
#
#
# def generate_data(data_type):
#     # 生成数据集
#     if data_type == "MNIST":
#         dataset = MNIST.MNIST()
#     elif data_type == "FashionMNIST":
#         dataset = FashionMNIST.FashionMNIST()
#     elif data_type == "CIFAR10":
#         dataset = CIFAR10.CIFAR10()
#     elif data_type == "CIFAR100":
#         dataset = CIFAR100.CIFAR100()
#     elif data_type == "ImageNette":
#         dataset = ImageNette.ImageNette()
#     else:
#         dataset = MNIST.MNIST()
#
#     c_x, c_y = dataset.get_train_data()
#     c_x_valid, c_y_valid = dataset.get_valid_data()
#     # c_x_test, c_y_test = dataset.get_test_data()
#     c_x_test, c_y_test = dataset.get_big_test_data()
#
#     return dataset, c_x, c_y, c_x_valid, c_y_valid, c_x_test, c_y_test
#
#
# def generate_clients(gpu_list, dataset, data_type, model_name, client_num, client_size, batch_size, use_iid_clients,
#                      c_x, c_y):
#     # 生成各个客户端的IID或non_IID的数据
#     if use_iid_clients == 0:
#         client_data_x, client_data_y = generate_iid_clients(c_x, c_y, client_num, int(len(c_x) / client_num))
#     else:
#         sorted_x, sorted_y = dataset.get_sorted_dataset()
#         client_data_x, client_data_y = generate_non_iid_clients(use_iid_clients, sorted_x, sorted_y, client_num,
#                                                                 client_size, len(sorted_x))
#         for i in range(client_num):
#             yl = len(client_data_y[i])
#             print(i, client_data_y[i][0], client_data_y[i][int(yl / 2 - 1)], client_data_y[i][int(yl / 2)],
#                   client_data_y[i][yl - 1])
#
#     # 生成clients
#     c_list = []
#     for c_id in range(client_num):
#         # if c_id % 100 < 5:
#         #     client_gpu_id = random.choice(gpu_list)
#         #     print(c_id, "use GPU:", client_gpu_id)
#         #     with tf.device('/gpu:' + str(client_gpu_id)):
#         #         new_client = Client(client_gpu_id, data_type, model_name, [client_data_x[c_id], client_data_y[c_id]],
#         #                             batch_size)
#         #
#         #     c_list.append(new_client)
#         client_gpu_id = random.choice(gpu_list)
#         print(c_id, "use GPU:", client_gpu_id)
#         with tf.device('/gpu:' + str(client_gpu_id)):
#             new_client = Client(client_gpu_id, data_type, model_name, [client_data_x[c_id], client_data_y[c_id]],
#                                 batch_size)
#
#         c_list.append(new_client)
#
#     return c_list
#
#
# def shuffle_ndarray_list(ndarray_list, shuffle_time):
#     for array in ndarray_list:
#         # 打乱 shuffle_time 次
#         for s_t in range(shuffle_time):
#             index1 = random.randint(0, len(array) - 1)
#             index2 = random.randint(0, len(array) - 1)
#             if index1 != index2:
#                 array[[index1, index2], :] = array[[index2, index1], :]
#
#
# def select_iid_clients(network, clients_list, client_index_list, test_x, test_y):
#     iid_clients_list = []
#     iid_clients_losses = []
#     iid_clients_index_list = []
#     index = 0
#
#     for client in clients_list:
#         if len(iid_clients_list) == 0:
#             iid_clients_index_list.append([client_index_list[index]])
#             iid_clients_list.append([client])
#             network.set_weights(client.get_client_weights())
#             accuracy, loss = network.evaluate_network(test_x, test_y)
#             iid_clients_losses.append(loss)
#         else:
#             # 与各类client依次进行比较
#             for s_c in range(len(iid_clients_list)):
#                 sum_client_weights_list = sum_nd_array_lists([client.get_client_weights(),
#                                                               iid_clients_list[s_c][0].get_client_weights()])
#                 averaged_client_weights_list = avg_nd_array_list(sum_client_weights_list, 2)
#                 network.set_weights(averaged_client_weights_list)
#                 accuracy, loss = network.evaluate_network(test_x, test_y)
#
#                 # 若是同类则将其归入其所属类的list中
#                 if loss > (iid_clients_losses[s_c] * 0.8):
#                     iid_clients_index_list[s_c].append(client_index_list[index])
#                     iid_clients_list[s_c].append(client)
#                     break
#
#                 # 若与已有的client种类都不同，即新种类的client，则新创建一个类别的list并将其加入
#                 if s_c == len(iid_clients_list) - 1:
#                     iid_clients_index_list.append([client_index_list[index]])
#                     iid_clients_list.append([client])
#                     network.set_weights(client.get_client_weights())
#                     accuracy, loss = network.evaluate_network(test_x, test_y)
#                     iid_clients_losses.append(loss)
#         index += 1
#
#     return iid_clients_list, iid_clients_index_list
#
#
# def get_coordinate_by_index_from_lists_list(index, lists_list):
#     start_index = 0
#     end_index = 0
#     row = -1
#     col = -1
#     for i in range(len(lists_list)):
#         if i == 0:
#             start_index = 0
#             end_index = len(lists_list[i]) - 1
#         else:
#             start_index += len(lists_list[i - 1])
#             end_index += len(lists_list[i])
#
#         if start_index <= index <= end_index:
#             row = i
#             col = index - start_index
#             break
#     return row, col
#
#
# def random_select_from_lists_list(lists_list, ratio):
#     random_selected_lists_list = []
#     sum_in_lists_list = 0
#
#     # 初始化random_selected_lists_list并求出lists_list中元素的总数
#     for i in range(len(lists_list)):
#         random_selected_lists_list.append([])
#         sum_in_lists_list += len(lists_list[i])
#
#     index_list = list(range(sum_in_lists_list))
#     print(index_list)
#
#     # 随机将lists_list的元素加入random_selected_lists_list中的对应位置
#     print("select number:", int(ratio * sum_in_lists_list))
#     random_indexes = random.sample(index_list, int(ratio * sum_in_lists_list))
#     for r_index in random_indexes:
#         row, col = get_coordinate_by_index_from_lists_list(r_index, lists_list)
#         random_selected_lists_list[row].append(lists_list[row][col])
#
#     return random_selected_lists_list
#
#
# def l2_regularization_of_array(array):
#     # 将一个任意维数的array进行L2正则化，即计算它在其相应的多维空间内到原点的距离
#     dimension_number = len(array.shape)
#     r_array = copy.deepcopy(array)
#     for i in range(dimension_number):
#         r_array = np.linalg.norm(r_array, axis=int(dimension_number - i - 1), keepdims=True)
#     return float(r_array)
#
#
# def generate_normal_distribution_list(lower, upper, mu, sigma, num):
#     # 有最大最小值约束的正态分布
#     x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
#     normal_distribution_list = x.rvs(num)  # 取其中的5个数，赋值给a, a为array类型
#     return normal_distribution_list
#
#
# def balance_select(q_s_list, c_s_list, select_num, strategy):
#     # q_s_list: queue_size_list
#     # c_s_list: cluster_size_list
#     q_s_sum = sum(q_s_list)
#     if strategy == 1:
#         ratio_list = []
#         for q_s in q_s_list:
#             if q_s == 0:
#                 ratio_list.append(1 - (1 / q_s_sum))
#             else:
#                 ratio_list.append(1 - (q_s / q_s_sum))
#         power_ratio_list = copy.deepcopy(ratio_list)
#         last_power_ratio_list = copy.deepcopy(power_ratio_list)
#         last_power_ratio_sum = copy.deepcopy(sum(power_ratio_list))
#
#         # 使sum(power_ratio_list)趋近于1
#         while sum(power_ratio_list) >= 1:
#             last_power_ratio_list = copy.deepcopy(power_ratio_list)
#             last_power_ratio_sum = copy.deepcopy(sum(power_ratio_list))
#             for p_r in power_ratio_list:
#                 power_ratio_list[power_ratio_list.index(p_r)] = p_r * ratio_list[power_ratio_list.index(p_r)]
#
#         # 选择最接近1的总和
#         if (1 - sum(power_ratio_list)) < (1 - last_power_ratio_sum):
#             best_power_ratio_list = power_ratio_list
#         else:
#             best_power_ratio_list = last_power_ratio_list
#     elif strategy == 2:
#         best_power_ratio_list = []
#         n = len(q_s_list)
#         for q_s in q_s_list:
#             best_power_ratio_list.append((1 / (n - 1)) * (1 - (q_s / q_s_sum)))
#     elif strategy == 4:
#         best_power_ratio_list = []
#         n = len(q_s_list)
#         avg_q_s = np.mean(q_s_list)
#         var_q_s = np.var(q_s_list)
#         print("avg_q_s:", avg_q_s)
#         print("var_q_s:", var_q_s)
#
#         # 统计 queue_size 小于平均 size 的 cluster 的数量
#         small_cluster_number = 0
#         small_q_s_sum = 0
#         for q_s in q_s_list:
#             if q_s < avg_q_s:
#                 small_cluster_number += 1
#                 small_q_s_sum += q_s
#
#         # 若当前方差过大，则 queue_size 大于平均 size 的 cluster 不进行分配
#         if var_q_s >= 50:
#             for q_s in q_s_list:
#                 if q_s >= avg_q_s:
#                     best_power_ratio_list.append(0)
#                 else:
#                     best_power_ratio_list.append((1 / (small_cluster_number - 1)) * (1 - (q_s / small_q_s_sum)))
#         else:
#             for q_s in q_s_list:
#                 best_power_ratio_list.append((1 / (n - 1)) * (1 - (q_s / q_s_sum)))
#     else:
#         print("Error strategy!!")
#         best_power_ratio_list = []
#         n = len(q_s_list)
#         for q_s in q_s_list:
#             best_power_ratio_list.append((1 / (n - 1)) * (1 - (q_s / q_s_sum)))
#
#     # 将c_s_list排序并获取对应的索引
#     c_s_rank_list = [index for index, value in sorted(list(enumerate(c_s_list)), key=lambda x: x[1])]
#
#     # 初始化select_list并统计各个c中缺少的数量
#     select_list = []
#     for b_p_r in range(len(best_power_ratio_list)):
#         best_select = int(select_num * best_power_ratio_list[b_p_r])
#         # 该c中充足
#         if best_select <= c_s_list[b_p_r]:
#             select_list.append(best_select)
#         # 该c中不足
#         else:
#             select_list.append(c_s_list[b_p_r])
#     insufficient_number = max(0, select_num - sum(select_list))
#
#     # 进行补充
#     while insufficient_number > 0:
#         for c_s_r in c_s_rank_list:
#             # 若该c中仍有剩余，则用其进行补充
#             if insufficient_number > 0 and c_s_list[c_s_r] > select_list[c_s_r]:
#                 supplement_number = 1  # min(insufficient_number, c_s_list[c_s_r] - select_list[c_s_r])
#                 select_list[c_s_r] = select_list[c_s_r] + supplement_number
#                 insufficient_number -= supplement_number
#
#     return select_list
#
#
# def calculate_accuracy_precision_recall_f1(true_labels_list, pred_labels_list):
#     # c_m = confusion_matrix(true_labels_list, pred_labels_list)
#     # print("c_m:", c_m)
#     accuracy = accuracy_score(true_labels_list, pred_labels_list)
#     precision = precision_score(true_labels_list, pred_labels_list, average='macro', zero_division=0)
#     recall = recall_score(true_labels_list, pred_labels_list, average='macro', zero_division=0)
#     f1 = f1_score(true_labels_list, pred_labels_list, average='macro', zero_division=0)
#     return accuracy, precision, recall, f1
#
#
# # if __name__ == '__main__':
# #     true_labels_list = [0, 1, 2, 0, 1, 2]
# #     pred_labels_list = [0, 2, 3, 0, 0, 1]
# #     true_labels_list = [0, 0, 0, 1, 1, 1]
# #     pred_labels_list = [0, 1, 1, 0, 1, 1]
# #     accuracy, precision, recall, f1 = calculate_accuracy_precision_recall_f1(true_labels_list, pred_labels_list)
# #     print(accuracy, precision, recall, f1)
