import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import numpy as np
import os

class AutonomousCarSimulator:
    def __init__(self, root): 
        self.root = root
        self.root.title("自走車模擬器")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 增加標籤 "選擇訓練檔案:"
        training_label = tk.Label(self.root, text="選擇訓練檔案:")
        training_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        # 設定下拉選單
        self.training_file_var = tk.StringVar()
        # self.training_file_var.set("選擇訓練檔案")
        self.training_file_menu = ttk.Combobox(root, textvariable=self.training_file_var)
        self.training_file_menu['values'] = ("train4dAll.txt", "train6dAll.txt")
        self.training_file_menu.grid(row=0, column=0, padx=100, pady=5, sticky='w')
       

        # 訓練按鈕
        self.train_button = tk.Button(root, text="Training", command=self.train_model)
        self.train_button.grid(row=0, column=1, padx=0, pady=5)

        # 運行按鈕
        self.run_button = tk.Button(root, text="Run", command=self.run_simulation)
        self.run_button.grid(row=0, column=2, padx=0, pady=5)

        # 新增：繪製資料集軌跡按鈕
        # self.plot_data_button = tk.Button(root, text="Plot Dataset", command=self.plot_dataset_trajectory)
        # self.plot_data_button.grid(row=0, column=3, padx=0, pady=5)

        # 繪圖區域
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

        # 初始化軌道和車子
        self.track_points = []
        self.start_position = None

        # 加載並顯示軌道和車輛
        self.load_track()
        self.draw_track()

        # 初始化 RBFN 模型參數
        self.centers = None  # 從訓練好的模型加載
        self.weights = None  # 從訓練好的模型加載
        self.sigma = None    # 從訓練好的模型加載

        # 添加距離顯示標籤
        self.distance_label = tk.Label(root, text="距離: ", justify=tk.LEFT)
        self.distance_label.grid(row=2, column=0, columnspan=3, sticky='w', padx=5)

    def on_closing(self): 
        self.root.quit() 
        self.root.destroy() 
        sys.exit()
        
    def load_track(self): 
        # 使用絕對路徑
        # file_path = os.path.join(os.path.dirname(__file__), "軌道座標點.txt")
        file_path = "./軌道座標點.txt"
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 讀取起始位置
            start_x, start_y, start_angle = map(float, lines[0].split(','))
            self.start_position = (start_x, start_y, start_angle)
            # 讀取軌道邊界節點
            for line in lines[3:]:  # 從第四行開始讀取軌道邊界節點
                x, y = map(float, line.split(','))
                self.track_points.append((x, y))
            self.track_points.append(self.track_points[0])  # 封閉軌道

    def draw_track(self, show_start_position=True):  
        # 清除之前的圖形
        self.ax.clear()
        # 繪製網格
        self.ax.grid(True)  # 開啟網格
        # 繪製軌道
        if self.track_points:
            x, y = zip(*self.track_points)
            self.ax.plot(x, y, 'b-')
            self.ax.set_xlabel("X axis")
            self.ax.set_ylabel("Y axis")
            self.ax.axis('equal')  # 保持比例

        # 繪製終點區域
        end_zone_x = [18, 30, 30, 18, 18]
        end_zone_y = [40, 40, 37, 37, 40]
        self.ax.fill(end_zone_x, end_zone_y, 'lightcoral', alpha=0.5)

        # 繪製起點橫線
        self.ax.plot([-7, 7], [0, 0], 'k-', linewidth=2)

        # 只在需要時繪製車輛起始位置
        if show_start_position and self.start_position:
            start_x, start_y, start_angle = self.start_position
            car_circle = plt.Circle((start_x, start_y), 3, color='red')  # 半徑為3
            self.ax.add_patch(car_circle)
            self.ax.annotate(f"starting angle: {start_angle}°", 
                            (start_x, start_y), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')

        # self.ax.legend()
        self.canvas.draw()

    def kmeans(self, X, k, max_iters=100):
        """
        K-means 聚類算法
        X: 輸入數據
        k: 群數
        max_iters: 最大迭代次數
        return: centers, std (中心點和標準差)
        """
        # 隨機選擇初始中心點
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, k, replace=False) # 隨機選擇 k 個樣本
        centers = X[idx] # 初始中心點
        
        for _ in range(max_iters):
            # 計算每個數據點到各個中心的距離
            distances = np.array([np.linalg.norm(X - center, axis=1) for center in centers])
            # 分配每個點到最近的中心
            labels = np.argmin(distances, axis=0)
            
            # 更新中心點
            new_centers = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                  else centers[i] for i in range(k)])
            
            # 檢查是否收斂
            if np.allclose(centers, new_centers):
                break
                
            centers = new_centers
        
        # 計算標準差（使用平均距離）
        distances = np.array([np.linalg.norm(X - center, axis=1) for center in centers])
        std = np.mean(distances)
        
        return centers, std

    def rbf(self, x, center, sigma):
        """
        高斯基底函數
        """
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))

    def calculate_phi_matrix(self, X, centers, sigma):  
        """
        計算 Phi 矩陣
        Phi = [1 phi_1(x_1) ... phi_J(x_1)
               ...
               1 phi_1(x_i) ... phi_J(x_i)
               ...
               1 phi_1(x_N) ... phi_J(x_N)]
        """
        n_samples = X.shape[0]  # N 筆資料
        n_centers = centers.shape[0]  # J 個中心點
        
        # 創建 Phi 矩陣，維度為 N*(J+1)，包含偏置項
        Phi = np.ones((n_samples, n_centers + 1))  # 第一列為 1 (偏置項)
        
        # 計算每個 RBF 的輸出
        for i in range(n_samples):
            for j in range(n_centers):
                Phi[i, j + 1] = self.rbf(X[i], centers[j], sigma)
        
        return Phi

    def train_model(self): 
        """
        訓練 RBFN 模型
        使用 Phi*W = y
        W = [theta, w_1, ..., w_J]^T
        """
        print("Training RBFN model ====================================================================")
        
        # 重新繪製軌道並顯示起始位置
        self.draw_track(show_start_position=True)
        
        selected_file = self.training_file_var.get()
        if not selected_file:
            print("請選擇訓練檔案！")
            return
        
        file_path = f"./{selected_file}"
        data = np.loadtxt(file_path)
        
        # 分離特徵和標籤
        self.input_dim = 3  # 預設為3維輸入
        if selected_file == "train4dAll.txt":
            X = data[:, :3]  # 前方距離、右方距離、左方距離
            y = data[:, 3]   # 方向盤角度
        else:  # train6dAll.txt
            self.input_dim = 5  # 設置為5維輸入
            X = data[:, :5]  # X座標、Y座標、前方距離、右方距離、左方距離
            y = data[:, 5]   # 方向盤角度
        
        # 使用 K-means 初始化中心點和標準差
        """ 設定群數 ========================================================================================"""
        k = 5

        self.centers, self.sigma = self.kmeans(X, k)
        
        # 計算 Phi 矩陣 (N*(J+1))
        Phi = self.calculate_phi_matrix(X, self.centers, self.sigma)
        
        # 使用虛擬反矩陣法計算權重 W = (Phi^T * Phi)^(-1) * Phi^T * y
        Phi_plus = np.linalg.pinv(Phi)  # ((J+1)*N)
        self.weights = np.dot(Phi_plus, y)  # ((J+1)*1)維
        
        print("模型訓練完成！")
        print(f"Centers shape: {self.centers.shape}")
        print(f"Weights shape: {self.weights.shape}")
        print(f"sigma: {self.sigma}")

    def calculate_distances(self, car_pos, car_angle):
        """
        計算車子和軌道邊界的距離
        car_pos: (x, y) 車子當前位置
        car_angle: 車子當前角度
        return: (前方距離, 右45度距離, 左45度距離)
        """
        x, y = car_pos
        angle_rad = np.radians(car_angle)
    
        # 感測器角度(相對於車子方向)
        angles = [0, -45, 45]  # front, right, left
        distances = []
    
        for sensor_angle in angles:
            # 計算感測器絕對角度
            abs_angle = angle_rad + np.radians(sensor_angle)
            
            # 創建感測器射線
            max_distance = 50
            ray_points = np.array([
                [x, y],
                [x + max_distance * np.cos(abs_angle),
                y + max_distance * np.sin(abs_angle)]
            ])
            
            min_dist = max_distance
            
            # 檢查射線與每個牆段是否相交
            for i in range(len(self.track_points) - 1):
                wall = np.array([self.track_points[i], self.track_points[i + 1]])
                
                # 線段相交計算
                x1, y1 = ray_points[0]
                x2, y2 = ray_points[1]
                x3, y3 = wall[0]
                x4, y4 = wall[1]
                
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator == 0:  # 若平行則跳過
                    continue
                    
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                
                if 0 <= t <= 1 and 0 <= u <= 1:  # 若相交則計算距離
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_y = y1 + t * (y2 - y1)
                    dist = np.sqrt((x - intersection_x)**2 + (y - intersection_y)**2)
                    min_dist = min(min_dist, dist)
        
            distances.append(min_dist)
        print(distances)
        print()
        return distances

    def predict_steering_angle(self, distances):
        """
        使用訓練好的 RBFN 模型預測方向盤角度
        distances: (前方距離, 右45度距離, 左45度距離)
        return: 預測的方向盤角度
        """
        if self.centers is None or self.weights is None or self.sigma is None:
            print("模型尚未訓練！")
            return 0
        
        # 根據訓練資料的維度調整輸入
        if self.input_dim == 5:  # 6D資料集
            x = np.array([*self.current_pos, *distances]).reshape(1, -1)  # 加入當前位置
        else:  # 4D資料集
            x = np.array(distances).reshape(1, -1)
        
        # 計算 Phi 矩陣 (1*(J+1))
        phi = np.ones(len(self.centers) + 1)  # 包含偏置項
        for j in range(len(self.centers)):
            phi[j + 1] = self.rbf(x[0], self.centers[j], self.sigma)
        
        # 預測角度 y = Phi * W
        predicted_angle = np.dot(phi, self.weights)
        return float(predicted_angle)

    def run_simulation(self):  
        """
        執行模擬並繪製車輛軌跡
        """
        print("Running simulation =====================================================================")
        
        # 初始化車輛位置和角度
        self.current_pos = (self.start_position[0], self.start_position[1])
        car_angle = self.start_position[2]
        phi = np.radians(car_angle)
        
        # 記錄車輛軌跡
        trajectory_x = [self.current_pos[0]]
        trajectory_y = [self.current_pos[1]]
        
        # 記錄移動數據
        track_records = []
        
        # 車輛參數
        b = 6  # 車輛軸距
        # 清除起始位置的標記
        self.draw_track(show_start_position=False)
        
        while True:
            # 計算距離
            distances = self.calculate_distances(self.current_pos, np.degrees(phi))
            
            # 更新距離顯示
            distance_str = f'front_dist: {distances[0]:.2f}  right_dist: {distances[1]:.2f}  left_dist: {distances[2]:.2f}'
            self.distance_label.config(text=distance_str)
            
            # 預測方向盤角度
            theta = self.predict_steering_angle(distances)
            
            # 記錄當前狀態
            if self.input_dim == 5:  # 6D
                record = [self.current_pos[0], self.current_pos[1], 
                         distances[0], distances[1], distances[2], theta]
            else:  # 4D
                record = [distances[0], distances[1], distances[2], theta]
            track_records.append(record)
            
            # 更新車輛位置和角度
            theta_rad = np.radians(theta)
            
            # 更新車輛位置和角度 (使用運動方程式)
            self.current_pos = (
                self.current_pos[0] + np.cos(phi + theta_rad) + np.sin(theta_rad) * np.sin(phi),
                self.current_pos[1] + np.sin(phi + theta_rad) - np.sin(theta_rad) * np.cos(phi)
            )
            # 更新車輛朝向角度
            phi -= np.arcsin((2 * np.sin(theta_rad)) / b)   # here is radius
            
            # 記錄軌跡
            trajectory_x.append(self.current_pos[0])
            trajectory_y.append(self.current_pos[1])
            
            # 更新圖形
            self.draw_track(show_start_position=False)
            # 繪製車輛軌跡
            # self.ax.plot(trajectory_x, trajectory_y, 'r-', label='Car Trajectory')
            self.ax.plot(trajectory_x, trajectory_y, 'r-')  # 移除 label='Car Trajectory'
            # 繪製當前車輛位置
            car_circle = plt.Circle(self.current_pos, 3, color='red')
            self.ax.add_patch(car_circle)
            # 繪製車輛朝向
            direction_length = 5
            direction_end = (
                self.current_pos[0] + direction_length * np.cos(phi),
                self.current_pos[1] + direction_length * np.sin(phi)
            )
            self.ax.plot([self.current_pos[0], direction_end[0]], 
                        [self.current_pos[1], direction_end[1]], 
                        'k-', linewidth=1)
            
            self.canvas.draw()
            
            # 檢查是否到達終點
            if self.check_end_zone(self.current_pos):
                print("到達終點！")
                # 保存移動記錄
                self.save_track_records(track_records)
                break
                
            # 檢查是否碰撞邊界
            if self.check_collision(self.current_pos):
                print("碰撞邊界！")
                break
            
            # 暫停一小段時間以便觀察
            self.root.update()
            self.root.after(20)  # 50ms 延遲

    def check_end_zone(self, pos):
        """檢查是否到達終點區域"""
        x, y = pos
        return (18 <= x <= 30) and (37 <= y <= 40)

    def check_collision(self, pos):
        """���查是否碰撞���界"""
        x, y = pos
        # 檢查是否碰撞軌道邊界
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i + 1]
            # 計算點到線段的距離
            distance = self.point_to_line_distance(pos, p1, p2)
            if distance < 3:  # 假設車輛半徑為3
                return True
        return False

    def point_to_line_distance(self, point, line_start, line_end):
        """計算點到線段的距離"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 線段長度的平方
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        
        # 點到線的投影位置參數 t
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))  # 投影點在線段上
        
        # 投影點
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        # 計算距離
        return np.sqrt((x - projection_x)**2 + (y - projection_y)**2)

    def save_track_records(self, records):
        """保存移動記錄到檔案"""
        # 確定檔案名稱
        filename = "track6D.txt" if self.input_dim == 5 else "track4D.txt"
        filepath = f"./{filename}"
        
        # 格式化並保存數據
        with open(filepath, 'w') as f:
            for record in records:
                # 格式化每個數值為7位小數
                formatted_record = " ".join([f"{x:.7f}" for x in record])
                f.write(formatted_record + "\n")
        
        print(f"移動記錄已保存到 {filename}")
    
    def plot_dataset_trajectory(self):
        """
        讀取選擇的資料集，並繪製其方向盤角度對應的軌跡
        包含動態顯示和停止條件
        """
        selected_file = self.training_file_var.get()
        if not selected_file:
            print("請選擇訓練檔案！")
            return
        
        # file_path = os.path.join(os.path.dirname(__file__), selected_file)
        file_path = os.path.join(os.path.dirname(__file__), "track4D.txt")
        data = np.loadtxt(file_path)
        
        # 取出方向盤角度
        theta_list = data[:, -1]
        
        # 初始化車輛位置和角度
        car_pos = (self.start_position[0], self.start_position[1])
        phi = np.radians(self.start_position[2])
        
        # 記錄軌跡
        trajectory_x = [car_pos[0]]
        trajectory_y = [car_pos[1]]
        b = 6  # 車輛軸距
        
        # 清除之前的圖形並重新繪製軌道
        self.draw_track()
        
        # 動態顯示軌跡
        for theta in theta_list:
            theta_rad = np.radians(theta)
            
            # 更新車輛位置和角度
            car_pos = (
                car_pos[0] + np.cos(phi + theta_rad) + np.sin(theta_rad) * np.sin(phi),
                car_pos[1] + np.sin(phi + theta_rad) - np.sin(theta_rad) * np.cos(phi)
            )
            phi -= np.arcsin(2 * np.sin(theta_rad) / b)
            
            # 記錄軌跡
            trajectory_x.append(car_pos[0])
            trajectory_y.append(car_pos[1])
            
            # 更新圖形
            self.draw_track()
            # 繪製軌跡
            self.ax.plot(trajectory_x, trajectory_y, 'g-', label='Dataset Trajectory')
            # 繪製當前車輛位置
            car_circle = plt.Circle(car_pos, 3, color='green')
            self.ax.add_patch(car_circle)
            # 繪製車輛朝向
            direction_length = 5
            direction_end = (
                car_pos[0] + direction_length * np.cos(phi),
                car_pos[1] + direction_length * np.sin(phi)
            )
            self.ax.plot([car_pos[0], direction_end[0]], 
                        [car_pos[1], direction_end[1]], 
                        'k-', linewidth=1)
            
            self.ax.legend()
            self.canvas.draw()
            
            # 檢查是否到達終點
            if self.check_end_zone(car_pos):
                print("到達終點！")
                break
                
            # 檢查是否碰撞邊界
            if self.check_collision(car_pos):
                print("碰撞邊界！")
                break
            
            # 暫停一小段時間以便觀察
            self.root.update()
            self.root.after(50)  # 50ms 延遲


# 創建主窗口
root = tk.Tk()
app = AutonomousCarSimulator(root)
root.mainloop()
# input("please input any key to exit!")
