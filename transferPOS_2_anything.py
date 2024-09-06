import datetime
import pandas as pd
import numpy as np
import plotly.express as plty
import plotly.graph_objects as go
import scipy.ndimage as ndimage
import scipy.fft as fft

from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors

class Toolbox:
    def __init__(self) -> None:
        print("Function of this toolbox:\n"+
              "  |--read_measure_file(filename)\n"+
              "     ->DataFrame\n"+
              "  |--diff(data, time_interval)\n"+
              "     ->diffed data\n"+
              "  |--fanuc_rfft(data, time_interval)\n"+
              "     ->[dB, freq]\n"+
              "  |--window(data, range)\n"+
              "     ->A series of data in the range\n"+
              "  |--find_natual_freq(dB, freq)\n"+
              "     ->return the freq which related highest dB\n"
              "  |--find_angle_by_point(x, y):\n"+
              "     ->calculate angle from (0, 0) to point\n"
              "  |--kNN(data: 1-D DataFrame, k, target_value)\n"+
              "     ->return nearest index\n"+
              "  |--plot_spike_respectively(feedrate_list, radius_list):\n"+
              "     ->find spike of quadrant 234 and export plotly html\n")
        self.data = None
        self.time_interval = None                    #確認資料點間隔時間
        self.moving_posc = None
        self.moving_posf = None
        self.VT_data = None
        self.AT_data = None
        self.JT_data = None
        pass
    
    def read_measure_file(self, filename):
        f = pd.read_csv(filename, skiprows=[0, 1, 3], encoding="utf-8")
        print(f"Columns shown below\n{[x for x in f.columns]}")
        return f
    
    def get_key_info_from_df(self, key, threshold = 5):       #自動挑選位移軸向
        try:
            pos_info = self.data
            column_pos = []
            pos_moving = None
            for ele in pos_info.columns:
                if key in ele:
                    column_pos.append(ele)
            if len(column_pos) > 1:
                for pos in column_pos:
                    for ele in pos_info[pos]:
                        if ele > threshold or ele < -threshold:
                            pos_moving = pos_info[pos]
                            break
            else:
                pos_moving = pos_info[key]
            return pos_moving
        except:
            print(f"{key} is not inside")
            return None
        

    def get_posc_and_posf(self):                             #自動選擇位置命令、位置回授
        self.moving_posc = self.get_key_info_from_df("POSC")
        self.moving_posf = self.get_key_info_from_df("POSF")

    def diff(self, data, time_interval):                              #(x2-x1)/(t2-t1)
        new_data = np.array([0.])
        for i in range(1, len(data)):
            new_data = np.append(new_data, (data[i]-data[i-1])/time_interval)
        new_data = ndimage.uniform_filter(new_data, 5)            #計算加速度時要進行濾波
        return new_data
    
    def fanuc_rfft(self, data, time_interval):                              #發那科-傅立葉轉換
        dB = fft.rfft(data)
        dB = np.abs(fft.rfft(data, norm="forward"))*2  #核心函數，正規化方式另外乘以2即可對應發那科頻響
        freq = fft.rfftfreq(len(data), time_interval)
        return dB, freq
    
    def window(self, data, _range):
        lower_bound = _range[0]
        upper_bound = _range[1]
        print(f"Catch data from {lower_bound} to {upper_bound}")
        try:
            temp = []
            index = []
            count = 0
            for i in data:
                count += 1
                if i >= lower_bound and i <= upper_bound:
                    index.append(count)
                    temp.append(i)
            return temp
        except:
            return ValueError("Something is weird.")
    
    def find_natual_freq(self, dB, freq):
        peaks, _ = find_peaks(dB)
        print(f"Natual freq is {freq[peaks][np.argmax(dB[peaks])]}")
        return freq[peaks][np.argmax(dB[peaks])]
    
    def find_angle_by_point(self, x, y):
        degree = np.arctan2(y, x)
        degree *= 180
        degree /= np.pi
        return degree
    
    def kNN(self, data:pd.DataFrame, k, target_value):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data.values)
        distances, indices = nbrs.kneighbors([[target_value]])

        print(f"Nearest point is {[data[x] for x in indices]}")
        print(f"return index {indices}")
        return indices.squeeze()

    def plot_spike_respectively(self, feedrate_list, radius_list):
        Quadrant2 = go.Figure()
        Quadrant3 = go.Figure()
        Quadrant4 = go.Figure()
        file_list = []
        for f in feedrate_list:
            for r in radius_list:
                file_list.append(f"F{f}_R{r}.csv")
        for filename in file_list:
            #檔名擷取進給與半徑
            r = int(filename[filename.find("_R")+len("_R"):filename.rfind(".csv")])
            f = int(filename[filename.find("F")+len("F"):filename.rfind("_R")])
            #資料讀取與整理
            fric_data = self.read_measure_file(filename)
            posf_xy = fric_data[["CH1:POS3D", "CH2:POS3D"]]
            posf_xy["CH1:POS3D"] += r
            
            #計算角度
            rad = np.arctan2(posf_xy["CH2:POS3D"], posf_xy["CH1:POS3D"])  #y, x
            degree = rad*180/np.pi
            degree[degree < 0]+=360
            #擷取對應角度區間線段
            temp_indices = []
            length = (posf_xy["CH1:POS3D"]**2+posf_xy["CH2:POS3D"]**2)**(1/2)
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(pd.DataFrame(degree))
            for i in [270, 180, 90, 30]:
                distances, indices = nbrs.kneighbors([[i]])
                temp_indices.extend(indices.squeeze().tolist())        
            temp_indices.sort()
            Q2 =length[temp_indices[2]:temp_indices[3]]
            Q3 =length[temp_indices[1]:temp_indices[2]]
            Q4 =length[temp_indices[0]:temp_indices[1]]
            #針對3象限繪圖
            Quadrant2.add_trace(go.Scatter(x=[0.001*x for x in range(len(Q2))], y = Q2, name=f"F{f}"))
            Quadrant3.add_trace(go.Scatter(x=[0.001*x for x in range(len(Q3))], y = Q3, name=f"F{f}"))
            Quadrant4.add_trace(go.Scatter(x=[0.001*x for x in range(len(Q4))], y = Q4, name=f"F{f}"))
        Quadrant2.show()
        Quadrant3.show()
        Quadrant4.show()
        today = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"
        Quadrant2.write_html(f"{today}_Quadrant2.html")
        print("Quadrant2 exported")
        Quadrant3.write_html(f"{today}_Quadrant3.html")
        print("Quadrant3 exported")
        Quadrant4.write_html(f"{today}_Quadrant4.html")
        print("Quadrant4 exported")