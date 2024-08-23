import pandas as pd
import numpy as np
import plotly.express as plty
import plotly.graph_objects as go
import scipy.ndimage as ndimage
import scipy.fft as fft

class Single_Axis:
    def __init__(self) -> None:
        print("Function of this toolbox:\n"+
              "  |--read_measure_file(filename)\n"+
              "     ->DataFrame\n"+
              "  |--diff(data)\n"+
              "     ->diffed data\n"+
              "  |--fanuc_rfft(data, time_interval)\n"+
              "     ->[dB, freq]\n"+
              "  |--window(data, range)\n"+
              "     ->A series of data in the range\n")
        self.data = None
        self.time_interval = None                    #確認資料點間隔時間
        self.moving_posc = None
        self.moving_posf = None
        self.VT_data = None
        self.AT_data = None
        self.JT_data = None
        pass
    
    def read_measure_file(self, filename):
        return pd.read_csv(filename, skiprows=[0, 1, 3], encoding="utf-8")
    
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

    def diff(self, data):                              #(x2-x1)/(t2-t1)
        new_data = np.array([0.])
        for i in range(1, len(data)):
            new_data = np.append(new_data, (data[i]-data[i-1])/self.time_interval)
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