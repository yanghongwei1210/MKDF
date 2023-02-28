import numpy as np
import torch
from Config import class_num
def convert_to_onehot(sca_label, class_num=4):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=4):
        # print('s_label',s_label.size())#[128])
        # print('t_label',t_label.size())#([128, 4])
        # exit()
        batch_size = s_label.size()[0]
        # print('batch_size',batch_size)
        s_sca_label = s_label.cpu().data.numpy().astype('int64')
        # print('s_sca_label',s_sca_label)
        s_vec_label = convert_to_onehot(s_sca_label)
        # print('s_vec_label',s_vec_label)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        # print('s_sum',s_sum)
        s_sum[s_sum == 0] = 100
        # print('s_sum',s_sum)
        s_vec_label = s_vec_label / s_sum
        # print('s_vec_label',s_vec_label)

        # print('t_label',t_label)
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        ###t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        # print('t_vec_label',t_vec_label)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        # print('t_sum',t_sum)
        t_vec_label = t_vec_label / t_sum
        # print('t_vec_label',t_vec_label)

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))
        # print('weight_ss',weight_ss)
        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        # print('set_s',set_s)
        # print('set_t',set_t)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                # print('s_tvec',s_tvec)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                # print('t_tvec',t_tvec)
                ss = np.dot(s_tvec, s_tvec.T)
                # print('ss',ss)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1
        # print('weight_st',weight_st)
        length = count  # len( set_s ) * len( set_t )
        # print('length',length)
        if length != 0:#####算出来是这个类和那个类，就只拉近这两个类的距离。
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        # print('weight_st',weight_st)
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
