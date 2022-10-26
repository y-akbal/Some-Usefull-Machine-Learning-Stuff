# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:58:31 2022

@author: yildirim.akbal
"""

class time_series_slices:
    def __init__(self, length = 64):
        self.length = length
    ##     
    ##
    def __return_preprocessed_data(self, data__):
        """
        args:
        data_ = here assumed to be list of pd dataframes
        
        """

        data_ = [data.interpolate() for data in data__] ### get rid of nan values.
        cls_list = []  #### class values for different time series.
        total_length = sum([len(data) for data in data_]) ### overall length to be used by endog and exog.
        X = []
        y = []
        for j,data in enumerate(data_):
            for i in range(len(data)-self.length):
                X.append(np.array(data[i:i+self.length]))
                y.append(np.array(data[i+self.length]))
                cls_list.append(j)
        return np.array(X), np.array(y), np.array(cls_list)
    ##
    ##
    def __split_data(self, data_, alpha):
        lens = [len(data) for data in data_]
        N_len = [int(alpha*N) for N in lens]
        data_train = [data[:N] for N, data in zip(N_len, data_)]
        data_test = [data[N:] for N, data in zip(N_len, data_)]
        for data in data_test:
            data.index = [i for i in range(len(data))]
        return data_train, data_test
    ##
    ##
    def fit_transform(self,data_, alpha = 0.8):
        assert isinstance(data_, list)
        assert all([isinstance(data, pd.Series) for data in data_])
        
        data_train, data_test = self.__split_data(data_, alpha)
        return self.__return_preprocessed_data(data_train), self.__return_preprocessed_data(data_test)