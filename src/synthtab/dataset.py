import pandas as pd
import numpy as np
from rich import print
import torch
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, config) -> None:
        print('🔄 Loading dataset...')

        self.data = pd.read_csv(config['path'])
        self.X_cat = None
        self.X_num = {}
        self.X_num['train'] = self.data.to_numpy()
        y_data = pd.read_csv(config['path_y'])['#play']
        LE = LabelEncoder()
        y_data['code'] = LE.fit_transform(y_data)
        self.y = {}
        self.y['train'] = y_data['code']
        print(self.y['train'].shape)
        print(self.X_num['train'].shape)
        #train = reduce_mem_usage(train)
        #self.data["#play"] = pd.read_csv('~/ml-workspace/PlayNet/handball_y_train.csv')["#play"]

        print('✅ Dataset loaded...')

    def reduce_mem(self) -> None:
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        start_mem = self.data.memory_usage().sum() / 1024**2
        print('🔄 Reducing memory usage...')
        print('💾 Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in self.data.columns:
            col_type = self.data[col].dtype
            
            if col_type != object:
                c_min = self.data[col].min()
                c_max = self.data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.data[col] = self.data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.data[col] = self.data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.data[col] = self.data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.data[col] = self.data[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.data[col] = self.data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.data[col] = self.data[col].astype(np.float32)
                    else:
                        self.data[col] = self.data[col].astype(np.float64)
            else:
                self.data[col] = self.data[col].astype('category')

        end_mem = self.data.memory_usage().sum() / 1024**2
        print('💾 Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('✅ Reduced by {:.1f}%...'.format(100 * (start_mem - end_mem) / start_mem))
    
    def __get_category_sizes__(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
        XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
        return [len(set(x)) for x in XT]

    def get_category_sizes(self):
        return [] if self.X_cat is None else self.__get_category_sizes__(self.X_cat)
        