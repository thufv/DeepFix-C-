"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from util.helpers import make_dir_if_not_exists
import os, sys

def _shuffle(items):
    np.random.shuffle(items)
    return items

class load_data:
    def __init__(self, fold, shuffle=True):
        data_folder  = os.path.join('network_inputs')
        fold_folder  = os.path.join(data_folder, 'fold_%d' % fold)
        self.tl_dict = np.load(os.path.join(data_folder, 'translate_dict.npy')).item()

        if not shuffle:
            # Load originals
            self.train_x = np.load(os.path.join(fold_folder, 'mutated-train.npy'))
            self.train_y = np.load(os.path.join(fold_folder, 'fixes-train.npy'))
            self.valid_x = np.load(os.path.join(fold_folder, 'mutated-validation.npy'))
            self.valid_y = np.load(os.path.join(fold_folder, 'fixes-validation.npy'))
            self.test_x = np.load(os.path.join(fold_folder, 'mutated-test.npy'))
            self.test_y = np.load(os.path.join(fold_folder, 'fixes-test.npy'))
            
        else:
            try:
                self.train_x = np.load(os.path.join(fold_folder, 'shuffled/mutated-train.npy'))
                self.train_y = np.load(os.path.join(fold_folder, 'shuffled/fixes-train.npy'))
                self.valid_x = np.load(os.path.join(fold_folder, 'shuffled/mutated-validation.npy'))
                self.valid_y = np.load(os.path.join(fold_folder, 'shuffled/fixes-validation.npy'))
                self.test_x = np.load(os.path.join(fold_folder, 'shuffled/mutated-test.npy'))
                self.test_y = np.load(os.path.join(fold_folder, 'shuffled/fixes-test.npy'))
                
                print "Successfully loaded shuffled data."
                sys.stdout.flush()
            
            # If not generate it
            except IOError:
                print "Generating shuffled data..."
                sys.stdout.flush()
    
                # Load originals
                self.train_x = np.load(os.path.join(fold_folder, 'mutated-train.npy'))
                self.train_y = np.load(os.path.join(fold_folder, 'fixes-train.npy'))
                self.valid_x = np.load(os.path.join(fold_folder, 'mutated-validation.npy'))
                self.valid_y = np.load(os.path.join(fold_folder, 'fixes-validation.npy'))
                self.test_x = np.load(os.path.join(fold_folder, 'mutated-test.npy'))
                self.test_y = np.load(os.path.join(fold_folder, 'fixes-test.npy'))
                
                # Shuffle
                self.train_x, self.train_y = zip(*_shuffle(zip(list(self.train_x), list(self.train_y))))
                self.valid_x, self.valid_y = zip(*_shuffle(zip(list(self.valid_x), list(self.valid_y))))
                self.test_x, self.test_y = zip(*_shuffle(zip(list(self.test_x), list(self.test_y))))
    
                # Convert to np array
                self.train_x, self.train_y = np.array(self.train_x), np.array(self.train_y)
                self.valid_x, self.valid_y = np.array(self.valid_x), np.array(self.valid_y)
                self.test_x, self.test_y = np.array(self.test_x), np.array(self.test_y)
            
                # Save for later
                make_dir_if_not_exists(os.path.join(fold_folder, 'shuffled'))
                
                np.save(os.path.join(fold_folder, 'shuffled/mutated-train.npy'), self.train_x)
                np.save(os.path.join(fold_folder, 'shuffled/fixes-train.npy'), self.train_y)
                np.save(os.path.join(fold_folder, 'shuffled/mutated-validation.npy'), self.valid_x)
                np.save(os.path.join(fold_folder, 'shuffled/fixes-validation.npy'), self.valid_y)
                np.save(os.path.join(fold_folder, 'shuffled/mutated-test.npy'), self.test_x)
                np.save(os.path.join(fold_folder, 'shuffled/fixes-test.npy'), self.test_y)
    
        # Check
        assert(len(self.train_x) == len(self.train_y))
        assert(len(self.valid_x) == len(self.valid_y))
        assert(len(self.test_x)  == len(self.test_y))

    def get_data(self):
        return self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y

    def get_dictionary(self):
        return self.tl_dict

    def get_in_seq_length(self):
        return np.shape(self.train_x)[1]

    def get_out_seq_length(self):
        return np.shape(self.train_y)[1]

    def get_vocabulary_size(self):
        return len(self.tl_dict)
